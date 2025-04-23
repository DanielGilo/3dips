import argparse
import torch
import gc
import numpy as np
import wandb
from torch.optim.sgd import SGD


import teachers
import students
import losses
from seva_utils import get_value_dict_of_scene
from plot_utils import plot_frames_row, plot_logs

from seva.seva.sampling import to_d, append_dims


device = torch.device('cuda:0')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse user arguments for an experiment.")

    parser.add_argument("--teacher_name", type=str, required=True, help="Name of the Teacher (string).")
    parser.add_argument("--student_name", type=str, required=True, help="Name of the Student (string).")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate (float).")
    parser.add_argument("--num_iterations", type=int, required=True, help="Number of iterations (integer).")

    args = parser.parse_args()
    return args


def get_timestep_linear_interp(i, num_iters, t_min, t_max):
    i = max(i, 0)
    t = int((1 - (i / num_iters)) * t_max + (i / num_iters) * t_min)
    return t


with torch.autocast("cuda"):
    args = parse_arguments()
    teacher_name = args.teacher_name
    lr = args.lr
    num_iters = args.num_iterations
    student_name = args.student_name
    text_target = ""
    exp_name = student_name

    config = {"text_target": text_target,  "num_iters": num_iters,  "lr": lr, "teacher_name": teacher_name}
    wb = wandb.init(
        project="3dips",
        name=exp_name,
        config=config
    );
    wb.log_code()

    scene_path = "/home/danielgilo/3dips/seva/assets_demo_cli/dl3d140-165f5af8bfe32f70595a1c9393a6e442acf7af019998275144f605b89a306557"

    value_dict = get_value_dict_of_scene(scene_path)


    # mask_tensor = torch.zeros((T,3,H,W), dtype=torch.uint8)
    # #mask_tensor[1:,:,H//4:3*H//4, W//4:3*W//4] = 255 # generalize to anchor location

    # plot_frames_row(mask_tensor.to(torch.uint8).permute((0,2,3,1)).cpu().numpy(), wb, "masks", "masks")

    # masked_gt = torch_gt.clone()
    # masked_gt[mask_tensor.permute((0,2,3,1)) > 0.5] = 255
    # plot_frames_row(masked_gt.cpu().numpy(), wb, "masked_gt", "masked_gt")

    # torch_gt = torch_gt.permute((0, 3, 1, 2)) # pil expects (C, H, W)
    # pil_gt = [to_pil_image(torch_gt[i]) for i in range(torch_gt.shape[0])]
    # pil_masks = [to_pil_image(mask_tensor[i]) for i in range(mask_tensor.shape[0])]


    
    show_interval = int(num_iters / 10)
    student = students.get_mv_student(student_name, value_dict=value_dict, device=device)
    #teacher = teachers.InpaintingTeacher(init_image=pil_gt, mask_image=pil_masks,
     #                                      model_id="stabilityai/stable-diffusion-2-inpainting", device="cuda",
     #                                      dtype=torch.float32)
    

    teacher = teachers.get_teacher(teacher_name, value_dict, device=device)
    teacher.plot_input(wb)
    

    output_log = {"z": [], "z_t": [], "teacher_pred": [], "t": []}


    

    latent_shape = student.get_latent_shape()
    latent = torch.randn(latent_shape).to("cuda")
    latent *= torch.sqrt(1.0 + student.discretization(1000, device="cuda")[0] ** 2.0) # from seva.sampling prepare_sampling_loop() - not sure why necessary
    s_in = latent.new_ones([latent.shape[0]])

    t_max = 999
    t_min = 20
    T = t_max * s_in # replicate for N_frames
    student_t = T

    text_embeddings = torch.stack([teacher.get_text_embeddings(""), teacher.get_text_embeddings(text_target)], dim=1)
    optimizer = SGD(params=student.get_trainable_parameters(), lr=lr)

    mask = torch.ones(latent_shape, dtype=torch.float32, device="cuda")
    mask[value_dict["cond_frames_mask"], ...] = 0.0 # on conditioning frame


    for i in range(num_iters):
        pred_eps, z0_student = student.predict_eps_and_sample(latent, student_t, 2.0)
        z0_student_orig = z0_student.clone()

       # eps = torch.randn(latent_shape, device="cuda")
        eps = pred_eps

        timestep = torch.stack([torch.tensor(student_t[0], dtype=torch.int)], dim=0).to("cuda")

        w_t = 1
        loss, z_t, pred_z0 = losses.get_sds_loss(z0_student, teacher, text_embeddings, 7.5, eps, 
                            timestep, w_t, mask)

        gc.collect()
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wb.log({"loss": loss})
        wb.log({"lr": optimizer.param_groups[0]['lr']})


        if (i+ 1) % 1 == 0:
            print("iteration: {}/{}, loss: {}".format(i + 1, num_iters, loss.item()))

        if (i % show_interval == 0) or (i == num_iters-1):
            out_frames = student.decode(z0_student_orig)
            z_t_frames = student.decode(z_t)
            pred_z0_frames = student.decode(pred_z0)

            plot_frames_row(out_frames, wb, "z", "t={}".format(timestep.item()))
            plot_frames_row(pred_z0_frames, wb, "teacher_pred", "t={}".format(timestep.item()))
            plot_frames_row(z_t_frames, wb, "z_t", "t={}".format(timestep.item()))
            
            output_log["z"].append(out_frames)
            output_log["z_t"].append(z_t_frames)
            output_log["teacher_pred"].append(pred_z0_frames)
            output_log["t"].append(timestep.item())



        # Euler step to next timestep
        with torch.no_grad():
            next_t = get_timestep_linear_interp(i, num_iters, t_min, t_max) * s_in
            curr_sigma = student.get_sigma_from_timestep(student_t)
            next_sigma = student.get_sigma_from_timestep(next_t)
            d = to_d(latent, curr_sigma, z0_student)
            dt = append_dims(next_sigma - curr_sigma, latent.ndim)
            latent = (latent + dt * d).detach()
            student_t = next_t
    
    plot_logs(wb, output_log)
