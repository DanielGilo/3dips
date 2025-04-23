import torch
from torch.optim.sgd import SGD
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
import numpy as np
import PIL

import matplotlib.pyplot as plt
import gc
import wandb

import students
import teachers
import losses
from controlnet_utils import get_canny_image, get_depth_estimation


import os
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] =":4096:8" # necessary when setting torch.use_dererministic_algorithms(True)

device = torch.device('cuda:0')

output_log = {"z": [], "z_t": [], "pred_z0": [], "t": []}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse user arguments for an experiment.")

    parser.add_argument("--teacher_name", type=str, required=True, help="Name of the Teacher (string).")
    parser.add_argument("--student_name", type=str, required=True, help="Name of the Student (string).")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate (float).")
    parser.add_argument("--do_mask_and_flip", action='store_true', help="Boolean flag to enable mask and flip.")
    parser.add_argument("--num_iterations", type=int, required=True, help="Number of iterations (integer).")

    args = parser.parse_args()
    return args


def show_output_log(wandb, model):
    nrows = len(output_log.keys()) - 1
    ncols = len(output_log["z"])
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, 3))
    for col_i in range(ncols):
        axs[0][col_i].imshow(model.decode(output_log["z"][col_i]))
        axs[1][col_i].imshow(model.decode(output_log["z_t"][col_i]))
        axs[2][col_i].imshow(model.decode(output_log["pred_z0"][col_i]))

        [axs[row_i][col_i].set_xticks([]) for row_i in range(nrows)]
        [axs[row_i][col_i].set_yticks([]) for row_i in range(nrows)]
        axs[0][col_i].set_title("t = {}".format(output_log["t"][col_i]))

    axs[0][0].set_ylabel("z")
    axs[1][0].set_ylabel("z_t")
    axs[2][0].set_ylabel("pred_z0")

    plt.tight_layout()
    wandb.log({"output_log": fig})
    plt.show(block=False)


def get_timestep_linear_interp(i, num_iters, t_min, t_max):
    i = max(i, 0)
    t = int((1 - (i / num_iters)) * t_max + (i / num_iters) * t_min)
    return t


def set_deterministic_state():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)


def image_optimization(teacher, text_target, num_iters, guidance_scale, lr, do_mask_and_flip, teacher_name):
    config = {"text_target": text_target,  "num_iters": num_iters,
              "teacher_guidance_scale": guidance_scale, "lr": lr, "teacher_name": teacher_name}
    wb = wandb.init(
        project="3dips",
        name="blank_canvas",
        config=config
    );
    wb.log_code()
    show_interval = int(num_iters / 10)

    latent_shape = (1,4,64,64)

    student = students.BlankCanvasStudent(latent_shape=latent_shape, device=device)

    text_embeddings = torch.stack([teacher.get_text_embeddings(""), teacher.get_text_embeddings(text_target)], dim=1)
    optimizer = SGD(params=student.get_trainable_parameters(), lr=lr)

    if do_mask_and_flip:
        masks = get_masks(latent_shape)
    else:
        mask = None

    for i in range(num_iters):
        z0_student = student.predict_sample()
        z0_student_orig = z0_student.clone()

        eps = torch.randn(latent_shape, device=device)

        if do_mask_and_flip:
            mask_i = torch.randint(0, len(masks), size=(1,)).item()
            mask = masks[mask_i]
            z0_student = z0_student * mask
            z0_student = z0_student + torch.flip(z0_student, dims=(-1,))

        # WIP! multiple predictions in batch dim
        # preds_student = []
        # for mask in masks:
        #     z0_student = z0_student_orig * mask
        #     z0_student = z0_student + torch.flip(z0_student, dims=(-1,))
        #     preds_student.append(z0_student)
        # preds_student = torch.cat(preds_student, dim=0)
        # mask = torch.cat(masks, dim=0)

        timestep = torch.randint(low=50, high=950,size=(latent_shape[0],), device=device, dtype=torch.long)

        w_t = 1
        loss, z_t, pred_z0 = losses.get_sds_loss(z0_student, teacher, text_embeddings, guidance_scale, eps, timestep, w_t, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wb.log({"loss": loss})
        wb.log({"lr": optimizer.param_groups[0]['lr']})
        if (i+ 1) % 50 == 0:
            print("iteration: {}/{}, loss: {}".format(i + 1, num_iters, loss.item()))

        if i % show_interval == 0:
            output_log["z"].append(z0_student_orig.clone().detach())
            output_log["z_t"].append(z_t.detach())
            output_log["pred_z0"].append(pred_z0.detach())
            output_log["t"].append(timestep.item())

            wb.log({"z": wandb.Image(teacher.decode(z0_student_orig.detach()), caption="z(t={})".format(timestep.item()))})
            wb.log({"z_t": wandb.Image(teacher.decode(z_t.detach()), caption="z_t(t={})".format(timestep.item()))})
            wb.log({"pred_z0": wandb.Image(teacher.decode(pred_z0.detach()), caption="pred_z0(t={})".format(timestep.item()))})

            out = teacher.decode(z0_student_orig)
            plt.imshow(out); plt.show(block=False)
    show_output_log(wb, teacher)


def turbo_sd(teacher, text_target, num_iters, guidance_scale, lr, do_mask_and_flip, exp_name):
    config = {"text_target": text_target, "num_iters": num_iters,
              "teacher_guidance_scale": guidance_scale, "lr": lr, "do_mask_and_flip?": do_mask_and_flip,
              "teacher_name": teacher_name}
    wb = wandb.init(
        project="3dips",
        name="Turbo_SD",
        config=config
    );
    wb.log_code()
    show_interval = int(num_iters / 10)
    latent_shape = (1, 4, 64, 64)

    student = students.SDStudent(model_id="stabilityai/sd-turbo", device=device, dtype=torch.float32)
    teacher_embeddings = torch.stack([teacher.get_text_embeddings(""), teacher.get_text_embeddings(text_target)], dim=1)
    student_embeddings = torch.stack([student.get_text_embeddings(""), student.get_text_embeddings("")], dim=1) # embeddings should be identical

    optimizer = SGD(params=student.get_trainable_parameters(), lr=lr)

    t_min = 200
    t_max = 999
    z_T = torch.randn(latent_shape, device=device, dtype=torch.float32)
    T = torch.stack([torch.tensor(t_max)] * latent_shape[0], dim=0).to(device)

    if do_mask_and_flip:
        masks = get_masks(latent_shape)
    else:
        mask = None

    for i in range(num_iters):
        eps_pred, z0_student = student.predict_eps_and_sample(z_T, T, 0.0, text_embeddings=student_embeddings)
        z0_student_orig = z0_student.clone().detach()

        if do_mask_and_flip:
            mask_i = torch.randint(0, len(masks), size=(1,)).item()
            mask = masks[mask_i]
            z0_student = z0_student * mask
            z0_student = z0_student + torch.flip(z0_student, dims=(-1,))
            # eps_pred = eps_pred * mask
            # eps_pred = eps_pred + torch.flip(eps_pred, dims=(-1,))


        t_min_ = int((1 - (i/num_iters)) * t_max + (i/num_iters) * t_min)
        timestep = torch.randint(low=t_min_, high=t_max+1, size=(1,)).item()
        #timestep = get_timestep_linear_interp(i, num_iters, t_min, t_max)
        timestep = torch.stack([torch.tensor(timestep)] * z0_student.shape[0], dim=0).to(device)
        w_t = 1

        loss, z_t, pred_z0 = losses.get_sds_loss(z0_student, teacher, teacher_embeddings, guidance_scale,
                                                           eps_pred, timestep, w_t, mask)

        gc.collect()
        torch.cuda.empty_cache()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wb.log({"loss": loss})
        wb.log({"lr": optimizer.param_groups[0]['lr']})
        if (i+ 1) % 50 == 0:
            print("iteration: {}/{}, loss: {}".format(i + 1, num_iters, loss.item()))
        if ((i % show_interval) == 0) or (i == (num_iters-1)):
            output_log["z"].append(z0_student_orig.detach())
            output_log["z_t"].append(z_t.detach())
            output_log["pred_z0"].append(pred_z0.detach())
            output_log["t"].append(timestep.item())

            wb.log({"z": wandb.Image(student.decode(z0_student_orig.detach()), caption="z(t={})".format(timestep.item()))})
            wb.log({"z_t": wandb.Image(teacher.decode(z_t.detach()), caption="z_t(t={})".format(timestep.item()))})
            wb.log({"pred_z0": wandb.Image(teacher.decode(pred_z0.detach()), caption="pred_z0(t={})".format(timestep.item()))})

            out = student.decode(z0_student_orig)
            plt.imshow(out);
            plt.show(block=False)
    show_output_log(wb, student)


def ddim_like_optimization_loop(optimizer, num_iters, student, student_guidance_scale,
                                teacher, teachers_prompts, teacher_guidance_scale, t_min, t_max, z_t, eta,
                                show_interval, wb, do_mask_and_flip):
    with torch.no_grad():
        embedding_null = teacher.get_text_embeddings("")
        embeddings_text_teacher = [teacher.get_text_embeddings(text) for text in teachers_prompts]
        teacher_texts_embeddings = [torch.stack([embedding_null, e], dim=1) for e in embeddings_text_teacher]
        student_text_embeddings = torch.stack([student.get_text_embeddings(""), student.get_text_embeddings("")],
                                     dim=1)

    T = torch.stack([torch.tensor(t_max)] * z_t.shape[0], dim=0).to(device)
    timestep = T.clone()
    eta_0 = eta

    if do_mask_and_flip:
        masks = get_masks(z_t.shape)
    else:
        mask = None

    for i in range(num_iters):
        # prev timestep prediction
        with torch.no_grad():
            eps_t, pred_z0 = student.predict_eps_and_sample(z_t, timestep, student_guidance_scale, student_text_embeddings)

        # ddim forward (noising) step to new timestep
        with torch.no_grad():
            next_timestep = timestep.clone()  # timestep from prev iteration, "t + 1"
            timestep = get_timestep_linear_interp(i, num_iters, t_min, t_max)  # new timestep, "t"
            timestep = torch.stack([torch.tensor(timestep)] * z_t.shape[0], dim=0).to(device)
            z_t = student.noise_to_timestep(pred_z0, timestep, eps_t)

        # image prediction - this is "g(theta)" in SDS terms
        eps_pred, z0_student = student.predict_eps_and_sample(z_t, timestep, student_guidance_scale, student_text_embeddings)
        z0_student_orig = z0_student.clone().detach()

        if do_mask_and_flip:
            mask_i = torch.randint(0, len(masks), size=(1,)).item()
            mask = masks[mask_i]
            z0_student = z0_student * mask
            z0_student = z0_student + torch.flip(z0_student, dims=(-1,))
            # eps_pred = eps_pred * mask
            # eps_pred = eps_pred + torch.flip(eps_pred, dims=(-1,))


        # adding stochasticity to the predicted noise, according to Eq. 12, 16 in https://arxiv.org/pdf/2010.02502 (DDIM paper).
        # if eta > 0:
        #     variance = student_pipe.scheduler._get_variance(next_timestep.to('cpu'), timestep.to('cpu')).to(device)
        #     std_dev_t = eta * variance ** 0.5
        #     direction_to_z_t_term = ((1 - alphas_cumprod[timestep] - std_dev_t ** 2) ** 0.5) * eps_pred
        #     stochastic_term = std_dev_t * torch.randn_like(eps_pred)
        #     eps = (direction_to_z_t_term + stochastic_term) / sigma_t
        # else:
        #     eps = eps_pred

        if eta_0 > 0:
            eta = np.exp(-20 * (i / num_iters)) * eta_0
            eps = ((1 - eta) ** 0.5) * eps_pred + (eta ** 0.5) * torch.randn_like(eps_pred)
        else:
            eps = eps_pred

        i_target = torch.randint(0, len(teacher_texts_embeddings), size=(1,)).item()
        w_t = 1
        loss, z_t, pred_z0 = losses.get_sds_loss(z0_student, teacher, teacher_texts_embeddings[i_target],
                                                           teacher_guidance_scale, eps, timestep, w_t, mask)
        gc.collect()
        torch.cuda.empty_cache()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wb.log({"loss": loss})
        wb.log({"lr": optimizer.param_groups[0]['lr']})
        if (i + 1) % 50 == 0:
            print("iteration: {}/{}, loss: {}".format(i + 1, num_iters, loss.item()))
        if ((i % show_interval) == 0) or (i == (num_iters - 1)):
            output_log["z"].append(z0_student_orig.detach())
            output_log["z_t"].append(z_t.detach())
            output_log["pred_z0"].append(pred_z0.detach())
            output_log["t"].append(timestep.item())

            wb.log({"z": wandb.Image(student.decode(z0_student_orig.detach()), caption="z(t={})".format(timestep.item()))})
            wb.log({"z_t": wandb.Image(teacher.decode(z_t.detach()), caption="z_t(t={})".format(timestep.item()))})
            wb.log({"pred_z0": wandb.Image(teacher.decode(pred_z0.detach()), caption="pred_z0(t={})".format(timestep.item()))})

            out = student.decode(z0_student_orig)
            plt.imshow(out);
            plt.show(block=False)

    show_output_log(wb, student)


def SD(teacher, texts_target, num_iters, guidance_scale, student_guidance_scale, lr, eta,
       do_mask_and_flip, teacher_name):
    config = {"texts_target": texts_target, "num_iters": num_iters,
              "teacher_guidance_scale": guidance_scale, "lr": lr, "eta": eta, "do_mask_and_flip?": do_mask_and_flip,
              "teacher_name": teacher_name}
    wb = wandb.init(
        project="3dips",
        name="SD",
        config=config
    ); wb.log_code()
    show_interval = int(num_iters / 10)

    latent_shape = (1, 4, 64, 64)

    student = students.SDStudent(model_id="stabilityai/stable-diffusion-2-1", device=device, dtype=torch.float32)

    optimizer = SGD(params=student.get_trainable_parameters(), lr=lr)
    t_min = 20
    t_max = 999

    z_t = torch.randn(latent_shape, device=device, dtype=torch.float32, requires_grad=False)
    ddim_like_optimization_loop(optimizer, num_iters, student, student_guidance_scale,
                                teacher, texts_target, guidance_scale, t_min, t_max, z_t, eta,
                                show_interval, wb, do_mask_and_flip)


def SD_lora(teacher, texts_target, num_iters, guidance_scale, student_guidance_scale, lr,
            eta, do_mask_and_flip, teacher_name):
    config = {"texts_target": texts_target, "num_iters": num_iters,
                 "teacher_guidance_scale": guidance_scale, "lr": lr, "eta": eta, "do_mask_and_flip?": do_mask_and_flip,
                 "teacher_name": teacher_name}
    wb = wandb.init(
        # Set the wandb project where this run will be logged.
        project="3dips",
        name="SD_LoRA",
        config=config
    ); wb.log_code()

    show_interval = int(num_iters / 10)
    latent_shape = (1, 4, 64, 64)
    student = students.SDLoRAStudent(model_id="stabilityai/stable-diffusion-2-1", device=device, dtype=torch.float32)

    optimizer = SGD(params=student.get_trainable_parameters(), lr=lr)
    t_min = 20
    t_max = 999

    z_t = torch.randn(latent_shape, device=device, dtype=torch.float32, requires_grad=False)
    ddim_like_optimization_loop(optimizer, num_iters, student, student_guidance_scale,
                                teacher, texts_target, guidance_scale, t_min, t_max, z_t, eta,
                                show_interval, wb, do_mask_and_flip)

def get_masks(shape):
    mask1 = torch.ones(shape, dtype=torch.float32, device=device)
    mask2 = torch.ones(shape, dtype=torch.float32, device=device)
    mask1[..., :(shape[-1] // 2)] = 0.0
    mask2[..., (shape[-1] // 2):] = 0.0

    #return [mask1, mask2]
    return [mask1]





def train_student(student_name, teacher, teachers_prompt, lr, num_iterations, do_mask_and_flip, teacher_name):
    if student_name == "blank_canvas":
        return image_optimization(teacher, teachers_prompt[0], num_iters=num_iterations, guidance_scale=7.5, lr=lr,
                do_mask_and_flip=do_mask_and_flip, teacher_name=teacher_name)
    if student_name == "turbo-sd":
        return turbo_sd(teacher, teachers_prompt[0], num_iters=num_iterations, guidance_scale=7.5, lr=lr,
                 do_mask_and_flip=do_mask_and_flip, teacher_name=teacher_name)
    if student_name == "sd":
        return SD(teacher, teachers_prompt, num_iters=num_iterations, guidance_scale=7.5, student_guidance_scale=7.5,
                lr=lr, eta=0.0, do_mask_and_flip=do_mask_and_flip, teacher_name=teacher_name)
    if student_name == "sd-lora":
        return SD_lora(teacher, teachers_prompt, num_iters=num_iterations, guidance_scale=7.5, student_guidance_scale=7.5,
                lr=lr, eta=0.0, do_mask_and_flip=do_mask_and_flip, teacher_name=teacher_name)
    raise ValueError("Wrong student_name: {}".format(student_name))


if __name__ == '__main__':
    set_deterministic_state()

    args = parse_arguments()
    teacher_name = args.teacher_name
    lr = args.lr
    num_iterations = args.num_iterations
    do_mask_and_flip = args.do_mask_and_flip
    student_name = args.student_name
    teacher, teachers_prompt = get_teacher(teacher_name)

    train_student(student_name, teacher, teachers_prompt, lr, num_iterations, do_mask_and_flip, teacher_name)





