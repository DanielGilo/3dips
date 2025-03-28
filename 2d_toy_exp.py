import torch
from torch.optim.sgd import SGD
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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] =":4096:8" # necessary when setting torch.use_dererministic_algorithms(True)

device = torch.device('cuda:0')

output_log = {"z": [], "z_t": [], "pred_z0": [], "t": []}

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
    plt.show()


def get_timestep_linear_interp(i, num_iters, t_min, t_max):
    i = max(i, 0)
    t = int((1 - (i / num_iters)) * t_max + (i / num_iters) * t_min)
    return t


def set_deterministic_state():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)


def image_optimization(teacher, text_target, num_iters=200, guidance_scale=7.5, lr=1e-1, do_mask_and_flip=False):
    config = {"text_target": text_target,  "num_iters": num_iters,
              "teacher_guidance_scale": guidance_scale, "lr": lr}
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
    optimizer = SGD(params=student.get_trainable_params(), lr=lr)

    if do_mask_and_flip:
        masks = get_masks(latent_shape)
    else:
        mask = None

    for i in range(num_iters):
        z0_student = student.predict_sample()
        z0_student_orig = z0_student.clone()

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

        eps = torch.randn(latent_shape, device=device)
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
            plt.imshow(out); plt.show()
    show_output_log(wb, teacher)


def turbo_sd(teacher, text_target, num_iters=200, guidance_scale=100, lr=1e-3, do_mask_and_flip=False):
    config = {"text_target": text_target, "num_iters": num_iters,
              "teacher_guidance_scale": guidance_scale, "lr": lr}
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

    t_min = 50
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
            eps_pred = eps_pred * mask
            eps_pred = eps_pred + torch.flip(eps_pred, dims=(-1,))


        t_min_ = int((1 - (i/num_iters)) * t_max + (i/num_iters) * t_min)
        timestep = torch.randint(low=t_min_, high=t_max+1, size=(1,)).item()
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
            plt.show()
    show_output_log(wb, student)


def ddim_like_optimization_loop(optimizer, num_iters, student, student_prompt, student_guidance_scale,
                                teacher, teachers_prompts, teacher_guidance_scale, t_min, t_max, z_t, eta,
                                show_interval, wb, do_mask_and_flip=False):
    with torch.no_grad():
        embedding_null = teacher.get_text_embeddings("")
        embeddings_text_teacher = [teacher.get_text_embeddings(text) for text in teachers_prompts]
        teacher_texts_embeddings = [torch.stack([embedding_null, e], dim=1) for e in embeddings_text_teacher]
        student_text_embeddings = torch.stack([student.get_text_embeddings(""), student.get_text_embeddings(student_prompt)],
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
            eps_pred = eps_pred * mask
            eps_pred = eps_pred + torch.flip(eps_pred, dims=(-1,))


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
            plt.show()

    show_output_log(wb, student)


def SD(teacher, texts_target, text_source, num_iters=200, guidance_scale=7.5, student_guidance_scale=7.5, lr=1e-3, eta=0.0,
       do_mask_and_flip=False):
    config = {"texts_target": texts_target, "text_source": text_source, "num_iters": num_iters,
              "teacher_guidance_scale": guidance_scale, "lr": lr, "eta": eta}
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
    ddim_like_optimization_loop(optimizer, num_iters, student, text_source, student_guidance_scale,
                                teacher, texts_target, guidance_scale, t_min, t_max, z_t, eta,
                                show_interval, wb, do_mask_and_flip)


def SD_lora(teacher, texts_target, text_source, num_iters=200, guidance_scale=7.5, student_guidance_scale=7.5, lr=1e-3,
            eta=0.0, do_mask_and_flip=False):
    config = {"texts_target": texts_target, "text_source": text_source, "num_iters": num_iters,
                 "teacher_guidance_scale": guidance_scale, "lr": lr, "eta": eta}
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
    ddim_like_optimization_loop(optimizer, num_iters, student, text_source, student_guidance_scale,
                                teacher, texts_target, guidance_scale, t_min, t_max, z_t, eta,
                                show_interval, wb, do_mask_and_flip)


def get_masks(shape):
    mask1 = torch.ones(shape, dtype=torch.float32, device=device)
    mask2 = torch.ones(shape, dtype=torch.float32, device=device)
    mask1[..., :(shape[-1] // 2)] = 0.0
    mask2[..., (shape[-1] // 2):] = 0.0

    return [mask1, mask2]


if __name__ == '__main__':

    # SD teacher

    #teacher = teachers.Teacher(model_id="stabilityai/stable-diffusion-2-1", device=device, dtype=torch.float32)
    #teachers_prompt = ["a photorealistic image of a man riding a horse"]

    # Inpainting teacher

    # init_image = PIL.Image.open("example_img.png").convert("RGB").resize((512, 512))
    # mask_image = PIL.Image.open("example_mask.png").convert("RGB").resize((512, 512))
    # teacher = teachers.InpaintingTeacher(init_image=init_image, mask_image=mask_image,
    #                                      model_id="runwayml/stable-diffusion-inpainting", device=device,
    #                                      dtype=torch.float32)
    #teachers_prompt = ["a photo of a striped cat sitting on a bench, facing the camera, high resolution"]

    # Controlnet canny teacher

    # init_image = PIL.Image.open("dog.png").convert("RGB").resize((512, 512))
    # canny_image = get_canny_image(init_image)
    # teacher = teachers.ControlNetTeacher(cond_image=canny_image, controlnet_id="lllyasviel/sd-controlnet-canny",
    #                                      model_id="runwayml/stable-diffusion-v1-5", device=device, dtype=torch.float32)
    # teachers_prompt = ["a dog facing the camera, center of the frame"]

    # Controlnet depth teacher

    init_image = PIL.Image.open("bee.png").convert("RGB").resize((512, 512))
    depth_image = get_depth_estimation(init_image)
    teacher = teachers.ControlNetTeacher(cond_image=depth_image, controlnet_id="lllyasviel/sd-controlnet-depth",
                                         model_id="runwayml/stable-diffusion-v1-5", device=device, dtype=torch.float32)
    teachers_prompt = ["a bee on a flower"]

    student_prompt = ""
    set_deterministic_state()
    do_mask_and_flip = True


    image_optimization(teacher, teachers_prompt[0], num_iters=3000, guidance_scale=7.5, lr=1e2,
                       do_mask_and_flip=do_mask_and_flip)
    #turbo_sd(teacher, teachers_prompt[0], num_iters=3000, guidance_scale=7.5, lr=1e-3, do_mask_and_flip=do_mask_and_flip)
    #SD(teacher, teachers_prompt, student_prompt, num_iters=10000, guidance_scale=7.5, student_guidance_scale=7.5,
    #             lr=1e-3, eta=0.0, do_mask_and_flip=do_mask_and_flip)
    #SD_lora(teacher, teachers_prompt, student_prompt, num_iters=10000, guidance_scale=7.5, student_guidance_scale=7.5,
    #               lr=1e-1, eta=0.0, do_mask_and_flip=do_mask_and_flip)

