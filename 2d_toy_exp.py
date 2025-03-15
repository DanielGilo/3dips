# adapted from Delta Denoising Score code: https://github.com/google/prompt-to-prompt/blob/main/DDS_zeroshot.ipynb

from typing import Tuple, Union, Optional, List

import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.optim.sgd import SGD
from torch.optim import lr_scheduler
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoPipelineForText2Image
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from torchvision.transforms.functional import pil_to_tensor
import gc

from dip.models import skip

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] =":4096:8" # necessary when setting torch.use_dererministic_algorithms(True)

T = torch.Tensor
TN = Optional[T]
TS = Union[Tuple[T, ...], List[T]]

device = torch.device('cuda:0')

output_log = {"z": [], "z_t": [], "pred_z0": [], "t": []}

def show_output_log():
    nrows = len(output_log.keys()) - 1
    ncols = len(output_log["z"])
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, 3))
    for col_i in range(ncols):
        axs[0][col_i].imshow(decode(output_log["z"][col_i], pipeline))
        axs[1][col_i].imshow(decode(output_log["z_t"][col_i], pipeline))
        axs[2][col_i].imshow(decode(output_log["pred_z0"][col_i], pipeline))

        [axs[row_i][col_i].set_xticks([]) for row_i in range(nrows)]
        [axs[row_i][col_i].set_yticks([]) for row_i in range(nrows)]
        axs[0][col_i].set_title("t = {}".format(output_log["t"][col_i]))

    axs[0][0].set_ylabel("z")
    axs[1][0].set_ylabel("z_t")
    axs[2][0].set_ylabel("pred_z0")

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def get_text_embeddings(pipe: StableDiffusionPipeline, text: str) -> T:
    tokens = pipe.tokenizer([text], padding="max_length", max_length=77, truncation=True,
                                   return_tensors="pt", return_overflowing_tokens=True).input_ids.to(device)
    return pipe.text_encoder(tokens).last_hidden_state.detach()

@torch.no_grad()
def denormalize(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image[0]


@torch.no_grad()
def decode(latent: T, pipe: StableDiffusionPipeline, im_cat: TN = None):
    image = pipeline.vae.decode((1 / 0.18215) * latent, return_dict=False)[0]
    image = denormalize(image)
    if im_cat is not None:
        image = np.concatenate((im_cat, image), axis=1)
    return Image.fromarray(image)


def init_pipe(device, dtype, unet, scheduler, unet_requires_grad=False) -> Tuple[UNet2DConditionModel, T, T]:
    with torch.inference_mode():
        alphas = torch.sqrt(scheduler.alphas_cumprod).to(device, dtype=dtype)
        sigmas = torch.sqrt(1 - scheduler.alphas_cumprod).to(device, dtype=dtype)
    if not unet_requires_grad:
        for p in unet.parameters():
            p.requires_grad = False
    return unet, alphas, sigmas

def get_eps_prediction_cfg(unet, prediction_type, z_t: T, timestep: T, text_embeddings: T, alpha_t: T, sigma_t: T,
                           get_raw=False, guidance_scale=7.5):
    latent_input = torch.cat([z_t] * 2)
    timestep = torch.cat([timestep] * 2)
    embedd = text_embeddings.permute(1, 0, 2, 3).reshape(-1, *text_embeddings.shape[2:])
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        e_t = unet(latent_input, timestep, embedd).sample
        if prediction_type == 'v_prediction':
            e_t = torch.cat([alpha_t] * 2) * e_t + torch.cat([sigma_t] * 2) * latent_input
        e_t_uncond, e_t = e_t.chunk(2)
        if get_raw:
            return e_t_uncond, e_t
        e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
        assert torch.isfinite(e_t).all()
    if get_raw:
        return e_t
    pred_z0 = (z_t - sigma_t * e_t) / alpha_t
    return e_t, pred_z0


class DistillationLoss:

    def noise_input(self, z, eps=None, timestep: Optional[int] = None):
        if timestep is None:
            b = z.shape[0]
            timestep = torch.randint(
                low=self.t_min,
                high=min(self.t_max, 1000) - 1,  # Avoid the highest timestep.
                size=(b,),
                device=z.device, dtype=torch.long)
        if eps is None:
            eps = torch.randn_like(z)
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        z_t = alpha_t * z + sigma_t * eps
        return z_t, eps, timestep, alpha_t, sigma_t

    def get_eps_prediction(self, z_t: T, timestep: T, text_embeddings: T, alpha_t: T, sigma_t: T, get_raw=False,
                           guidance_scale=7.5):
        return get_eps_prediction_cfg(self.unet, self.prediction_type, z_t, timestep, text_embeddings, alpha_t,
                                      sigma_t, get_raw=get_raw, guidance_scale=guidance_scale)

    def get_loss_distill(self, z, text_embeddings, guidance_scale):
        with torch.inference_mode():
            z_t, eps, timestep, alpha_t, sigma_t = self.noise_input(z, eps=None, timestep=None)
            e_t, pred_z0 = self.get_eps_prediction(z_t, timestep, text_embeddings, alpha_t, sigma_t,
                                             guidance_scale=guidance_scale)
            assert torch.isfinite(pred_z0).all()
            pred_z0 = torch.nan_to_num(pred_z0.detach(), 0.0, 0.0, 0.0)
        w_t = (alpha_t.detach().clone() ** self.alpha_exp) * (sigma_t.detach().clone() ** self.sigma_exp)
        residual = ((z - pred_z0.detach().clone()) ** 2)
        assert torch.isfinite(residual).all()
        loss_distill = w_t * residual

        return loss_distill.sum() / (z.shape[2] * z.shape[3])


    # from DreamFusion appendix A1: sds_loss(x) = weight(t) * dot(stopgrad[epshat_t - eps], x) where x = g(theta)
    # and its grad is the known formula: weight(t) * (epshat_t - eps) * grad(x)
    def get_sds_loss(self, z: T, text_embeddings: T, guidance_scale, eps=None, timestep=None) -> T:
        with torch.inference_mode():
            z_t, eps, timestep, alpha_t, sigma_t = self.noise_input(z, eps=eps, timestep=timestep)
            e_t, pred_z0 = self.get_eps_prediction(z_t, timestep, text_embeddings, alpha_t, sigma_t,
                                             guidance_scale=guidance_scale)
            w_t = (alpha_t ** self.alpha_exp) * (sigma_t ** self.sigma_exp)
            grad_z = w_t * (e_t - eps)
            assert torch.isfinite(grad_z).all()
            grad_z = torch.nan_to_num(grad_z.detach(), 0.0, 0.0, 0.0)
        sds_loss = grad_z.clone() * z
        del grad_z
        sds_loss = sds_loss.sum() / (z.shape[2] * z.shape[3]) # Daniel: normalization is necessary to avoid exploding grads

        return sds_loss, z_t, pred_z0, timestep

    def __init__(self, device, pipe: StableDiffusionPipeline, dtype=torch.float32):
        self.t_min = 50
        self.t_max = 950
        self.alpha_exp = 0
        self.sigma_exp = 0
        self.dtype = dtype
        self.unet, self.alphas, self.sigmas = init_pipe(device, dtype, pipe.unet, pipe.scheduler)
        self.prediction_type = pipe.scheduler.config.prediction_type



model_id  =  ["stabilityai/stable-diffusion-2-1"][0]
pipeline = StableDiffusionPipeline.from_pretrained(model_id,).to(device)

def set_deterministic_state():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)


def image_optimization(pipe: StableDiffusionPipeline, text_target: str, num_iters=200, guidance_scale=7.5, lr=1e-1) -> None:
    dds_loss = DistillationLoss(device, pipe)
    show_interval = int(num_iters / 10)

    # image_source = torch.from_numpy(image).float().permute(2, 0, 1) / 127.5 - 1
    image_source = torch.rand((512,512,3)).float().permute(2, 0, 1) * 2.0 - 1.0
    image_source = image_source.unsqueeze(0).to(device)
    with torch.no_grad():
        z_source = pipeline.vae.encode(image_source)['latent_dist'].mean * pipeline.vae.config.scaling_factor
        image_target = image_source.clone()
        embedding_null = get_text_embeddings(pipeline, "")
        embedding_text_target = get_text_embeddings(pipeline, text_target)
        embedding_target = torch.stack([embedding_null, embedding_text_target], dim=1)

    image_target.requires_grad = False

    z_target = z_source.clone()
    z_target.requires_grad = True
    optimizer = SGD(params=[z_target], lr=lr)

    for i in range(num_iters):
        loss, z_t, pred_z0, timestep = dds_loss.get_sds_loss(z_target, embedding_target, guidance_scale=guidance_scale)
        optimizer.zero_grad()
        loss.backward()  #used to be (2000 * loss).backward()
        optimizer.step()
        if (i+ 1) % 50 == 0:
            print("iteration: {}/{}, loss: {}".format(i + 1, num_iters, loss.item()))

        if i % show_interval == 0:
            output_log["z"].append(z_target.clone().detach())
            output_log["z_t"].append(z_t.detach())
            output_log["pred_z0"].append(pred_z0.detach())
            output_log["t"].append(timestep.item())

            out = decode(z_target, pipeline, im_cat=None)
            plt.imshow(out); plt.show()
    show_output_log()


def deep_image_prior(pipe, text_target, num_iters=200, guidance_scale=100):
    dds_loss = DistillationLoss(device, pipe)

    image_source = torch.rand((512, 512, 3)).float().permute(2, 0, 1) * 2.0 - 1.0
    image_source = image_source.unsqueeze(0).to(device)
    with torch.no_grad():
        z_source = pipeline.vae.encode(image_source)['latent_dist'].mean * pipeline.vae.config.scaling_factor
        embedding_null = get_text_embeddings(pipeline, "")
        embedding_text_target = get_text_embeddings(pipeline, text_target)
        embedding_target = torch.stack([embedding_null, embedding_text_target], dim=1)

    z_source.requires_grad = False

    net = skip(
        z_source.shape[1], z_source.shape[1],
        num_channels_down=[8, 16, 32, 64, 128],
        num_channels_up=[8, 16, 32, 64, 128],
        num_channels_skip=[0, 0, 0, 4, 4],
        upsample_mode='bilinear',
        need_sigmoid=False, need_bias=True, pad='reflection', act_fun='LeakyReLU').to(device)
    net.requires_grad = True
    #optimizer = SGD(params=net.parameters(), lr=1e-2)
    optimizer = AdamW(params=net.parameters(), lr=1e-5)

    for i in range(num_iters):
        z_target = net(z_source)
        loss = dds_loss.get_sds_loss(z_target, embedding_target, guidance_scale=guidance_scale)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+ 1) % 500 == 0:
            print("iteration: {}/{}, abs loss: {}".format(i + 1, num_iters, abs(loss.item())))
        if (i + 1) % 1000 == 0:
            # out = decode(z_taregt, pipeline, im_cat=image)
            out = decode(z_target, pipeline, im_cat=None)
            plt.imshow(out); plt.colorbar()
            #plt.imshow(z_target[0,0,...].detach().cpu().numpy()); plt.colorbar()
            plt.show()


def deep_image_prior_test(pipe, prompt, num_iters):
    target_image = pil_to_tensor(pipe(prompt=prompt, guidance_scale=7.5).images[0]) / 255
    target_image = torch.unsqueeze(target_image, 0).to(device)
    seed = torch.rand(target_image.shape).to(device)

    net = skip(
        target_image.shape[1], target_image.shape[1],
        num_channels_down=[8, 16, 32, 64, 128],
        num_channels_up=[8, 16, 32, 64, 128],
        num_channels_skip=[0, 0, 0, 4, 4],
        upsample_mode='bilinear',
        need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').to(device)

    net.requires_grad = True
    target_image.requires_grad = False
    seed.requires_grad = False

    optimizer = SGD(params=net.parameters(), lr=1e-2)

    loss_fn = nn.MSELoss(reduction='mean')
    for i in range(num_iters):
        loss = loss_fn(net(seed), target_image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print("iteration: {}/{}, loss: {}".format(i + 1, num_iters, loss.item()))
        if (i + 1) % 500 == 0:
            # out = decode(z_taregt, pipeline, im_cat=image)
            plt.imshow(net(seed)[0].permute(1,2,0).detach().cpu().numpy())
            plt.show()


def turbo_sd(pipe, text_target, num_iters=200, guidance_scale=100, lr=1e-3):
    dds_loss = DistillationLoss(device, pipe)
    show_interval = int(num_iters / 10)

    image_source = torch.rand((512, 512, 3)).float().permute(2, 0, 1) * 2.0 - 1.0
    image_source = image_source.unsqueeze(0).to(device)
    with torch.no_grad():
        z_source = pipeline.vae.encode(image_source)['latent_dist'].mean * pipeline.vae.config.scaling_factor
        embedding_null = get_text_embeddings(pipeline, "")
        embedding_text_target = get_text_embeddings(pipeline, text_target)
        embedding_target = torch.stack([embedding_null, embedding_text_target], dim=1)

    model_id = ["stabilityai/sd-turbo"][0]
    sd_turbo_pipe = AutoPipelineForText2Image.from_pretrained(model_id, ).to(device)
    unet, alphas, sigmas = init_pipe(device, torch.float32, sd_turbo_pipe.unet, sd_turbo_pipe.scheduler, unet_requires_grad=True)

    unet.requires_grad = True
    unet.train()

    T = torch.stack([torch.tensor(999)] * z_source.shape[0], dim=0).to(device)
    embed = get_text_embeddings(sd_turbo_pipe, "")
    #embed = get_text_embeddings(sd_turbo_pipe, "The style of Van Gogh")

    torch.manual_seed(0)
    z_T = torch.randn_like(z_source, requires_grad=False).to(device)
    alpha_T = alphas[T, None, None, None].clone().detach()
    sigma_T = sigmas[T, None, None, None].clone().detach()
    del z_source
    del image_source

    optimizer = SGD(params=unet.parameters(), lr=lr)
    #optimizer = SGD(params=[z_T], lr=1e-1)

    t_min = 50
    t_max = 999

    for i in range(num_iters):
        eps_pred = unet(z_T, timestep=T, encoder_hidden_states=embed).sample
        z = (z_T - sigma_T * eps_pred) / alpha_T

        t_min_ = int((1 - (i/num_iters)) * t_max + (i/num_iters) * t_min)
        #timestep = int((1 - (i/num_iters)) * t_max + (i/num_iters) * t_min)
        timestep = torch.randint(low=t_min_, high=1000, size=(1,)).item()
        timestep = torch.stack([torch.tensor(timestep)] * z.shape[0], dim=0).to(device)

        loss, z_t, pred_z0, timestep = dds_loss.get_sds_loss(z, embedding_target, guidance_scale=guidance_scale, eps=eps_pred,
                                     timestep=timestep)

        gc.collect()
        torch.cuda.empty_cache()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+ 1) % 50 == 0:
            print("iteration: {}/{}, loss: {}".format(i + 1, num_iters, loss.item()))
        if ((i % show_interval) == 0) or (i == (num_iters-1)):
            output_log["z"].append(z.detach())
            output_log["z_t"].append(z_t.detach())
            output_log["pred_z0"].append(pred_z0.detach())
            output_log["t"].append(timestep.item())

            out = decode(z, sd_turbo_pipe, im_cat=None)
            plt.imshow(out);
            plt.show()
    show_output_log()



def SD(pipe, texts_target, text_source, num_iters=200, guidance_scale=7.5, student_guidance_scale=7.5, lr=1e-3, eta=0.0):
    dds_loss = DistillationLoss(device, pipe)
    show_interval = int(num_iters / 10)

    image_source = torch.rand((512, 512, 3)).float().permute(2, 0, 1) * 2.0 - 1.0
    image_source = image_source.unsqueeze(0).to(device)
    with torch.no_grad():
        z_source = pipeline.vae.encode(image_source)['latent_dist'].mean * pipeline.vae.config.scaling_factor
        embedding_null = get_text_embeddings(pipeline, "")
        embedding_text_targets = [get_text_embeddings(pipeline, text) for text in texts_target]
        embeddings_target = [torch.stack([embedding_null, e], dim=1) for e in embedding_text_targets]

    model_id = ["stabilityai/stable-diffusion-2-1"][0]
    student_pipe = AutoPipelineForText2Image.from_pretrained(model_id, ).to(device)
    unet, alphas, sigmas = init_pipe(device, torch.float32, student_pipe.unet, student_pipe.scheduler, unet_requires_grad=True)
    alphas_cumprod = student_pipe.scheduler.alphas_cumprod.to(device, dtype=torch.float64)

    unet.requires_grad = True
    unet.train()

    T = torch.stack([torch.tensor(999)] * z_source.shape[0], dim=0).to(device)
    embed = get_text_embeddings(student_pipe, text_source)

    embed_source = torch.stack([embedding_null, embed], dim=1)


    z_T = torch.randn_like(z_source, requires_grad=False).to(device)
    alpha_t = alphas[T, None, None, None].clone().detach()
    sigma_t = sigmas[T, None, None, None].clone().detach()
    del z_source
    del image_source

    optimizer = SGD(params=unet.parameters(), lr=lr)

    t_min = 20
    t_max = 999

    z_t = z_T
    timestep = T.clone()
    for i in range(num_iters):
        # prev timestep prediction
        with torch.no_grad():
            eps_t, pred_z0 = get_eps_prediction_cfg(unet, student_pipe.scheduler.config.prediction_type, z_t, timestep, embed_source,
                                                    alpha_t, sigma_t, False, guidance_scale=student_guidance_scale)

        # ddim forward (noising) step to new timestep
        with torch.no_grad():
            timestep = int((1 - (i / num_iters)) * t_max + (i / num_iters) * t_min)
            timestep = torch.stack([torch.tensor(timestep)] * z_T.shape[0], dim=0).to(device)
            alpha_t = alphas[timestep, None, None, None]
            sigma_t = sigmas[timestep, None, None, None]
            z_t = alpha_t * pred_z0 + sigma_t * eps_t

        # image prediction - this is "g(theta)" in SDS terms
        eps_pred, z = get_eps_prediction_cfg(unet, student_pipe.scheduler.config.prediction_type, z_t, timestep, embed_source, alpha_t,
                                                    sigma_t, False, guidance_scale=student_guidance_scale)

        # adding stochasticity to the predicted noise, according to Eq. 12 in https://arxiv.org/pdf/2010.02502 (DDIM paper).
        # Using float64 precision to better reproduce the deterministic case, still not perfect due to rounding errors.
        eta = torch.tensor(eta, dtype=torch.float64, device='cuda')
        eps_pred = eps_pred.to(dtype=torch.float64)
        sigma_t_64 = sigma_t.to(dtype=torch.float64)
        direction_to_z_t_term = ((1 - alphas_cumprod[timestep] - eta**2) ** 0.5) * eps_pred
        stochastic_term = eta * torch.randn_like(eps_pred)
        eps = ((direction_to_z_t_term + stochastic_term) / sigma_t_64).to(dtype=torch.float32)

        #eps = eps_pred.to(dtype=torch.float32)
        i_target = torch.randint(0, len(texts_target), size=(1,)).item()
        loss, z_t, pred_z0, timestep = dds_loss.get_sds_loss(z, embeddings_target[i_target], guidance_scale=guidance_scale, eps=eps,
                                     timestep=timestep)

        gc.collect()
        torch.cuda.empty_cache()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if (i+ 1) % 50 == 0:
            print("iteration: {}/{}, loss: {}".format(i + 1, num_iters, loss.item()))
        if ((i % show_interval) == 0) or (i == (num_iters-1)):
            output_log["z"].append(z.detach())
            output_log["z_t"].append(z_t.detach())
            output_log["pred_z0"].append(pred_z0.detach())
            output_log["t"].append(timestep.item())

            out = decode(z, student_pipe, im_cat=None)
            plt.imshow(out);
            plt.show()

    show_output_log()


def SD_lora(pipe, text_target, text_source, num_iters=200, guidance_scale=7.5, student_guidance_scale=7.5, lr=1e-3):
    from peft import LoraConfig # keep here
    dds_loss = DistillationLoss(device, pipe)
    show_interval = int(num_iters / 10)

    image_source = torch.rand((512, 512, 3)).float().permute(2, 0, 1) * 2.0 - 1.0
    image_source = image_source.unsqueeze(0).to(device)
    with torch.no_grad():
        z_source = pipeline.vae.encode(image_source)['latent_dist'].mean * pipeline.vae.config.scaling_factor
        embedding_null = get_text_embeddings(pipeline, "")
        embedding_text_target = get_text_embeddings(pipeline, text_target)
        embedding_target = torch.stack([embedding_null, embedding_text_target], dim=1)

    model_id = ["stabilityai/stable-diffusion-2-1"][0]
    student_pipe = AutoPipelineForText2Image.from_pretrained(model_id, ).to(device)
    unet, alphas, sigmas = init_pipe(device, torch.float32, student_pipe.unet, student_pipe.scheduler, unet_requires_grad=False)

    unet_lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    unet.train()

    T = torch.stack([torch.tensor(999)] * z_source.shape[0], dim=0).to(device)
    embed = get_text_embeddings(student_pipe, text_source)
    embed_source = torch.stack([embedding_null, embed], dim=1)

    z_T = torch.randn_like(z_source, requires_grad=False).to(device)
    alpha_t = alphas[T, None, None, None].clone().detach()
    sigma_t = sigmas[T, None, None, None].clone().detach()
    del z_source
    del image_source

    optimizer = SGD(params=lora_layers, lr=lr)

    t_min = 50
    t_max = 999

    z_t = z_T
    timestep = T.clone()
    for i in range(num_iters):
        # prev timestep prediction
        with torch.no_grad():
            eps_t, pred_z0 = get_eps_prediction_cfg(unet, student_pipe.scheduler.config.prediction_type, z_t, timestep, embed_source,
                                                    alpha_t, sigma_t, False, guidance_scale=student_guidance_scale)

        # ddim forward (noising) step to new timestep
        with torch.no_grad():
            timestep = int((1 - (i / num_iters)) * t_max + (i / num_iters) * t_min)
            timestep = torch.stack([torch.tensor(timestep)] * z_T.shape[0], dim=0).to(device)
            alpha_t = alphas[timestep, None, None, None]
            sigma_t = sigmas[timestep, None, None, None]
            z_t = alpha_t * pred_z0 + sigma_t * eps_t

        eps_pred, z = get_eps_prediction_cfg(unet, student_pipe.scheduler.config.prediction_type, z_t, timestep, embed_source, alpha_t,
                                                    sigma_t, False, guidance_scale=student_guidance_scale)

        loss, z_t, pred_z0, timestep = dds_loss.get_sds_loss(z, embedding_target, guidance_scale=guidance_scale, eps=eps_pred,
                                     timestep=timestep)

        gc.collect()
        torch.cuda.empty_cache()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if (i+ 1) % 50 == 0:
            print("iteration: {}/{}, loss: {}".format(i + 1, num_iters, loss.item()))
        if ((i % show_interval) == 0) or (i == (num_iters-1)):
            output_log["z"].append(z.detach())
            output_log["z_t"].append(z_t.detach())
            output_log["pred_z0"].append(pred_z0.detach())
            output_log["t"].append(timestep.item())

            out = decode(z, student_pipe, im_cat=None)
            plt.imshow(out);
            plt.show()

    show_output_log()



if __name__ == '__main__':
    teachers_prompt = ["a photorealistic image of a man on a horse"]
    #student_prompt = "A photorealistic image of a man in the beach"
    student_prompt = ""
    set_deterministic_state()
    #image_optimization(pipeline,prompt, num_iters=3000, guidance_scale=100, lr=1e2)
    #turbo_sd(pipeline,prompt, num_iters=3000, guidance_scale=7.5, lr=1e-3)
    SD(pipeline, teachers_prompt, student_prompt, num_iters=10000, guidance_scale=7.5, student_guidance_scale=7.5, lr=1e-3, eta=0.05)
    #SD_lora(pipeline, teacher_prompt, student_prompt, num_iters=10000, guidance_scale=7.5, student_guidance_scale=7.5, lr=1e-2)
    #deep_image_prior(pipeline, prompt, num_iters=10000)
    #deep_image_prior_test(pipeline,prompt, num_iters=3000)

