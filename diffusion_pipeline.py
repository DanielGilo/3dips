import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

# seva imports
from seva.seva.utils import load_model
from seva.seva.model import SGMWrapper
from seva.seva.modules.autoencoder import AutoEncoder
from seva.seva.modules.conditioner import CLIPConditioner
from seva.seva.sampling import DDPMDiscretization, DiscreteDenoiser, MultiviewCFG, VanillaCFG, append_dims
from seva.seva.eval import unload_model
from lora_diffusion import inject_trainable_lora
from einops import repeat

from plot_utils import seva_tensor_to_np_plottable

class DiffusionPipeline:
    def __init__(self, model_id, device, dtype, do_compile=True):
        self.device = device
        self.dtype= dtype
        self.pipeline = self.get_pipeline(model_id)

        self.latent_shape = [self.pipeline.vae.config.latent_channels,
                             self.pipeline.unet.config.sample_size,
                             self.pipeline.unet.config.sample_size]
        self.pixel_space_shape = [3, self.latent_shape[1] * self.pipeline.vae_scale_factor,
                                  self.latent_shape[2] * self.pipeline.vae_scale_factor]
        
        # set zero terminal SNR - problematic for eps-predicting models https://arxiv.org/pdf/2305.08891
        #self.pipeline.scheduler = DDPMScheduler.from_config(
        #    self.pipeline.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing")
        #self.scheduler = self.pipeline.scheduler

        beta_start = 0.0001
        beta_end = 0.02 # default is 0.02
        betas = torch.linspace(beta_start, beta_end, 1000, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)


        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae

        #self.unet.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=self.dtype)
        #self.vae.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=self.dtype)


        if do_compile:
            self.unet.compile()
            self.vae.compile()

        #self.pipeline.enable_vae_slicing()
        #self.pipeline.enable_vae_tiling()

        self.prediction_type = self.pipeline.scheduler.prediction_type
        with torch.inference_mode():
            # self.alphas = torch.sqrt(self.scheduler.alphas_cumprod).to(self.device, dtype=dtype)
            # self.sigmas = torch.sqrt(1 - self.scheduler.alphas_cumprod).to(self.device, dtype=dtype)
            self.alphas = torch.sqrt(alphas_cumprod).to(self.device, dtype=dtype)
            self.sigmas = torch.sqrt(1 - alphas_cumprod).to(self.device, dtype=dtype)
        for p in self.unet.parameters():
            p.requires_grad = False
        for p in self.pipeline.text_encoder.parameters():
            p.requires_grad = False

    def get_pipeline(self, model_id):
        return StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device)

    def noise_to_timestep(self, z0, timestep, eps):
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        z_t = alpha_t * z0 + sigma_t * eps
        return z_t
    
    @torch.no_grad()
    def invert_to_timestep(self, z0, guidance_scale, text_embeddings, final_timestep, n_inv_steps, batch_s, batch_e):
        latent = z0.clone()
        timesteps = torch.linspace(0, final_timestep.item(), n_inv_steps, dtype=torch.int, device=self.device).unsqueeze(dim=1)

        for i in range(n_inv_steps):

            # Final timestep reached
            if (i >= n_inv_steps - 1):
                break
            
            timestep = timesteps[i]

            pred_eps, _ = self.predict_eps_and_sample(latent, timestep, guidance_scale, text_embeddings, batch_s, batch_e)
            alpha_t = self.alphas[timestep, None, None, None]
            sigma_t = self.sigmas[timestep, None, None, None]

            next_timestep = timesteps[i+1]
            alpha_next_t = self.alphas[next_timestep, None, None, None]
            sigma_next_t = self.sigmas[next_timestep, None, None, None]

            # Update step
            latent = (latent - sigma_t * pred_eps) * (alpha_next_t / alpha_t) + sigma_next_t * pred_eps
        
        return latent

    # @torch.no_grad()
    # def ddim_sample(self, z_t, guidance_scale, text_embeddings, initial_timestep, n_steps, batch_s, batch_e):
    #     latent = z_t.clone()
    #     timesteps = torch.linspace(initial_timestep.item(), 0.0, n_steps, dtype=torch.int, device=self.device).unsqueeze(dim=1)

    #     for i in tqdm(range(n_steps)):

    #         # Final timestep reached
    #         if (i >= n_steps - 1):
    #             break
            
    #         timestep = timesteps[i]

    #         pred_eps, _ = self.predict_eps_and_sample(latent, timestep, guidance_scale, text_embeddings, batch_s, batch_e)
    #         alpha_t = self.alphas[timestep, None, None, None]
    #         sigma_t = self.sigmas[timestep, None, None, None]

    #         next_timestep = timesteps[i+1]
    #         alpha_next_t = self.alphas[next_timestep, None, None, None]
    #         sigma_next_t = self.sigmas[next_timestep, None, None, None]

    #         # Update step
    #         predicted_z0 = (latent - sigma_t * pred_eps) / alpha_t
    #         direction_pointing_to_zt = sigma_next_t * pred_eps
    #         latent = alpha_next_t * predicted_z0 + direction_pointing_to_zt
        
    #     return latent

    @torch.no_grad()
    def ddim_sample(self, z_t, guidance_scale, prompt, initial_timestep, n_steps, batch_s, batch_e):
        pipe = self.pipeline
        start_latents = z_t
        # Encode prompt
        text_embeddings = pipe._encode_prompt(prompt, self.device, 1, True, "")

        # Set num inference steps
        pipe.scheduler.set_timesteps(n_steps, device=self.device)

        # Create a random starting point if we don't have one already
        if start_latents is None:
            start_latents = torch.randn(1, 4, 64, 64, device=self.device)
            start_latents *= pipe.scheduler.init_noise_sigma

        latents = start_latents.clone()

        for i in tqdm(range(initial_timestep, n_steps)):

            t = pipe.scheduler.timesteps[i]

            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            # Predict the noise residual
            #noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample      

            # Perform guidance
            #noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            #noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            noise_pred, _ = self.predict_eps_and_sample(latents, t.unsqueeze(0), guidance_scale, text_embeddings.unsqueeze(0), batch_s, batch_e)


            # Normally we'd rely on the scheduler to handle the update step:
            # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

            # Instead, let's do it ourselves:
            prev_t = max(1, t.item() - (1000 // n_steps))  # t-1
            alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
            alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
            predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
            latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt
        
        return latents

    def prepare_latent_input_for_unet(self, z_t, batch_s, batch_e):
        return torch.cat([z_t] * 2)

    def predict_eps_and_sample(self, z_t, timestep, guidance_scale, text_embeddings, batch_s, batch_e):
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        latent_input = self.prepare_latent_input_for_unet(z_t, batch_s, batch_e)
        timestep = torch.cat([timestep] * 2)
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(-1, *text_embeddings.shape[2:])
        embedd = torch.cat([embedd] * z_t.shape[0])
        timestep = torch.cat([timestep] * z_t.shape[0])
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            e_t = self.unet(latent_input, timestep, embedd).sample
            if self.prediction_type == 'v_prediction':
                e_t = torch.cat([alpha_t] * timestep.shape[0]) * e_t + torch.cat([sigma_t] * timestep.shape[0]) * latent_input
            e_t_uncond, e_t = e_t.chunk(2)
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
            #assert torch.isfinite(e_t).all()
        pred_z0 = (z_t - sigma_t * e_t) / alpha_t
        return e_t, pred_z0

    @torch.no_grad()
    def get_text_embeddings(self, text: str):
        tokens = self.pipeline.tokenizer([text], padding="max_length", max_length=77, truncation=True,
                                return_tensors="pt", return_overflowing_tokens=True).input_ids.to(self.device)
        return self.pipeline.text_encoder(tokens).last_hidden_state.detach()
    
    def unload_text_encoder(self):
        self.pipeline.text_encoder.to("cpu")

    def decode(self, latent, type="pil", do_postprocess=True):
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            images = self.vae.decode((1 / self.vae.config.scaling_factor) * latent, return_dict=False)[0]
        if do_postprocess:
            images = self.pipeline.image_processor.postprocess(images, output_type=type)
        return images


@torch.no_grad()
def denormalize(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image



class SevaPipeline:
    def __init__(self, device, value_dict, do_compile=True):
        self.device = device
        self.model = SGMWrapper(load_model(device="cpu", verbose=True)).to(device=self.device)
        # loading to cpu to save memory
        self.ae = AutoEncoder(chunk_size=1).to(device=self.device)
        self.ae.module.eval().requires_grad_(False)
        #self.ae.module.enable_slicing()
        #self.ae.module.enable_tiling()
        #self.ae.module.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn)


        if do_compile:
            self.model.compile()
            self.ae.compile()

        self.conditioner = CLIPConditioner().to("cpu")
        self.discretization = DDPMDiscretization()
        self.denoiser = DiscreteDenoiser(discretization=self.discretization, num_idx=1000, device=device)
        #self.guider = MultiviewCFG(cfg_min=1.2) # from seva demo.py
        self.guider = VanillaCFG()
        self.version_dict = {
            "H": 576,
            "W": 576,
            "T": 21,
            "C": 4,
            "f": 8,
            "options": {},
        }

        self.value_dict = value_dict
        self.n_input_frames = value_dict["cond_frames_mask"].sum().item()
        self.n_padding = self.version_dict["T"] - value_dict["num_imgs_no_padding"]

        self.latent_shape = [self.version_dict["T"],self.version_dict["C"],
                self.version_dict["H"] // self.version_dict["f"], self.version_dict["W"] // self.version_dict["f"]] # [21, 4, 72, 72] 

        self.c, self.uc, self.additional_model_inputs, self.additional_sampler_inputs = self.prepare_model_inputs()

        self.model.train()
        self.set_model_requires_grad()

    def set_model_requires_grad(self):
        self.model.requires_grad_(False)

    def prepare_model_inputs(self):
        """
        adapted from seva's eval.py do_sample function.
        """
        
        imgs = self.value_dict["cond_frames"].to(device=self.device)
        input_masks = self.value_dict["cond_frames_mask"].to(device=self.device)
        pluckers = self.value_dict["plucker_coordinate"].to(device=self.device)

        T = self.version_dict["T"]

        encoding_t = 1 # not sure what that is
        with torch.no_grad():
            self.conditioner = self.conditioner.to(device=self.device)

            input_latents = self.ae.encode(imgs[input_masks], encoding_t)
            latents = torch.nn.functional.pad(
                input_latents, (0, 0, 0, 0, 0, 1), value=1.0
            )
            c_crossattn = repeat(self.conditioner(imgs[input_masks]).mean(0), "d -> n 1 d", n=T)
            uc_crossattn = torch.zeros_like(c_crossattn)
            c_replace = latents.new_zeros(T, *latents.shape[1:])
            c_replace[input_masks] = latents
            uc_replace = torch.zeros_like(c_replace)
            c_concat= torch.cat(
                [
                    repeat(
                        input_masks,
                        "n -> n 1 h w",
                        h=pluckers.shape[2],
                        w=pluckers.shape[3],
                    ),
                    pluckers,
                ],
                1,
            )
            uc_concat = torch.cat(
                [pluckers.new_zeros(T, 1, *pluckers.shape[-2:]), pluckers], 1
            )
            c_dense_vector = pluckers
            uc_dense_vector = c_dense_vector
            c = {
                "crossattn": c_crossattn,
                "replace": c_replace,
                "concat": c_concat,
                "dense_vector": c_dense_vector,
            }
            uc = {
                "crossattn": uc_crossattn,
                "replace": uc_replace,
                "concat": uc_concat,
                "dense_vector": uc_dense_vector,
            }
            unload_model(self.conditioner)

            additional_model_inputs = {"num_frames": T}
            # additional_sampler_inputs = {
            #     "c2w": self.value_dict["c2w"].to("cuda"),
            #     "K": self.value_dict["K"].to("cuda"),
            #     "input_frame_mask": self.value_dict["cond_frames_mask"].to("cuda")
            # } # for MultiviewCFG, additional inputs are needed
            additional_sampler_inputs = {} # for vanilla CFG, no additional inputs are needed

            return c, uc, additional_model_inputs, additional_sampler_inputs


    def prepare_input_for_denoiser(self, z_t, sigma):
        latent_shape = self.latent_shape
        if (z_t.shape[0] != latent_shape[0]): # need to add input frames (zeros, as they will be replaced anyway before prediction) and pad with last output frame
            z_t = torch.cat([torch.zeros((self.n_input_frames, latent_shape[1], latent_shape[2], latent_shape[3]), device="cuda"), z_t], dim=0) # filling shape to latent shape
            z_t = torch.cat([z_t, z_t[-1].unsqueeze(0).repeat(self.n_padding, 1, 1, 1)], dim=0)
        assert z_t.shape[0] == latent_shape[0]
        input, sigma, c = self.guider.prepare_inputs(z_t, sigma, self.c, self.uc)
        return input, sigma, c
    
    def get_sigma_from_timestep(self, timestep):
        return self.denoiser.idx_to_sigma(timestep.to(dtype=torch.int))

    def predict_eps_and_sample(self, z_t, timestep, guidance_scale=2.0):     
        sigma_orig = self.get_sigma_from_timestep(timestep)
        if sigma_orig.shape[0] == 1:
            s_in = torch.ones([self.latent_shape[0]], device="cuda")
            sigma_orig = sigma_orig.item() * s_in

        input, sigma, c = self.prepare_input_for_denoiser(z_t, sigma_orig)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_z0_c_and_uc = self.denoiser(self.model, input, sigma, c.copy(), **self.additional_model_inputs)

            # logic taken from seva's DiscreteDenoiser __call__() method, 
            # here we isolate the network output (which is the predicted noise) from the denoiser's return value.
            sigma = append_dims(sigma, input.ndim)
            if "replace" in c: # for input frames (mask = True) use the clean latents
                x, mask = c.pop("replace").split((input.shape[1], 1), dim=1)
                input = input * (1 - mask) + x * mask
            c_skip, c_out, _, _ = self.denoiser.scaling(sigma)
            pred_eps_c_and_uc = (pred_z0_c_and_uc - input * c_skip) / c_out

            pred_z0 = self.guider(pred_z0_c_and_uc, sigma_orig, guidance_scale, **self.additional_sampler_inputs)
            pred_eps = self.guider(pred_eps_c_and_uc, sigma_orig, guidance_scale, **self.additional_sampler_inputs)

        return pred_eps, pred_z0
    
    def get_only_output_frames(self, preds):
        assert preds.shape[0] == self.latent_shape[0]
        return preds[self.n_input_frames:len(preds)-self.n_padding]

    def noise_to_timestep(self, z0, timestep, eps):
        # TODO: verify!!
        eps  = eps / eps.var()
        sigma_zero = self.get_sigma_from_timestep(timestep*0.0)
        sigma_t = self.get_sigma_from_timestep(timestep)


        z_t = z0 + eps * append_dims(sigma_t**2 - sigma_zero**2, z0.ndim) ** 0.5

        return z_t

    def get_text_embeddings(self, s): #filler
        return None


    def decode(self, latent, type="np"):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            samples = self.ae.decode(latent, 1)
        if type != "pt": # np or PIL
            samples = seva_tensor_to_np_plottable(samples.detach())  
        if type == "PIL":
            samples = [Image.fromarray(sample) for sample in samples]

        return samples
