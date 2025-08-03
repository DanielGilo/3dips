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
from seva.seva.sampling import DDPMDiscretization, DiscreteDenoiser, MultiviewCFG, VanillaCFG, append_dims, to_d
from seva.seva.eval import unload_model
from lora_diffusion import inject_trainable_lora
from einops import repeat

from plot_utils import seva_tensor_to_np_plottable

class DiffusionPipeline:
    def __init__(self, model_id, device, dtype):
        self.device = device
        self.dtype= dtype
        self.pipeline = self.get_pipeline(model_id)


    def get_pipeline(self, model_id):
        return StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device)

    
    # from https://arxiv.org/pdf/2405.15891 A.1
    def noise_to_timestep_using_inversion(self, z0, guidance_scale, timestep, n_inv_steps, batch_s, batch_e):
        step_size = 30
        cfg_inv = 0.0 # SDI recommended -7.5
        timestep_inv = min(999, timestep + step_size)
        z_t_inv = self.invert_to_timestep(z0, cfg_inv, timestep_inv, n_inv_steps, batch_s, batch_e)
        eps_inv = (z_t_inv - self.alphas[timestep_inv] * z0) / self.sigmas[timestep_inv]

        z_t = self.alphas[timestep] * z0 + self.sigmas[timestep] * eps_inv
        return z_t
    


    def prepare_latent_input_for_unet(self, z_t, batch_s, batch_e):
        return torch.cat([z_t] * 2)


    def decode(self, latent, type="pil", do_postprocess=True):
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            images = self.vae.decode((1 / self.vae.config.scaling_factor) * latent, return_dict=False)[0].to(device="cpu")
        if do_postprocess:
            images = self.pipeline.image_processor.postprocess(images, output_type=type)
        return images


class DiffusersStableDiffusionPipeline(DiffusionPipeline):
    def __init__(self, model_id, prompt, device="cuda", dtype=torch.float16, do_compile=True):
        super().__init__(model_id, device, dtype)
        self.latent_shape = [self.pipeline.vae.config.latent_channels,
                             self.pipeline.unet.config.sample_size,
                             self.pipeline.unet.config.sample_size]
        self.pixel_space_shape = [3, self.latent_shape[1] * self.pipeline.vae_scale_factor,
                                  self.latent_shape[2] * self.pipeline.vae_scale_factor]

        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae

        if do_compile:
            self.unet.compile()
            self.vae.compile()

        self.prediction_type = self.pipeline.scheduler.prediction_type
        with torch.inference_mode():
            self.alphas = torch.sqrt(self.pipeline.scheduler.alphas_cumprod).to(self.device, dtype=dtype)
            self.sigmas = torch.sqrt(1 - self.pipeline.scheduler.alphas_cumprod).to(self.device, dtype=dtype)
        for p in self.unet.parameters():
            p.requires_grad = False
        for p in self.pipeline.text_encoder.parameters():
            p.requires_grad = False
        
        self.text_embeddings = torch.stack([self.get_text_embeddings(""), self.get_text_embeddings(prompt)], dim=1)
        self.unload_text_encoder()
        
    def predict_eps_and_sample(self, z_t, timestep, guidance_scale, batch_s, batch_e):
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        latent_input = self.prepare_latent_input_for_unet(z_t, batch_s, batch_e)
        timestep = torch.cat([timestep] * 2)
        text_embeddings = torch.cat([self.text_embeddings] * z_t.shape[0])
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(-1, *text_embeddings.shape[2:])
        timestep = torch.cat([timestep] * z_t.shape[0])
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            e_t = self.unet(latent_input, timestep, embedd).sample
            if self.prediction_type == 'v_prediction':
                e_t = torch.cat([alpha_t] * timestep.shape[0]) * e_t + torch.cat([sigma_t] * timestep.shape[0]) * latent_input
            e_t_uncond, e_t = e_t.chunk(2)
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
        pred_z0 = (z_t - sigma_t * e_t) / alpha_t
        return e_t, pred_z0

    @torch.no_grad()
    def get_text_embeddings(self, text: str):
        tokens = self.pipeline.tokenizer([text], padding="max_length", max_length=77, truncation=True,
                                return_tensors="pt", return_overflowing_tokens=True).input_ids.to(self.device)
        return self.pipeline.text_encoder(tokens).last_hidden_state.detach()
    
    def unload_text_encoder(self):
        self.pipeline.text_encoder.to("cpu")

    def noise_to_timestep(self, z0, timestep, eps):
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        z_t = alpha_t * z0 + sigma_t * eps
        return z_t
    
    def invert_to_timestep(self, z0, guidance_scale, final_timestep, n_inv_steps, batch_s, batch_e):
        latent = z0.clone()
        n_inv_steps += 1
        timesteps = torch.linspace(0, final_timestep, n_inv_steps, dtype=torch.int, device=self.device)

        for i in tqdm(range(1, n_inv_steps)):      
            t = timesteps[i]

            noise_pred, _ = self.predict_eps_and_sample(latent, t.unsqueeze(0), guidance_scale, batch_s, batch_e)

            current_t = timesteps[i - 1]
            next_t = t
            alpha_t = self.alphas[current_t]
            sigma_t = self.sigmas[current_t]
            alpha_t_next = self.alphas[next_t]
            sigma_t_next = self.sigmas[next_t]

            # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
            latent = (latent - sigma_t * noise_pred) * (alpha_t_next / alpha_t) + sigma_t_next * noise_pred
        
        return latent

    def multi_step_sample(self, z_t, guidance_scale, initial_timestep, n_steps, batch_s, batch_e, eta=0.0):
        # DDIM
        latents = z_t.clone()
        n_steps += 1
        timesteps = torch.linspace(initial_timestep, 0, steps=n_steps, device="cuda", dtype=torch.int)

        for i in tqdm(range(n_steps-1)):
            t = timesteps[i]

            noise_pred, _ = self.predict_eps_and_sample(latents, t.unsqueeze(0), guidance_scale, batch_s, batch_e)

            prev_t = timesteps[i+1]

            alpha_prod_t = self.pipeline.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = self.pipeline.scheduler.alphas_cumprod[prev_t]
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
            std_dev_t = eta * variance ** (0.5)

            alpha_t = self.alphas[t.item()]
            sigma_t = self.sigmas[t.item()]
            alpha_t_prev = self.alphas[prev_t]
            sigma_t_prev = self.sigmas[prev_t]
            predicted_x0 = (latents - sigma_t * noise_pred) / alpha_t
            direction_pointing_to_xt = (sigma_t_prev ** 2 - std_dev_t ** 2) ** (0.5) * noise_pred

            variance_noise = torch.randn(z_t.shape).to("cuda")
            variance = std_dev_t * variance_noise

            latents = alpha_t_prev * predicted_x0 + direction_pointing_to_xt + variance

        eps = (z_t - self.alphas[initial_timestep] * latents) / self.sigmas[initial_timestep]
        return eps, latents



class SevaPipeline:
    def __init__(self, device, value_dict, do_compile=True):
        self.device = device
        self.model = SGMWrapper(load_model(weight_name="modelv1.1.safetensors", device="cpu", verbose=True)).to(device=self.device)
        # loading to cpu to save memory
        self.ae = AutoEncoder(chunk_size=1).to(device=self.device)
        self.ae.module.eval().requires_grad_(False)


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
    
    def euler_edm_step(self, latent, curr_t, next_t, cfg_w):
        # values used by seva/demo.py
        s_churn = 0.0
        s_noise = 1.0
        gamma = 0.0

        curr_sigma = self.get_sigma_from_timestep(curr_t)
        next_sigma = self.get_sigma_from_timestep(next_t)

        sigma_hat = curr_sigma * (gamma + 1.0) + 1e-6
        eps = torch.randn_like(latent) * s_noise
        latent = latent + eps * append_dims(sigma_hat**2 - curr_sigma**2, latent.ndim) ** 0.5

        _, z0_student = self.predict_eps_and_sample(latent, curr_t, cfg_w)
        d = to_d(latent, curr_sigma, z0_student)
        dt = append_dims(next_sigma - curr_sigma, latent.ndim)
        # latent = (latent + dt * d).detach()
        latent = (latent + dt * d)

        return latent, z0_student

    def euler_edm_sample(self, z_t, timestep, n_steps, cfg_w):
        latent = z_t
        sample_dt = timestep[0] // n_steps

        while timestep[0] > 0:
            next_timestep = timestep - sample_dt
            if next_timestep[0] < 0:
                next_timestep *= 0

            print(f"step t = {int(timestep[0])} -> {int(next_timestep[0])}")

            latent, _ = self.euler_edm_step(latent, timestep, next_timestep, cfg_w)
            timestep = next_timestep
        
        return latent
    

    def euler_edm_sample_guided_latents(self, z_t, timestep, n_steps, cfg_w, reference_latent, guidance_scale=0.0):
        """
        Performs EDM sampling with additional guidance towards reference_latent using MSE loss on predicted z0.
        Args:
            z_t: initial latent
            timestep: starting timestep (tensor)
            n_steps: number of steps
            cfg_w: classifier-free guidance weight
            reference_latent: target latent for guidance (same shape as predicted z0)
            guidance_scale: strength of guidance (float)
        Returns:
            latent: final latent after guided sampling
        """

        latent = z_t.clone().detach()
        sample_dt = timestep[0] // n_steps

        while timestep[0] > 0:
            next_timestep = timestep - sample_dt
            if next_timestep[0] < 0:
                next_timestep *= 0

            print(f"step t = {int(timestep[0])} -> {int(next_timestep[0])}")

            latent = latent.detach().requires_grad_(True)
            next_latent, z0_student = self.euler_edm_step(latent, timestep, next_timestep, cfg_w)

            # Ensure all tensors are in float format for loss calculation
            latent = latent.float()
            reference_latent = reference_latent.float()
            z0_student = z0_student.float()

            # Guidance: gradient step towards reference_latent using predicted z0
            mse_loss = torch.nn.functional.mse_loss(z0_student, reference_latent)
            print("loss:", mse_loss.item())

            grad = torch.autograd.grad(mse_loss, latent, retain_graph=False)[0]
            latent = (next_latent - guidance_scale * grad).detach()

            timestep = next_timestep

        return latent


    def euler_edm_sample_guided_pixels(self, z_t, timestep, n_steps, cfg_w, reference_frames,
                                        guidance_scale=0.0, batch_size=2):
        """
        Performs EDM sampling with MSE guidance computed in batches over decoded pixels.
        Only decodes small slices of predicted z0 at a time, but computes the full gradient w.r.t. the latent.
        """

        latent = z_t.clone().detach()
        sample_dt = timestep[0] // n_steps

        # Create LPIPS model (Alex is default; others: 'vgg', 'squeeze')
        # import lpips 
        # lpips_loss_fn = lpips.LPIPS(net='vgg').to("cuda")
        # lpips_loss_fn.eval()  # Important: LPIPS is not meant to be trained

        while timestep[0] > 0:
            next_timestep = timestep - sample_dt
            if next_timestep[0] < 0:
                next_timestep *= 0

            print(f"step t = {int(timestep[0])} -> {int(next_timestep[0])}")

            latent = latent.detach().requires_grad_(True)
            next_latent, z0_student = self.euler_edm_step(latent, timestep, next_timestep, cfg_w)

            grad_z0_accum = torch.zeros_like(z0_student)

            B = z0_student.shape[0]
            for i in range(0, B, batch_size):
                z0_batch = z0_student[i:i + batch_size].detach().requires_grad_(True)
                ref_batch = reference_frames[i:i + batch_size].float()

                # Decode to pixel space
                x0_batch = self.decode(z0_batch, type="pt").float()

                # Compute L1 loss
                loss = torch.nn.functional.l1_loss(x0_batch, ref_batch)
                print(f"  batch {i}-{i+batch_size}: L1 loss = {loss.item()}")

                # Compute MSE loss
                # loss = torch.nn.functional.mse_loss(x0_batch, ref_batch)
                # print(f"  batch {i}-{i+batch_size}: loss = {loss.item()}")
                
                # Compute perceptual loss
                # x0_batch = (x0_batch * 2 - 1.0).cuda()  # Scale to [-1, 1] for LPIPS
                # ref_batch = (ref_batch * 2 - 1.0).cuda()  # Scale to [-1, 1] for LPIPS
                # loss = lpips_loss_fn(x0_batch, ref_batch).mean()
                # print(f"  batch {i}-{i+batch_size}: LPIPS loss = {loss.item()}")

                # Compute gradient w.r.t. z0_batch
                grad_z0 = torch.autograd.grad(loss, z0_batch, retain_graph=(i + batch_size < B))[0].detach()

                # Store in grad_z0_accum
                grad_z0_accum[i:i + batch_size] = grad_z0

                # Optional: clear memory
                del z0_batch, x0_batch, loss, grad_z0
                torch.cuda.empty_cache()

            # Now compute gradient w.r.t. latent using chain rule: ∂loss/∂latent = ∂z0/∂latentᵗ * ∂loss/∂z0
            grad_latent = torch.autograd.grad(z0_student, latent, grad_outputs=grad_z0_accum)[0]

            # Euler update
            latent = (next_latent - guidance_scale * grad_latent).detach()
            timestep = next_timestep

        return latent



    def decode(self, latent, type="np"):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            samples = self.ae.decode(latent, 1).to(device="cpu")
        if type != "pt": # np or PIL
            samples = seva_tensor_to_np_plottable(samples.detach())  
        if type == "PIL":
            samples = [Image.fromarray(sample) for sample in samples]

        return samples
