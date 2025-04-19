import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

class DiffusionPipeline:
    def __init__(self, model_id, device, dtype):
        self.device = device
        self.pipeline = self.get_pipeline(model_id)
        self.scheduler = self.pipeline.scheduler
        self.unet = self.pipeline.unet
        self.prediction_type = self.pipeline.scheduler.prediction_type
        with torch.inference_mode():
            self.alphas = torch.sqrt(self.scheduler.alphas_cumprod).to(self.device, dtype=dtype)
            self.sigmas = torch.sqrt(1 - self.scheduler.alphas_cumprod).to(self.device, dtype=dtype)
        for p in self.unet.parameters():
            p.requires_grad = False

    def get_pipeline(self, model_id):
        return StableDiffusionPipeline.from_pretrained(model_id, ).to(self.device)

    def noise_to_timestep(self, z0, timestep, eps):
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        z_t = alpha_t * z0 + sigma_t * eps
        return z_t

    def prepare_latent_input_for_unet(self, z_t):
        return torch.cat([z_t] * 2)

    def predict_eps_and_sample(self, z_t, timestep, guidance_scale, text_embeddings):
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        latent_input = self.prepare_latent_input_for_unet(z_t)
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
            assert torch.isfinite(e_t).all()
        pred_z0 = (z_t - sigma_t * e_t) / alpha_t
        return e_t, pred_z0

    @torch.no_grad()
    def get_text_embeddings(self, text: str):
        tokens = self.pipeline.tokenizer([text], padding="max_length", max_length=77, truncation=True,
                                return_tensors="pt", return_overflowing_tokens=True).input_ids.to(self.device)
        return self.pipeline.text_encoder(tokens).last_hidden_state.detach()

    @torch.no_grad()
    def decode(self, latent):
        image = self.pipeline.vae.decode((1 / self.pipeline.vae.config.scaling_factor) * latent, return_dict=False)[0]
        image = denormalize(image)
        return Image.fromarray(image)


@torch.no_grad()
def denormalize(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image[0]

