from diffusion_pipeline import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from overrides import override
import torch


class Teacher(DiffusionPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class InpaintingTeacher(Teacher):
    def __init__(self, init_image, mask_image, **kwargs):
        super().__init__(**kwargs)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.pipeline.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        self.mask_latents, self.masked_image_latents = self.prepare_image_and_mask_latents(init_image, mask_image)

    @override
    def get_pipeline(self, model_id):
        return StableDiffusionInpaintPipeline.from_pretrained(model_id, ).to(self.device)

    @torch.no_grad()
    def prepare_image_and_mask_latents(self, init_image, mask_image):
        height, width = init_image.height, init_image.width
        init_image = self.pipeline.image_processor.preprocess(init_image, height=height, width=width, crops_coords=None,
                                                          resize_mode="default").to(dtype=torch.float32)
        mask_condition = self.mask_processor.preprocess(mask_image, height=height, width=width, crops_coords=None,
                                                          resize_mode="default").to(dtype=torch.float32)
        masked_image = init_image * (mask_condition < 0.5)

        # resize mask to latent dimension
        mask = torch.nn.functional.interpolate(
            mask_condition, size=(height // self.pipeline.vae_scale_factor, width // self.pipeline.vae_scale_factor)
        )
        mask = mask.to(device=self.device)

        masked_image = masked_image.to(device=self.device)
        masked_image_latent = self.pipeline.vae.encode(masked_image).latent_dist.mean * self.pipeline.vae.config.scaling_factor

        mask_latents = torch.cat([mask] * 2).to(device=self.device)
        masked_image_latents = torch.cat([masked_image_latent] * 2).to(device=self.device)

        return mask_latents, masked_image_latents

    @override
    def prepare_latent_input_for_unet(self, z_t):
        b_size = z_t.shape[0]
        z_t = torch.cat([z_t] * 2)
        mask_latents = torch.cat([self.mask_latents] * b_size)
        masked_image_latents = torch.cat([self.masked_image_latents] * b_size)
        return torch.cat([z_t, mask_latents, masked_image_latents], dim=1)



class ControlNetTeacher(Teacher):
    def __init__(self, cond_image, controlnet_id, **kwargs):
        self.controlnet_id = controlnet_id
        super().__init__(**kwargs)
        self.vae_scale_factor = 2 ** (len(self.pipeline.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        self.cond_image = self.prepare_cond_image(cond_image)

    @override
    def get_pipeline(self, model_id):
        controlnet = ControlNetModel.from_pretrained(self.controlnet_id, ).to(self.device)
        return StableDiffusionControlNetPipeline.from_pretrained(model_id, controlnet=controlnet).to(self.device)

    @torch.no_grad()
    def prepare_cond_image(self, cond_image):
        height, width = cond_image.height, cond_image.width
        cond_image = self.control_image_processor.preprocess(cond_image, height=height, width=width).to(dtype=torch.float32)

        #image_batch_size = image.shape[0]
        # if image_batch_size == 1:
        #     repeat_by = batch_size
        # else:
        #     # image batch size is the same as prompt batch size
        #     repeat_by = num_images_per_prompt
        #
        # image = image.repeat_interleave(repeat_by, dim=0)

        cond_image = cond_image.to(device=self.device, dtype=torch.float32)
        cond_image = torch.cat([cond_image] * 2)

        return cond_image

    @override
    def predict_eps_and_sample(self, z_t, timestep, guidance_scale, text_embeddings):
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        latent_input = self.prepare_latent_input_for_unet(z_t)
        timestep = torch.cat([timestep] * 2)
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(-1, *text_embeddings.shape[2:])
        embedd = torch.cat([embedd] * z_t.shape[0])
        timestep = torch.cat([timestep] * z_t.shape[0])
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            down_block_res_samples, mid_block_res_sample = self.pipeline.controlnet(
                latent_input,
                timestep,
                encoder_hidden_states=embedd,
                controlnet_cond=self.cond_image,
                conditioning_scale=1.0,
                guess_mode=False,
                return_dict=False,
            )

            e_t = self.unet(latent_input, timestep, embedd, down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample).sample
            e_t_uncond, e_t = e_t.chunk(2)
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
            assert torch.isfinite(e_t).all()
        pred_z0 = (z_t - sigma_t * e_t) / alpha_t
        return e_t, pred_z0