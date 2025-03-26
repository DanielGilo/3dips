from diffusion_pipeline import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers import StableDiffusionInpaintPipeline
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
        self.prepare_image_and_mask_latents(init_image, mask_image)

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

        self.mask_latents = torch.cat([mask] * 2).to(device=self.device)
        self.masked_image_latents = torch.cat([masked_image_latent] * 2).to(device=self.device)

    @override
    def prepare_latent_input_for_unet(self, z_t):
        b_size = z_t.shape[0]
        z_t = torch.cat([z_t] * 2)
        mask_latents = torch.cat([self.mask_latents] * b_size)
        masked_image_latents = torch.cat([self.masked_image_latents] * b_size)
        return torch.cat([z_t, mask_latents, masked_image_latents], dim=1)

