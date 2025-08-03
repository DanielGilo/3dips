from diffusion_pipeline import DiffusersStableDiffusionPipeline, DiffusionPipeline, SevaPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetPipeline, ControlNetModel, \
                    StableDiffusionUpscalePipeline, StableDiffusion3ControlNetPipeline, SD3ControlNetModel, \
                    StableDiffusionInstructPix2PixPipeline, BitsAndBytesConfig, SD3Transformer2DModel

from overrides import override
import torch
import numpy as np
import PIL
from PIL import Image
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
import torchvision.transforms as transforms
from plot_utils import plot_frames_row, plot_frames
from controlnet_utils import get_canny_image, get_depth_estimation, get_mask_from_image_by_prompt, get_random_mask_for_inpainting
import torch.nn as nn
from tqdm.auto import tqdm



class Teacher(DiffusersStableDiffusionPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def forward_operator(self, x):
        raise NotImplementedError
    
    def encode(self, x, height, width):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            x_enc = self.pipeline.vae.encode(x).latent_dist.mean * self.pipeline.vae.config.scaling_factor
        return x_enc
    



class InstructPixToPixTeacher(Teacher):
    def __init__(self, gt_images, editing_prompt, image_guidance_scale, **kwargs):
        super().__init__(**kwargs)

        self.gt_images = [image.resize(self.pixel_space_shape[1:]) for image in gt_images]

        self.pipeline.text_encoder.cuda()
        self.editing_prompt_embeds = self.pipeline._encode_prompt(
                editing_prompt,
                self.device,
                1,
                True,
                negative_prompt=None,
                prompt_embeds=None,
                negative_prompt_embeds=None,
            )
        self.unload_text_encoder()

        self.source_image_latents = self.prepare_image_latents(self.gt_images)
        self.image_guidance_scale = image_guidance_scale # needs to be >=1. Higher image guidance scale encourages generated images that are closely
                                        #linked to the source `image`, usually at the expense of lower image quality

    @override
    def get_pipeline(self, model_id):
        return StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device)

    def prepare_image_latents(self, source_images):
        source_images = self.pipeline.image_processor.preprocess(source_images).to(device=self.device, dtype=self.dtype)
        source_image_latents = self.vae.encode(source_images).latent_dist.mode()
        source_image_latents = torch.cat([source_image_latents], dim=0)
        uncond_image_latents = torch.zeros_like(source_image_latents)
        source_image_latents = torch.cat([source_image_latents, source_image_latents, uncond_image_latents], dim=0)
        return source_image_latents


    @override
    def prepare_latent_input_for_unet(self, z_t, batch_s, batch_e):
        z_t = torch.cat([z_t] * 3)
        #source_images = torch.cat([self.source_image_latents[batch_s:batch_e]])
        source_images = self.source_image_latents # TODO: need to consider batch_s, batch_e
        return torch.cat([z_t, source_images], dim=1)


    @override
    def predict_eps_and_sample(self, z_t, timestep, guidance_scale, batch_s, batch_e):
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        latent_input = self.prepare_latent_input_for_unet(z_t, batch_s, batch_e)
        timestep = torch.cat([timestep] * 2)
        text_embeddings = torch.cat([self.editing_prompt_embeds.unsqueeze(0)] * z_t.shape[0])
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(-1, *text_embeddings.shape[2:])
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # predict the noise residual
            noise_pred = self.unet(
                latent_input,
                timestep[0],
                encoder_hidden_states=embedd,
                added_cond_kwargs=None,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]

            # perform guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_image)
                + self.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )

        pred_z0 = (z_t - sigma_t * noise_pred) / alpha_t
        return noise_pred, pred_z0
    
    def plot_input(self, wb):
        plot_frames(self.gt_images, wb, "GT", "GT")
        plot_frames(self.forward_operator(self.gt_images), wb, "measurments", "measurments", save_as_pdf=True)

    @override
    def forward_operator(self, x):
        return x



class TextConditionedTeacher(Teacher):
    def __init__(self, gt_images, **kwargs):
        super().__init__(**kwargs)
        self.gt_images = gt_images # for plot_input only
    
    @override
    def forward_operator(self, x):
        return x

    def plot_input(self, wb):
         plot_frames_row(self.gt_images, wb, "GT", "GT")




class UpscaleTeacher(Teacher):
    def __init__(self, gt_images, **kwargs):
        super().__init__(**kwargs)
        self.gt_images = [image.resize(self.pixel_space_shape[1:]) for image in gt_images] # for plot input
        self.low_res_images = self.prepare_low_res_images(self.gt_images)
        noise_level = torch.tensor([0.0], dtype=torch.long, device=self.device)  # default in pipeline is 20 - not sure why
        self.noise_level = torch.cat([noise_level] * self.low_res_images.shape[0] * 2)

    def prepare_low_res_images(self, images):
        low_res_images = self.forward_operator(images)
        low_res_images = self.pipeline.image_processor.preprocess(low_res_images).to(dtype=self.dtype, device=self.device)

        return low_res_images

    @override
    def get_pipeline(self, model_id):
        return StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device)

    # original - only downscaling!
    @override
    def forward_operator(self, x):
        if isinstance(x, np.ndarray):
            if x.dtype != np.uint8:
                assert x.min() >= 0 and x.max() <= 1, "Float images should be in range [0, 1]"
                x = (x * 255).astype(np.uint8)
            x = [Image.fromarray(img) for img in x]
            
        low_res_x = [img.resize((self.latent_shape[1], self.latent_shape[2])) for img in x] # latent shape matches low-res shape
        return low_res_x

    @override
    def prepare_latent_input_for_unet(self, z_t, batch_s, batch_e):
        z_t = torch.cat([z_t] * 2)
        low_res_imgs = torch.cat([self.low_res_images[batch_s:batch_e]] * 2)
        return torch.cat([z_t, low_res_imgs], dim=1)

    @override
    def predict_eps_and_sample(self, z_t, timestep, guidance_scale, batch_s, batch_e):
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        latent_input = self.prepare_latent_input_for_unet(z_t, batch_s, batch_e)
        timestep = torch.cat([timestep] * 2)
        text_embeddings = torch.cat([self.text_embeddings] * z_t.shape[0])
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(-1, *text_embeddings.shape[2:])
        timestep = torch.cat([timestep] * z_t.shape[0])
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            e_t = self.unet(latent_input, timestep, embedd, class_labels=self.noise_level).sample
            if self.prediction_type == 'v_prediction':
                e_t = torch.cat([alpha_t] * timestep.shape[0]) * e_t + torch.cat([sigma_t] * timestep.shape[0]) * torch.cat([z_t] * 2)
            e_t_uncond, e_t = e_t.chunk(2)
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
            #assert torch.isfinite(e_t).all()
        pred_z0 = (z_t - sigma_t * e_t) / alpha_t
        return e_t, pred_z0
    
    def plot_input(self, wb):
        plot_frames(self.gt_images, wb, "GT", "GT")
        plot_frames(self.forward_operator(self.gt_images), wb, "measurments", "measurments", save_as_pdf=True)

    @override
    def decode(self, latent, type="PIL", do_postprocess=True):
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.pipeline.upcast_vae()

        # Ensure latents are always the same type as the VAE
        latent = latent.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        images = self.vae.decode(latent / self.vae.config.scaling_factor, return_dict=False)[0]

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        
        if do_postprocess:
            images = self.pipeline.image_processor.postprocess(images, output_type=type)
        return images

        
class DeblurByUpscaleTeacher(UpscaleTeacher):
    def __init__(self, gt_images, **kwargs):
        self.blur_sigma = 11.0

        super().__init__(gt_images, **kwargs)

        neg_prompt = "blurry, low quality, deformed, distorted, bad quality, low resolution, pixelated, noisy, blurry edges"
        self.pipeline.text_encoder.cuda()
        self.text_embeddings = torch.stack([self.get_text_embeddings(neg_prompt), self.get_text_embeddings(kwargs["prompt"])], dim=1)
        self.unload_text_encoder

    @override
    def prepare_low_res_images(self, images):
        blurred_images = self.forward_operator(images)
        low_res_images = [img.resize((self.latent_shape[1], self.latent_shape[2])) for img in blurred_images] # latent shape matches low-res shape
        low_res_images = self.pipeline.image_processor.preprocess(low_res_images).to(dtype=self.dtype, device=self.device)

        return low_res_images
    
    # blurring 
    @override
    def forward_operator(self, x):
        if isinstance(x, np.ndarray):
            if x.dtype != np.uint8:
                assert x.min() >= 0 and x.max() <= 1, "Float images should be in range [0, 1]"
                x = (x * 255).astype(np.uint8)
            x = [Image.fromarray(img) for img in x]
        gaussian_blur = transforms.GaussianBlur(kernel_size=49, sigma=self.blur_sigma)
        return [gaussian_blur(img) for img in x]


class InpaintingTeacher(Teacher):
    def __init__(self, init_images, mask_images, **kwargs):
        super().__init__(**kwargs)
        self.mask_images = [image.resize(self.pixel_space_shape[1:]) for image in mask_images] # for plot input
        self.gt_images = [image.resize(self.pixel_space_shape[1:]) for image in init_images] # for plot input
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.pipeline.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        self.mask_latents, self.masked_image_latents = self.prepare_image_and_mask_latents(self.gt_images, self.mask_images)


    @override
    def get_pipeline(self, model_id):
        return StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device)

    @torch.no_grad()
    def prepare_image_and_mask_latents(self, init_image, mask_image):
        height, width = init_image[0].height, init_image[0].width
        #height, width = init_image.shape[-2:]
        init_image = self.pipeline.image_processor.preprocess(init_image, height=height, width=width, crops_coords=None,
                                                          resize_mode="default").to(dtype=self.dtype)
        mask_condition = self.mask_processor.preprocess(mask_image, height=height, width=width, crops_coords=None,
                                                          resize_mode="default").to(dtype=self.dtype)
        masked_image = init_image * (mask_condition < 0.5)

        # resize mask to latent dimension
        mask = torch.nn.functional.interpolate(
            mask_condition, size=(height // self.pipeline.vae_scale_factor, width // self.pipeline.vae_scale_factor)
        )
        mask = mask.to(device=self.device)

        masked_image = masked_image.to(device=self.device)
        masked_image_latent = self.pipeline.vae.encode(masked_image).latent_dist.mean * self.pipeline.vae.config.scaling_factor

        mask_latents = mask
        masked_image_latents = masked_image_latent

        del masked_image
        del mask
        torch.cuda.empty_cache()

        return mask_latents, masked_image_latents

    @override
    def prepare_latent_input_for_unet(self, z_t, batch_s, batch_e):
        z_t = torch.cat([z_t] * 2)
        mask_latents = torch.cat([self.mask_latents[batch_s:batch_e]] * 2)
        masked_image_latents = torch.cat([self.masked_image_latents[batch_s:batch_e]] * 2)

        return torch.cat([z_t, mask_latents, masked_image_latents], dim=1)
    
    @override
    def forward_operator(self, x):
        masked_x = np.asarray(x).copy()
        if masked_x.dtype != np.uint8:
            assert masked_x.min() >= 0 and masked_x.max() <= 1, "Float images should be in range [0, 1]"
            masked_x = (masked_x * 255).astype(np.uint8)
        mask_resized = [mask.resize((masked_x.shape[1], masked_x.shape[2])) for mask in self.mask_images]
        masks = np.asarray(mask_resized)
        masked_x[masks > 0.5] = 255

        return masked_x

    # @override
    # def encode(self, x, height, width):
    #     mask = torch.stack([pil_to_tensor(mask_image) for mask_image in self.mask_image]).to(self.device)
    #     gt = torch.stack([pil_to_tensor(gt_image) for gt_image in self.gt_images]).to(self.device)
    #     gt = (gt - gt.max()) / (gt.max() - gt.min())
    #     x = (x * (mask > 0.5)) + (gt * (mask < 0.5))
    #     return super().encode(x, height, width)
    
    def plot_input(self, wb):
        plot_frames(self.forward_operator(self.gt_images), wb, "measurments", "measurments", save_as_pdf=True)
        plot_frames(self.gt_images, wb, "GT", "GT")


class OracleTeacher(Teacher):
    def __init__(self, gt_images, **kwargs):
        super().__init__(**kwargs)
        
        self.unet.to("cpu")
        torch.cuda.empty_cache()

        self.gt_images = gt_images # for plot input

        with torch.no_grad():
            self.gt_latents = self.encode(gt_images, width=gt_images[0].width, height=gt_images[0].height)

    @override
    def predict_eps_and_sample(self, z_t, timestep, guidance_scale, batch_s, batch_e):
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]

        # oracle "prediction"
        pred_z0 = self.gt_latents

        e_t = (z_t - alpha_t * pred_z0) / sigma_t
        return e_t, pred_z0
    
    def plot_input(self, wb):
        plot_frames_row(self.gt_images, wb, "GT", "GT")

    @override
    def forward_operator(self, x):
        """
        @param x: a torch tensor in image space of shape [b, 3, H, W]
        """
        return x



class ControlNetTeacher(Teacher):
    def __init__(self, gt_images, controlnet_id, **kwargs):
        self.controlnet_id = controlnet_id
        super().__init__(**kwargs)

        self.gt_images = [image.resize(self.pixel_space_shape[1:]) for image in gt_images] # for plot input
        cond_image = self.forward_operator(self.gt_images)

        self.vae_scale_factor = 2 ** (len(self.pipeline.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )

        self.cond_image = self.prepare_cond_image(cond_image)

    @override
    def get_pipeline(self, model_id):
        controlnet = ControlNetModel.from_pretrained(self.controlnet_id, torch_dtype=self.dtype).to(self.device)
        #controlnet.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=self.dtype)
        return StableDiffusionControlNetPipeline.from_pretrained(model_id, controlnet=controlnet, torch_dtype=self.dtype).to(self.device)

    @torch.no_grad()
    def prepare_cond_image(self, cond_image):
        height, width = cond_image[0].height, cond_image[0].width
        cond_image = self.control_image_processor.preprocess(cond_image, height=height, width=width).to(dtype=self.dtype)
        cond_image = cond_image.to(device=self.device, dtype=self.dtype)

        return cond_image

    @override
    def predict_eps_and_sample(self, z_t, timestep, guidance_scale, batch_s, batch_e):
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        latent_input = self.prepare_latent_input_for_unet(z_t, batch_s, batch_e)
        timestep = torch.cat([timestep] * 2)
        text_embeddings = torch.cat([self.text_embeddings] * z_t.shape[0])
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(-1, *text_embeddings.shape[2:])
        timestep = torch.cat([timestep] * z_t.shape[0])
        controlnet_cond = torch.cat([self.cond_image[batch_s:batch_e]] * 2)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            down_block_res_samples, mid_block_res_sample = self.pipeline.controlnet(
                latent_input,
                timestep,
                encoder_hidden_states=embedd,
                controlnet_cond=controlnet_cond,
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
    
    def plot_input(self, wb):
        plot_frames(self.gt_images, wb, "GT", "GT")
        plot_frames(self.forward_operator(self.gt_images), wb, "measurments", "measurments", save_as_pdf=True)
    

class DepthTeacher(ControlNetTeacher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @override
    def forward_operator(self, x):
        """
        @param x: a torch tensor in image space of shape [b, 3, H, W]
        """
        return get_depth_estimation(x)
    

class CannyTeacher(ControlNetTeacher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @override
    def forward_operator(self, x):
        """
        @param x: a torch tensor in image space of shape [b, 3, H, W]
        """
        return get_canny_image(x)


class StableDiffusion3ControlNetTeacher(DiffusionPipeline):
    def __init__(self, gt_images, prompt, controlnet_id, **kwargs):
        self.controlnet_id = controlnet_id

        super().__init__(kwargs["model_id"], kwargs["device"], kwargs["dtype"])
        self.latent_shape = [self.pipeline.vae.config.latent_channels,
                             self.pipeline.transformer.config.sample_size,
                             self.pipeline.transformer.config.sample_size]
        self.pixel_space_shape = [3, self.latent_shape[1] * self.pipeline.vae_scale_factor,
                                  self.latent_shape[2] * self.pipeline.vae_scale_factor]

        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae

        # from https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_3#performance-optimizations-for-sd3
        if kwargs["do_compile"]:
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = False
            torch._inductor.config.coordinate_descent_check_all_directions = True

            self.transformer.to(memory_format=torch.channels_last)
            self.vae.to(memory_format=torch.channels_last)
            self.transformer = torch.compile(self.transformer, mode="max-autotune", fullgraph=True)
            self.vae.decode = torch.compile(self.vae.decode, mode="max-autotune", fullgraph=True)

        #self.prediction_type = self.pipeline.scheduler.prediction_type
        for p in self.transformer.parameters():
            p.requires_grad = False
        for p in self.pipeline.text_encoder.parameters():
            p.requires_grad = False
        for p in self.pipeline.text_encoder_2.parameters():
            p.requires_grad = False
        for p in self.pipeline.text_encoder_3.parameters():
            p.requires_grad = False

        # not sure about those! https://github.com/huggingface/diffusers/blob/v0.34.0/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L459
        self.sigmas = torch.flip(self.pipeline.scheduler.sigmas.to(self.device, dtype=self.dtype), dims=(0,))
        #self.alphas = 1.0 - self.sigmas

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        if self.pipeline.controlnet.config.force_zeros_for_pooled_projection:
            self.vae_shift_factor = 0
        else:
            self.vae_shift_factor = self.vae.config.shift_factor
        self.vae.enable_slicing()

        # self.gt_images are resized for measuring PSNR etc, but cond image is first blurred in forward_operator and then resized in prepare_cond_image
        self.gt_imgs_orig_size = gt_images # for input plot
        self.gt_images = [image.resize(self.pixel_space_shape[1:]) for image in gt_images] # for comparison
        cond_image = self.forward_operator(gt_images)
        self.cond_image = self.prepare_cond_image(cond_image)

        self.prompt_embeds, _, self.pooled_prompt_embeds, _ = self.pipeline.encode_prompt(prompt=prompt,prompt_2=prompt, prompt_3=prompt ,do_classifier_free_guidance=True)
        
        # self.pipeline.text_encoder.cpu()
        # self.pipeline.text_encoder_2.cpu()
        # self.pipeline.text_encoder_3.cpu()

        self.pipeline.text_encoder = None
        self.pipeline.text_encoder_2 = None
        self.pipeline.text_encoder_3 = None
        torch.cuda.empty_cache()

    def get_pipeline(self, model_id):
        nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype)
        model_nf4 = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=self.dtype)

        controlnet = SD3ControlNetModel.from_pretrained(self.controlnet_id, torch_dtype=self.dtype).to(self.device)
        return StableDiffusion3ControlNetPipeline.from_pretrained(model_id, transformer=model_nf4, 
                                                                  controlnet=controlnet, torch_dtype=self.dtype).to(self.device)

    @torch.no_grad()
    def prepare_cond_image(self, cond_image):
        height, width = cond_image[0].height, cond_image[0].width
        cond_image = self.image_processor.preprocess(cond_image, height=1024, width=1024).to(dtype=self.dtype)
        cond_image = cond_image.to(device=self.device, dtype=self.dtype)

        cond_image = self.vae.encode(cond_image).latent_dist.sample()
        cond_image = (cond_image - self.vae_shift_factor) * self.vae.config.scaling_factor

        return cond_image

    def predict_eps_and_sample(self, z_t, timestep, guidance_scale, batch_s, batch_e):
        sigma_t = self.sigmas[timestep]
        latent_input = self.prepare_latent_input_for_unet(z_t, batch_s, batch_e)
        timestep = torch.cat([timestep] * 2)
        timestep = torch.cat([timestep] * z_t.shape[0])
        controlnet_cond = torch.cat([self.cond_image[batch_s:batch_e]] * 2)

        if self.pipeline.controlnet.config.force_zeros_for_pooled_projection:
            # instantx sd3 controlnet used zero pooled projection
            controlnet_pooled_projections = torch.zeros_like(self.pooled_prompt_embeds)
        else:
            controlnet_pooled_projections = self.pooled_prompt_embeds

        if self.pipeline.controlnet.config.joint_attention_dim is not None:
            controlnet_encoder_hidden_states = self.prompt_embeds
        else:
            # SD35 official 8b controlnet does not use encoder_hidden_states
            controlnet_encoder_hidden_states = None

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # controlnet(s) inference
            control_block_samples = self.pipeline.controlnet(
                hidden_states=latent_input,
                timestep=timestep,
                encoder_hidden_states=controlnet_encoder_hidden_states,
                pooled_projections=controlnet_pooled_projections,
                joint_attention_kwargs=None,
                controlnet_cond=controlnet_cond,
                conditioning_scale=1.0,
                return_dict=False,
            )[0]

            e_t = self.transformer(
                hidden_states=latent_input,
                timestep=timestep,
                encoder_hidden_states=self.prompt_embeds,
                pooled_projections=self.pooled_prompt_embeds,
                block_controlnet_hidden_states=control_block_samples,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            e_t_uncond, e_t = e_t.chunk(2)
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
            assert torch.isfinite(e_t).all()
        pred_z0 = z_t - sigma_t * e_t
        return e_t, pred_z0
    
    def multi_step_sample(self, z_t, guidance_scale, initial_timestep, n_steps, batch_s, batch_e):
        # Deterministic EulerEDM
        latents = z_t.clone()
        n_steps += 1
        timesteps = torch.linspace(initial_timestep, 0, steps=n_steps, device="cuda", dtype=torch.int)
        
        sigmas = timesteps / 1000
        sigmas = self.pipeline.scheduler.shift * sigmas / (1 + (self.pipeline.scheduler.shift - 1) * sigmas)
        timesteps = (sigmas * 1000).to(dtype=torch.int)


        for i in tqdm(range(n_steps-1)):
            t = timesteps[i]

            noise_pred, _ = self.predict_eps_and_sample(latents, t.unsqueeze(0), guidance_scale, batch_s, batch_e)
            # Upcast to avoid precision issues when computing prev_sample
            noise_pred = noise_pred.to(torch.float32)

            prev_t = timesteps[i+1]

            sigma_t = sigmas[i]
            sigma_t_prev = sigmas[i+1]

            dt = sigma_t_prev - sigma_t

            #print(f"from t={t.item()} to t={prev_t.item()}")

            latents = latents + dt * noise_pred

        latents = latents.to(self.dtype)
        eps = (z_t - latents) / self.sigmas[initial_timestep]
        return eps, latents
    
    def noise_to_timestep(self, z0, timestep, eps):
        sigma_t = self.sigmas[timestep]
        z_t = (1-sigma_t) * z0 + sigma_t * eps
        return z_t

    def decode(self, latent, type="pil", do_postprocess=True):
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            latent = (latent / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            images = self.vae.decode(latent, return_dict=False)[0].to(device="cpu")
        if do_postprocess:
            images = self.pipeline.image_processor.postprocess(images, output_type=type)
        return images
    
    def plot_input(self, wb):
        plot_frames(self.gt_imgs_orig_size, wb, "GT", "GT")
        plot_frames(self.forward_operator(self.gt_imgs_orig_size), wb, "measurments", "measurments", save_as_pdf=True)


class DeblurTeacherSD3(StableDiffusion3ControlNetTeacher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_operator(self, x):
        # assuming x is a PIL image 
        if isinstance(x, np.ndarray):
            if x.dtype != np.uint8:
                assert x.min() >= 0 and x.max() <= 1, "Float images should be in range [0, 1]"
                x = (x * 255).astype(np.uint8)
            x = [Image.fromarray(img) for img in x]
        gaussian_blur = transforms.GaussianBlur(kernel_size=49, sigma=2.0)
        return [gaussian_blur(img) for img in x]


class SevaTeacher(SevaPipeline):
    def __init__(self, gt_images, **kwargs):
        super().__init__(**kwargs)
        self.gt_images = gt_images # for input plot only

    def forward_operator(self, x):
        return x
    
    @override
    def predict_eps_and_sample(self, z_t, timestep, guidance_scale=1.0):
        pred_eps, pred_z0 =  super().predict_eps_and_sample(z_t, timestep, guidance_scale)
        return self.get_only_output_frames(pred_eps), self.get_only_output_frames(pred_z0)

    def plot_input(self, wb):
        plot_frames_row(self.gt_images, wb, "GT", "GT")
        plot_frames_row(self.forward_operator(self.gt_images), wb, "measurments", "measurments", save_as_pdf=True)
        plot_frames_row(self.forward_operator(self.gt_images[:self.n_input_frames]), wb, "input_views", "input_views")




def get_inpainting_teacher(dtype=torch.float16, masked_obj="road", **kwargs):
    init_images = kwargs["gt_images"]
    H, W = init_images[0].height, init_images[0].width

    #pil_mask = get_mask_from_image_by_prompt(init_images, masked_obj)
    pil_mask = get_random_mask_for_inpainting(init_images, mask_size=(128, 128), num_masks=5) 
    
    teacher = InpaintingTeacher(init_images=init_images, mask_images=pil_mask, prompt=kwargs["prompt"],
                                           model_id="stabilityai/stable-diffusion-2-inpainting", device=kwargs["device"],
                                          dtype=dtype)
    return teacher



def get_teacher(teacher_name, **kwargs):
    if teacher_name == "text-conditioned":
        return TextConditionedTeacher(gt_images=kwargs["gt_images"], prompt=kwargs["prompt"], 
                                      model_id="stabilityai/stable-diffusion-2-1",
                                       device=kwargs["device"], dtype=torch.float16, do_compile=kwargs["do_compile"])
    if teacher_name == "inpainting":
        return get_inpainting_teacher(**kwargs)
    if teacher_name == "upscale":
        return UpscaleTeacher(gt_images=kwargs["gt_images"], prompt=kwargs["prompt"],
                              model_id="stabilityai/stable-diffusion-x4-upscaler",
                              device=kwargs["device"], dtype=torch.float16, do_compile=kwargs["do_compile"])
    if teacher_name == "canny":
        return CannyTeacher(gt_images=kwargs["gt_images"], prompt=kwargs["prompt"],controlnet_id="lllyasviel/sd-controlnet-canny",
                                         model_id="runwayml/stable-diffusion-v1-5", device=kwargs["device"], dtype=torch.float16, do_compile=kwargs["do_compile"])
    if teacher_name == "depth":
        return DepthTeacher(gt_images=kwargs["gt_images"], prompt=kwargs["prompt"], controlnet_id="lllyasviel/sd-controlnet-depth",
                                         model_id="runwayml/stable-diffusion-v1-5", device=kwargs["device"], dtype=torch.float16, do_compile=kwargs["do_compile"])
    if teacher_name == "pix2pix":
        return InstructPixToPixTeacher(gt_images=kwargs["gt_images"],prompt=kwargs["prompt"],
                                        editing_prompt=kwargs["editing_prompt"], image_guidance_scale=kwargs["image_guidance_scale"],
                                       model_id="timbrooks/instruct-pix2pix", device=kwargs["device"],
                                       dtype=torch.float16, do_compile=kwargs["do_compile"])
    if teacher_name == "deblur":
        return DeblurByUpscaleTeacher(gt_images=kwargs["gt_images"], prompt=kwargs["prompt"],
                              model_id="stabilityai/stable-diffusion-x4-upscaler",
                              device=kwargs["device"], dtype=torch.float16, do_compile=kwargs["do_compile"])
    if teacher_name == "oracle":
        return OracleTeacher(gt_images=kwargs["gt_images"], prompt=kwargs["prompt"],
                             model_id="stabilityai/stable-diffusion-2-1", device=kwargs["device"], dtype=torch.float16, do_compile=kwargs["do_compile"])
    if teacher_name == "seva":
        return SevaTeacher(gt_images=kwargs["gt_images"], value_dict=kwargs["value_dict"], device=kwargs["device"], do_compile=kwargs["do_compile"])
    raise ValueError("Wrong teacher_name: {}".format(teacher_name))
