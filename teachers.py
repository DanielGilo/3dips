from diffusion_pipeline import DiffusionPipeline, SevaPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionUpscalePipeline
from overrides import override
import torch
import numpy as np
import PIL
from PIL import Image
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from plot_utils import plot_frames_row, plot_frames
from controlnet_utils import get_canny_image, get_depth_estimation, get_mask_from_image_by_prompt
import torch.nn as nn


class Teacher(DiffusionPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def forward_operator(self, x):
        raise NotImplementedError
    
    def encode(self, x, height, width):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            x_enc = self.pipeline.vae.encode(x).latent_dist.mean * self.pipeline.vae.config.scaling_factor
        return x_enc
    


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
    def predict_eps_and_sample(self, z_t, timestep, guidance_scale, text_embeddings, batch_s, batch_e):
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        latent_input = self.prepare_latent_input_for_unet(z_t, batch_s, batch_e)
        timestep = torch.cat([timestep] * 2)
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(-1, *text_embeddings.shape[2:])
        embedd = torch.cat([embedd] * z_t.shape[0])
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
        plot_frames(self.forward_operator(self.gt_images), wb, "measurments", "measurments")

    @override
    def decode(self, latent, type="PIL", do_postprocess=True):
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.pipeline.upcast_vae()

        # Ensure latents are always the same type as the VAE
        latent = latent.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        images = self.vae.decode(latent / self.vae.config.scaling_factor, return_dict=False)[0]
        # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        #     images = self.vae.decode((1 / self.vae.config.scaling_factor) * latent, return_dict=False)[0]

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        
        if do_postprocess:
            images = self.pipeline.image_processor.postprocess(images, output_type=type)
        return images

        


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
        masks = np.asarray(self.mask_images)
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
        plot_frames(self.forward_operator(self.gt_images), wb, "measurments", "measurments")
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
    def predict_eps_and_sample(self, z_t, timestep, guidance_scale, text_embeddings, batch_s, batch_e):
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

        #image_batch_size = image.shape[0]
        # if image_batch_size == 1:
        #     repeat_by = batch_size
        # else:
        #     # image batch size is the same as prompt batch size
        #     repeat_by = num_images_per_prompt
        #
        # image = image.repeat_interleave(repeat_by, dim=0)

        cond_image = cond_image.to(device=self.device, dtype=self.dtype)
        #cond_image = torch.cat([cond_image] * 2)

        return cond_image

    @override
    def predict_eps_and_sample(self, z_t, timestep, guidance_scale, text_embeddings, batch_s, batch_e):
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]
        latent_input = self.prepare_latent_input_for_unet(z_t, batch_s, batch_e)
        timestep = torch.cat([timestep] * 2)
        embedd = text_embeddings.permute(1, 0, 2, 3).reshape(-1, *text_embeddings.shape[2:])
        embedd = torch.cat([embedd] * z_t.shape[0])
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
        plot_frames(self.forward_operator(self.gt_images), wb, "measurments", "measurments")
    

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
        plot_frames_row(self.forward_operator(self.gt_images), wb, "measurments", "measurments")
        plot_frames_row(self.forward_operator(self.gt_images[:self.n_input_frames]), wb, "input_views", "input_views")




def get_inpainting_teacher(**kwargs):
    init_images = kwargs["gt_images"]
    H, W = init_images[0].height, init_images[0].width
    # mask_tensor = torch.zeros((len(init_images),3,H,W), dtype=torch.uint8)
    # mask_tensor[...,H//4:3*H//4, W//4:3*W//4] = 255

    pil_mask = get_mask_from_image_by_prompt(init_images, "car")
    
    # masked_gt = torch_gt.clone()
    # masked_gt[mask_tensor.permute((0,2,3,1)) > 0.5] = 255
    # plot_frames_row(masked_gt.cpu().numpy(), wb, "masked_gt", "masked_gt")

    # pil_masks = [to_pil_image(mask_tensor[i]) for i in range(mask_tensor.shape[0])]

    teacher = InpaintingTeacher(init_images=init_images, mask_images=pil_mask,
                                           model_id="stabilityai/stable-diffusion-2-inpainting", device=kwargs["device"],
                                          dtype=torch.float16)
    return teacher



def get_teacher(teacher_name, **kwargs):
    if teacher_name == "text-conditioned":
        return TextConditionedTeacher(gt_images=kwargs["gt_images"], model_id="stabilityai/stable-diffusion-2-1",
                                       device=kwargs["device"], dtype=torch.float16, do_compile=kwargs["do_compile"])
    if teacher_name == "inpainting":
        return get_inpainting_teacher(**kwargs)
    if teacher_name == "upscale":
        return UpscaleTeacher(gt_images=kwargs["gt_images"], model_id="stabilityai/stable-diffusion-x4-upscaler",
                              device=kwargs["device"], dtype=torch.float16, do_compile=kwargs["do_compile"])
    if teacher_name == "canny":
        return CannyTeacher(gt_images=kwargs["gt_images"], controlnet_id="lllyasviel/sd-controlnet-canny",
                                         model_id="runwayml/stable-diffusion-v1-5", device=kwargs["device"], dtype=torch.float16, do_compile=kwargs["do_compile"])
    if teacher_name == "depth":
        return DepthTeacher(gt_images=kwargs["gt_images"], controlnet_id="lllyasviel/sd-controlnet-depth",
                                         model_id="runwayml/stable-diffusion-v1-5", device=kwargs["device"], dtype=torch.float16, do_compile=kwargs["do_compile"])
    if teacher_name == "oracle":
        return OracleTeacher(gt_images=kwargs["gt_images"], 
                             model_id="stabilityai/stable-diffusion-2-1", device=kwargs["device"], dtype=torch.float16, do_compile=kwargs["do_compile"])
    if teacher_name == "seva":
        return SevaTeacher(gt_images=kwargs["gt_images"], value_dict=kwargs["value_dict"], device=kwargs["device"], do_compile=kwargs["do_compile"])
    raise ValueError("Wrong teacher_name: {}".format(teacher_name))