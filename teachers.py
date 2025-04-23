from diffusion_pipeline import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from overrides import override
import torch
import PIL
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from plot_utils import plot_frames_row



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
        height, width = init_image[0].height, init_image[0].width
        #height, width = init_image.shape[-2:]
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

        del masked_image
        del mask
        torch.cuda.empty_cache()

        return mask_latents, masked_image_latents

    @override
    def prepare_latent_input_for_unet(self, z_t):
        b_size = z_t.shape[0]
        z_t = torch.cat([z_t] * 2)
        #mask_latents = torch.cat([self.mask_latents] * b_size)
        #masked_image_latents = torch.cat([self.masked_image_latents] * b_size)
        #return torch.cat([z_t, mask_latents, masked_image_latents], dim=1)
        return torch.cat([z_t, self.mask_latents, self.masked_image_latents], dim=1)
    


class OracleTeacher(Teacher):
    def __init__(self, gt_images, **kwargs):
        super().__init__(**kwargs)
        
        self.unet.to("cpu")
        torch.cuda.empty_cache()

        self.gt_images = gt_images # for plot input

        gt = self.pipeline.image_processor.preprocess(gt_images, height=gt_images[0].height, width=gt_images[0].width, crops_coords=None,
                                                          resize_mode="default").to(dtype=torch.float32).to(self.device)
        with torch.no_grad():
            self.gt_latents = self.pipeline.vae.encode(gt).latent_dist.mean * self.pipeline.vae.config.scaling_factor

    @override
    def predict_eps_and_sample(self, z_t, timestep, guidance_scale, text_embeddings):
        alpha_t = self.alphas[timestep, None, None, None]
        sigma_t = self.sigmas[timestep, None, None, None]

        # oracle "prediction"
        pred_z0 = self.gt_latents

        e_t = (z_t - alpha_t * pred_z0) / sigma_t
        return e_t, pred_z0
    
    def plot_input(self, wb):
        plot_frames_row(self.gt_images, wb, "GT", "GT")

    




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




def get_inpainting_teacher(**kwargs):
    # init_image = [PIL.Image.open("example_img.png").convert("RGB").resize((512, 512))]
    # mask_image = [PIL.Image.open("example_mask.png").convert("RGB").resize((512, 512))]
    teacher = InpaintingTeacher(init_image=kwargs["init_images"], mask_image=kwargs["mask_images"],
                                          model_id="stabilityai/stable-diffusion-2-inpainting", device=kwargs["device"],
                                          dtype=torch.float32)
    return teacher


def get_canny_teacher(**kwargs):
    init_image = PIL.Image.open("dog.png").convert("RGB").resize((512, 512))
    canny_image = get_canny_image(init_image)
    teacher = ControlNetTeacher(cond_image=canny_image, controlnet_id="lllyasviel/sd-controlnet-canny",
                                        model_id="runwayml/stable-diffusion-v1-5", device=kwargs["device"], dtype=torch.float32)
    return teacher


def get_depth_teacher(**kwargs):
    init_image = PIL.Image.open("dog.png").convert("RGB").resize((512, 512))
    depth_image = get_depth_estimation(init_image)
    teacher = ControlNetTeacher(cond_image=depth_image, controlnet_id="lllyasviel/sd-controlnet-depth",
                                         model_id="runwayml/stable-diffusion-v1-5", device=kwargs["device"], dtype=torch.float32)
    return teacher


def extract_gt_images_from_value_dict(value_dict):
    torch_gt = (((value_dict["cond_frames"] + 1) / 2.0)* 255).clamp(0, 255).to(torch.uint8)
    pil_gt = [to_pil_image(torch_gt[i]) for i in range(torch_gt.shape[0])]
    return pil_gt

def get_teacher(teacher_name, value_dict, **kwargs):
    if teacher_name == "text-conditioned":
        return Teacher(model_id="stabilityai/stable-diffusion-2-1", device=kwargs["device"], dtype=torch.float32)
    if teacher_name == "inpainting":
        return get_inpainting_teacher(**kwargs)
    if teacher_name == "canny":
        return get_canny_teacher(**kwargs)
    if teacher_name == "depth":
        return get_depth_teacher(**kwargs)
    if teacher_name == "oracle":
        return OracleTeacher(gt_images=extract_gt_images_from_value_dict(value_dict), 
                             model_id="stabilityai/stable-diffusion-2-1", device=kwargs["device"], dtype=torch.float32)
    raise ValueError("Wrong teacher_name: {}".format(teacher_name))