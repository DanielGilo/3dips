from teachers import CannyTeacher, TextConditionedTeacher
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from tqdm.auto import tqdm


# Sample function (regular DDIM)
@torch.no_grad()
def sample(
    pipe,
    prompt,
    start_step=0,
    start_latents=None,
    guidance_scale=3.5,
    num_inference_steps=30,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device="cuda",
):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    for i in tqdm(range(start_step, num_inference_steps)):

        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Normally we'd rely on the scheduler to handle the update step:
        # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Instead, let's do it ourselves:
        prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt

    # Post-processing
    images = pipe.decode_latents(latents)
    images = pipe.numpy_to_pil(images)

    return images

gt_images = [Image.open("figs/gt0.png")]
prompt = "a vase with a plant on top of a table in a garden, realistic, high quality, detailed, 4k"

with torch.no_grad():
    dtype = torch.float16
    # teacher = CannyTeacher(gt_images=gt_images, controlnet_id="lllyasviel/sd-controlnet-canny",
    #                     model_id="runwayml/stable-diffusion-v1-5", device="cuda", dtype=torch.float16, do_compile=False)
    teacher = teacher = TextConditionedTeacher(gt_images=gt_images, model_id="runwayml/stable-diffusion-v1-5",
                                       device="cuda", dtype=dtype, do_compile=False)
    teacher.pipeline.scheduler = DDIMScheduler.from_config(teacher.pipeline.scheduler.config)


    # text_embeddings = torch.stack([teacher.get_text_embeddings(""), teacher.get_text_embeddings(prompt)], dim=1)


    z_t = torch.randn((1, 4, 64, 64), device="cuda", dtype=dtype)
    T = torch.ones(z_t.shape[0]) * 999
    # z_0 = teacher.ddim_sample(z_t, 7.5, text_embeddings, T, 100, 0, 1)

    z_0 = teacher.ddim_sample(z_t, 7.5, prompt, 0, 1000, 0, 1)


    x0 = teacher.decode(z_0.detach(), "pil", do_postprocess=True)

    x0[0].save("output_figs/teacher_ddim.png")
    

    # x0 = sample(pipe=teacher.pipeline, prompt=prompt, guidance_scale=7.5)
    # x0[0].save("output_figs/teacher_ddim_using_sample.png")



# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# prompt = "a vase with a plant on top of a table in a garden"

# # im = pipe(prompt=prompt,).images[0]
# # im.save("output_figs/sd2_1_diffusers.png")

# x0 = sample(pipe = pipe, prompt=prompt,
#     start_step=0,
#     start_latents=None,
#     guidance_scale=7.5,
#     num_inference_steps=50,
#     num_images_per_prompt=1,
#     do_classifier_free_guidance=True,
#     negative_prompt="",
#     device="cuda"
# )

# x0[0].save("output_figs/sd1_5_sample.png")
# print("Sampled image saved as output_figs/sd1_5_sample.png")