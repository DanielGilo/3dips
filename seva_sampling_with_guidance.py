import torch
from PIL import Image
from torchvision import transforms
from diffusion_pipeline import SevaPipeline
import os
import re
from seva_utils import get_value_dict_of_scene
from plot_utils import plot_frames


def load_reference_images(directory, n_views_total, new_size=(576, 576)):
    images = []
    transform = transforms.Compose([
        transforms.Resize(new_size),
        transforms.ToTensor(),            # Converts to [0, 1]
        transforms.Normalize((0.5,), (0.5,))  # Scales to [-1, 1]
    ])

    # Compile regex to match: final_z0_frame_<n>*.png where n ∈ [0, 7]
    pattern = re.compile(r"^final_z0_frame_([0-{}])\D.*\.png$".format(n_views_total - 1))
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            filepath = os.path.join(directory, filename)
            img = Image.open(filepath).convert("RGB")
            tensor = transform(img)
            images.append((int(match.group(1)), tensor))

    # Sort by the frame index (0–7) to keep consistent order
    images.sort(key=lambda x: x[0])
    return torch.stack([tensor for _, tensor in images])

device = "cuda"
t_sample = 300
sampling_guidance_scale = 1

scene_path = "/home/danielgilo/3dips/seva/assets_demo_cli/dl3d140-165f5af8bfe32f70595a1c9393a6e442acf7af019998275144f605b89a306557"
num_inputs = 1

#num_views_total = 9
#input_json = os.path.join(scene_path, f"{num_views_total}_train_test_split_1.json")
#output_json = os.path.join(scene_path, "train_test_split_1.json")
#shutil.copyfile(input_json, output_json)

value_dict = get_value_dict_of_scene(scene_path, num_inputs)

# Load reference frames
x0 = load_reference_images("wandb/run-20250726_231022-68bmbzun/files/media/images/", value_dict["num_imgs_no_padding"])
padding = torch.cat([x0[-1:]] * (21 - x0.shape[0]))
x0 = torch.cat([x0, padding])

s_in = torch.ones([21])

with torch.no_grad():
    seva = SevaPipeline(device=device, value_dict=value_dict, do_compile=False)
    seva.model.eval() 

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        z0 = seva.ae.encode(x0.cuda())

    timestep = t_sample * s_in
    eps = torch.randn_like(z0, device="cuda")
    sample_dt = 25
    n_sample_timesteps = t_sample // sample_dt

    
    latent = seva.noise_to_timestep(z0, timestep, eps)

# with grad due to guidance
#latent = seva.euler_edm_sample_guided_latents(latent, timestep, n_sample_timesteps, 2.0, reference_latent=z0, guidance_scale=sampling_guidance_scale)
latent = seva.euler_edm_sample_guided_pixels(latent, timestep, n_sample_timesteps, 2.0, reference_frames=x0, guidance_scale=sampling_guidance_scale)


with torch.no_grad():
    frames = seva.decode(latent, type="np")
    frames_out = seva.get_only_output_frames(frames)
    plot_frames(frames_out, None, f"guidance_res_{t_sample}_{sampling_guidance_scale}", "t=0", save_as_pdf=True)

    before_guidance = seva.decode(z0, type="np")
    before_guidance_out = seva.get_only_output_frames(before_guidance)
    plot_frames(before_guidance_out, None, "before_guidance", "t=0", save_as_pdf=True)
