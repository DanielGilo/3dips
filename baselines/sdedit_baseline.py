
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] =":4096:8" # necessary when setting torch.use_dererministic_algorithms(True), increases memory by 24MB


from diffusion_pipeline import SevaPipeline
import argparse
from teachers import get_teacher
from seva_utils import get_value_dict_of_scene
import torch
import numpy as np
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
import wandb
from plot_utils import plot_frames
from metrics import run_metric_suite_editing, run_metric_suite_controlled


device = "cuda"
dtype = torch.float16


def main(video, teacher_name, prompt, editing_prompt, edited_prompt, timestep, value_dict, wb):
    with torch.no_grad():
        teacher = get_teacher(teacher_name, prompt=prompt, editing_prompt=editing_prompt, value_dict={}, device=device, gt_images=video, do_compile=False)

        z_t = torch.randn((len(video), teacher.latent_shape[0], teacher.latent_shape[1], teacher.latent_shape[2]), device=device, dtype=dtype)
        measurments = teacher.forward_operator(video) # measurements with original (seva) resolution

        # sample individually with teacher
        _, z_0 = teacher.multi_step_sample(z_t, 7.5, 999, 50, 0, len(video))

        # plot outputs
        x0_np =  teacher.decode(z_0, type="np", do_postprocess=True)
        plot_frames(x0_np, wb, "individial_preds", "")

        # prepare to SEVA encoding
        x0_pt =  teacher.decode(z_0, type="pt", do_postprocess=False).to(device=device, dtype=dtype) # [-1.0, 1.0]
        x0_pt = F.interpolate(x0_pt, size=(576, 576), mode='bilinear', align_corners=False) # [-1.0, 1.0]

        # concat 1 GT view, pad to 21 frames
        all_gt = value_dict["cond_frames"]  # [-1.0, 1.0]
        num_inputs = 1
        input_view_torch_gt = all_gt[:num_inputs].to(device=device, dtype=dtype)
        padding = torch.cat([x0_pt[-1:]] * (20 - x0_pt.shape[0]))
        x0_pt = torch.cat([input_view_torch_gt, x0_pt, padding])

        #teacher = None # free memory

        seva = SevaPipeline(device=device, value_dict=value_dict, do_compile=False)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            z0_seva = seva.ae.encode(x0_pt)

        # forward diffusion to desired starting point
        timestep = timestep * torch.ones(21, dtype=dtype, device=device)
        eps = torch.randn_like(z0_seva)
        latent = seva.noise_to_timestep(z0_seva, timestep, eps)

        # seva sampling
        z_0 = seva.euler_edm_sample(latent, timestep, 50, 2.0)

        # plot outputs
        output_frames = seva.decode(z_0, type="np")[num_inputs:-len(padding)]
        plot_frames(output_frames, wb, "SDEdit_output", "", save_as_pdf=True)
        plot_frames(measurments, wb, "measurements", "", save_as_pdf=True)

        seva = None  # free memory
        torch.cuda.empty_cache()

        # Eval
        if teacher_name == "pix2pix":
            run_metric_suite_editing(np.asarray(video), np.asarray(output_frames), prompt, edited_prompt, wb)
        else:
            run_metric_suite_controlled(np.asarray(output_frames), np.asarray(measurments), teacher.forward_operator, wb)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SDEdit baseline")
    parser.add_argument("--teacher_name", type=str, required=True, help="Name of the teacher model (e.g., pix2pix, controlnet)")
    parser.add_argument("--scene_path", type=str, required=True, help="Path to the scene folder")
    parser.add_argument("--prompt", type=str, required=True, help="Original prompt describing the scene")
    parser.add_argument("--timestep", type=int, required=True, help="Timestep to noise individial predictions before sampling with Seva")
    parser.add_argument("--editing_prompt", type=str, default="", help="Instructional editing prompt (for editing teachers)")
    parser.add_argument("--edited_prompt", type=str, default="", help="Final appearance prompt after editing")

    args = parser.parse_args()

    # Load scene
    num_inputs = 1
    value_dict = get_value_dict_of_scene(args.scene_path, num_inputs)
    all_gt = value_dict["cond_frames"]  # range [-1.0, 1.0]
    torch_gt = (((all_gt[num_inputs:value_dict["num_imgs_no_padding"]] + 1) / 2.0) * 255).clamp(0, 255).to(torch.uint8)
    pil_gt = [to_pil_image(torch_gt[i]) for i in range(torch_gt.shape[0])]

    #video = [frame.resize((512, 512)) for frame in pil_gt]
    video = pil_gt

    # Init wandb
    wb = wandb.init(
        project="3dips",
        name="sdedit baseline",
        config={
            "teacher_name": args.teacher_name,
            "scene_path": args.scene_path,
            "prompt": args.prompt,
            "timestep": args.timestep,
            "editing_prompt": args.editing_prompt,
            "edited_prompt": args.edited_prompt
        }
    )

    del all_gt, torch_gt, pil_gt

    main(video, args.teacher_name, args.prompt, args.editing_prompt, args.edited_prompt, args.timestep, value_dict, wb)