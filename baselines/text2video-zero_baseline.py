import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] =":4096:8" # necessary when setting torch.use_dererministic_algorithms(True), increases memory by 24MB

from PIL import Image
import wandb
import argparse
import imageio
import torch
import numpy as np
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor
from torchvision.transforms.functional import to_pil_image
from seva_utils import get_value_dict_of_scene
from teachers import get_teacher
from metrics import run_metric_suite_controlled, run_metric_suite_editing
from plot_utils import plot_frames


device = "cuda"
dtype = torch.float16

def main(video, teacher_name, prompt, editing_prompt, edited_prompt, wb):
    # Editing
    if teacher_name == "pix2pix":
        with torch.no_grad():
            model_id = "timbrooks/instruct-pix2pix"
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
            pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=3))

            result = pipe(prompt=[editing_prompt] * len(video), image=video).images

            # Wandb log
            wb.log({"original_video": wandb.Video(np.asarray(video).transpose((0, 3, 1, 2)), format="mp4", fps=3, caption="original video"),
                    "edited_video": wandb.Video(np.asarray(result).transpose((0, 3, 1, 2)), format="mp4", fps=3, caption="edited video")})
            plot_frames(result, wb, "edited frames", "", save_as_pdf=True)

            # Eval
            run_metric_suite_editing(np.asarray(video), np.asarray(result), prompt, edited_prompt, wb)

    # ControlNet
    else:
        with torch.no_grad():
            teacher = get_teacher(teacher_name, prompt=prompt, editing_prompt="", value_dict={}, device=device, gt_images=video, do_compile=False)

            measurements = teacher.forward_operator(video)

            pipe = teacher.pipeline
            pipe = pipe.to(device="cuda")
            pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
            pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2)) 

            # fix latents for all frames
            latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16).repeat(len(measurements), 1, 1, 1)

            result = pipe(prompt=[prompt] * len(measurements), image=measurements, latents=latents).images

            # Wandb log
            wb.log({"measurements_video": wandb.Video(np.asarray(measurements).transpose((0, 3, 1, 2)), format="mp4", fps=3, caption="measurements video"),
                    "result_video": wandb.Video(np.asarray(result).transpose((0, 3, 1, 2)), format="mp4", fps=3, caption="result video")})
            plot_frames(result, wb, "result frames", "", save_as_pdf=True)
            plot_frames(np.asarray(measurements), wb, "measurements", "", save_as_pdf=True)

            # Eval
            run_metric_suite_controlled(np.asarray(result), np.asarray(measurements), teacher.forward_operator, wb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text2video-zero baseline")
    parser.add_argument("--teacher_name", type=str, required=True, help="Name of the teacher model (e.g., pix2pix, controlnet)")
    parser.add_argument("--scene_path", type=str, required=True, help="Path to the scene folder")
    parser.add_argument("--prompt", type=str, required=True, help="Original prompt describing the scene")
    parser.add_argument("--editing_prompt", type=str, default="", help="Instructional editing prompt (for editing teachers)")
    parser.add_argument("--edited_prompt", type=str, default="", help="Final appearance prompt after editing")

    args = parser.parse_args()

    # Load scene
    num_inputs = 1
    value_dict = get_value_dict_of_scene(args.scene_path, num_inputs)
    all_gt = value_dict["cond_frames"]  # range [-1.0, 1.0]
    torch_gt = (((all_gt[num_inputs:value_dict["num_imgs_no_padding"]] + 1) / 2.0) * 255).clamp(0, 255).to(torch.uint8)
    pil_gt = [to_pil_image(torch_gt[i]) for i in range(torch_gt.shape[0])]

    video = [frame.resize((512, 512)) for frame in pil_gt]

    # Init wandb
    wb = wandb.init(
        project="3dips",
        name="text2video-zero baseline",
        config={
            "teacher_name": args.teacher_name,
            "scene_path": args.scene_path,
            "prompt": args.prompt,
            "editing_prompt": args.editing_prompt,
            "edited_prompt": args.edited_prompt
        }
    )

    main(video, args.teacher_name, args.prompt, args.editing_prompt, args.edited_prompt, wb)