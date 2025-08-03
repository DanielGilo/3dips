
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] =":4096:8" # necessary when setting torch.use_dererministic_algorithms(True), increases memory by 24MB


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


def main(video, teacher_name, prompt, editing_prompt, edited_prompt, wb):
    with torch.no_grad():
        teacher = get_teacher(teacher_name, prompt=prompt, editing_prompt=editing_prompt, value_dict={}, device=device, gt_images=video, do_compile=False)
        pipe = teacher.pipeline
        pipe = pipe.to(device="cuda") 

        measurements = teacher.forward_operator(video)

        if teacher_name == "pix2pix":
            result = pipe(prompt=[editing_prompt] * len(measurements), image=measurements).images
        else:
            result = pipe(prompt=[prompt] * len(measurements), image=measurements).images

        # Plot outputs
        plot_frames(result, wb, "result", "", save_as_pdf=True)
        plot_frames(measurements, wb, "measurements", "", save_as_pdf=True)

        torch.cuda.empty_cache()

        # Eval
        if teacher_name == "pix2pix":
            run_metric_suite_editing(np.asarray(video), np.asarray(result), prompt, edited_prompt, wb)
        else:
            run_metric_suite_controlled(np.asarray(result), np.asarray(measurements), teacher.forward_operator, wb)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run individual predictions baseline")
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

    #video = [frame.resize((512, 512)) for frame in pil_gt]
    video = pil_gt

    # Init wandb
    wb = wandb.init(
        project="3dips",
        name="individual preds baseline",
        config={
            "teacher_name": args.teacher_name,
            "scene_path": args.scene_path,
            "prompt": args.prompt,
            "editing_prompt": args.editing_prompt,
            "edited_prompt": args.edited_prompt
        }
    )

    del all_gt, torch_gt, pil_gt

    main(video, args.teacher_name, args.prompt, args.editing_prompt, args.edited_prompt, wb)