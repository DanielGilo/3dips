import os
import subprocess

scene_1 = {
    "path": "/home/danielgilo/3dips/seva/weka/home-jensen/reve/datasets/reconfusion_export/co3d-viewcrafter/car",
    "prompt": "a high-quality, detailed, and professional image of a light-blue car",
    "editing_prompts": [
        "Make the car red",
        "Make it night",
        "Make it in the style of Van Gogh",
        "Make it snowy",
        "Make it Minecraft style"
    ],
    "edited_prompts": [
        "a high-quality, detailed, and professional image of a red car",
        "a high-quality, detailed, and professional image of a light-blue car in the night",
        "a high-quality, detailed, and professional image of a light-blue car in the style of Van Gogh",
        "a high-quality, detailed, and professional image of a light-blue car in the snow",
        "a high-quality, detailed, and professional image of a light-blue car in Minecraft style"
    ]
}

scene_3 = {
    "path": "/home/danielgilo/3dips/seva/assets_demo_cli/garden_flythrough",
    "prompt": "a high-quality, detailed, and professional image of a vase with a plant, on a round table in a garden",
    "editing_prompts": [
        "Swap the plant with roses",
        "turn the table to rosewood table",
        "make it look like it just rained"
    ],
    "edited_prompts": [
        "a high-quality, detailed, and professional image of a vase with roses, on a round table in a garden",
        "a high-quality, detailed, and professional image of a vase with a plant, on a round rosewood table in a garden",
        "a high-quality, detailed, and professional image of a vase with roses, on a round table in a garden, just after it rained"
    ]
}


### from DGE https://arxiv.org/pdf/2404.18929?#page=26.68
scene_4 = {
    "path": "/home/danielgilo/3dips/in2n-datasets/face",  
    "prompt": "A man with curly hair in a grey jacket",
    "editing_prompts": [
        "Give him a Venetian mask",
        "Give him a checkered jacket",
        "Turn him into spider man with a mask"
    ],
    "edited_prompts": [
        "A man with curly hair in a grey jacket with a Venetian mask",
        "A man with curly hair in a checkered cloth",
        "A spider man with a mask and curly hair"
    ]
}

scene_5 = {
    "path": "/home/danielgilo/3dips/in2n-datasets/bear", 
    "prompt": "A stone bear in a garden",
    "editing_prompts": [
        "Make the bear look like a robot",
        "Make the bear look like a panda",
        "Make the color of the bear look like rainbow"
    ],
    "edited_prompts": [
        "A robotic bear in the garden",
        "A panda in the garden",
        "A bear with rainbow color"
    ]
}

scene_6 = {
    "path": "/home/danielgilo/3dips/in2n-datasets/person-small",
    "prompt": "A man standing next to a wall wearing a blue T-shirt and brown pants",
    "editing_prompts": [
        "Make the man look like a mosaic sculpture",
        "Make the person wear a shirt with a pineapple pattern",
        "Turn him into Iron Man",
        "Turn the man into a robot"
    ],
    "edited_prompts": [
        "A man looks like a mosaic sculpture standing next to a wall",
        "A man wearing a shirt with a pineapple pattern",
        "An Iron Man stands to a wall",
        "A robot stands to a wall"
    ]
}

scenes = [scene_4]
teachers = ["pix2pix"]

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH -c 10
#SBATCH --job-name=3dips_{jobname}
#SBATCH --gres=gpu:H200:1
#SBATCH -o runs_output/slurm.%N.%j.out
#SBATCH -e runs_output/slurm.%N.%jerr.out
#SBATCH --requeue

{python_cmd}
"""

os.makedirs("sbatch_scripts", exist_ok=True)

for scene in scenes:
    for teacher in teachers:
        if teacher == "pix2pix":
            for editing_prompt, edited_prompt in zip(scene["editing_prompts"], scene["edited_prompts"]):
                jobname = f"{teacher}_{os.path.basename(scene['path'])}_{editing_prompt.replace(' ', '_')[:20]}"
                python_cmd = (
                    f"/home/danielgilo/miniconda3/envs/3dips/bin/python multiview_conditional_distillation.py "
                    f'--scene_path "{scene["path"]}" '
                    f'--prompt "{scene["prompt"]}" '
                    f'--editing_prompt "{editing_prompt}" '
                    f'--edited_prompt "{edited_prompt}" '
                    f'--teacher_name "{teacher}" '
                    f'--student_name "seva_frozen_mv_transformer" '
                    f'--loss "sds_shared" '
                    f'--lr 1e-4 '
                    f'--final_lr 5e-5 '
                    f'--n_warmup_steps 800 '
                    f'--n_distill_per_timestep 50 '
                    f'--n_distill_initial_timestep 800 '
                    f'--distill_dt 25 '
                    f'--sample_dt 3 '
                    f'--t_min 0 '
                    f'--teacher_timestep_shape_factor 2.0 '
                    f'--teacher_cfg 12.5 '
                    f'--image_cfg 3.0 '
                    f'--n_teacher_iters 1 '
                    f'--eta_teacher 0.0 '
                    f'--pred_teacher_interval 1 '
                    f'--cfg_w 2.0 '
                    f'--distillation_guidance_scale 1.0 '
                    f'--sampling_guidance_scale 0.0'
                )
                sbatch_script = SBATCH_TEMPLATE.format(jobname=jobname, python_cmd=python_cmd)
                sbatch_filename = f"sbatch_scripts/run_sbatch_{jobname}.sh"
                with open(sbatch_filename, "w") as f:
                    f.write(sbatch_script)
                print(f"Submitting {sbatch_filename}")
                subprocess.run(["sbatch", sbatch_filename])
        else:
            jobname = f"{teacher}_{os.path.basename(scene['path'])}"
            python_cmd = (
                f"/home/danielgilo/miniconda3/envs/3dips/bin/python multiview_conditional_distillation.py "
                f'--scene_path "{scene["path"]}" '
                f'--prompt "{scene["prompt"]}" '
                f'--teacher_name "{teacher}" '
                f'--student_name "seva" '
                f'--loss "sds_shared" '
                f'--lr 1e-4 '
                f'--final_lr 5e-5 '
                f'--n_warmup_steps 800 '
                f'--n_distill_per_timestep 50 '
                f'--n_distill_initial_timestep 800 '
                f'--distill_dt 25 '
                f'--sample_dt 3 '
                f'--t_min 0 '
                f'--teacher_timestep_shape_factor 0.5 '
                f'--teacher_cfg 5.0 '
                f'--n_teacher_iters 1 '
                f'--eta_teacher 0.0 '
                f'--pred_teacher_interval 1 '
                f'--cfg_w 4.0 '
                f'--distillation_guidance_scale 1.0 '
                f'--sampling_guidance_scale 0.0'
            )
            sbatch_script = SBATCH_TEMPLATE.format(jobname=jobname, python_cmd=python_cmd)
            sbatch_filename = f"sbatch_scripts/run_sbatch_{jobname}.sh"
            with open(sbatch_filename, "w") as f:
                f.write(sbatch_script)
            print(f"Submitting {sbatch_filename}")
            subprocess.run(["sbatch", sbatch_filename])
