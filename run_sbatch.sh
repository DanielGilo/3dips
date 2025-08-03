#!/bin/bash
#SBATCH -c 10 # number of cores
#SBATCH --job-name=3dips
##SBATCH --gres=gpu:L40:1
#SBATCH --gres=gpu:A100:1
##SBATCH --gres=gpu:1
##SBATCH --nodelist=newton5,nlp-h200-1,chuck1,chuck2,entropy1,entropy2
#SBATCH -o runs_output/slurm.%N.%j.out    # stdout goes here
#SBATCH -e runs_output/slurm.%N.%jerr.out   # stderr goes here
#SBATCH --requeue
##SBATCH --nodelist=bruno3
##SBATCH --partition=espresso
##SBATCH --account=espresso

#EXAMPLE OF  PYTHON JOB
#/home/danielgilo/miniconda3/envs/3dips/bin/python 2d_toy_exp.py --teacher_name "inpainting" --student_name "sd" --lr 1.0e-2 --num_iterations 10000

/home/danielgilo/miniconda3/envs/3dips/bin/python multiview_conditional_distillation.py \
                                                --scene_path "/home/danielgilo/3dips/in2n-datasets/face" \
                                                --num_views 9 \
                                                --prompt "A man with curly hair in a grey jacket" \
                                                --editing_prompt  "Give him a Venetian mask" \
                                                --edited_prompt  "A man with curly hair in a grey jacket with a Venetian mask" \
                                                --teacher_name "pix2pix" \
                                                --student_name "seva" \
                                                --loss "sds_shared" \
                                                --lr 1e-4 \
                                                --final_lr 5e-5 \
                                                --n_warmup_steps 800 \
                                                --n_distill_per_timestep 50 \
                                                --n_distill_initial_timestep 800 \
                                                --distill_dt 25 \
                                                --sample_dt 3 \
                                                --t_min 0 \
                                                --teacher_timestep_shape_factor 0.5 \
                                                --teacher_cfg 7.5 \
                                                --image_cfg 1.5 \
                                                --n_teacher_iters 1 \
                                                --cfg_w 1.0 \
                                                --sampling_guidance_scale 2.0


## testing

# /home/danielgilo/miniconda3/envs/3dips/bin/python multiview_conditional_distillation.py \
#                                                 --scene_path "/home/danielgilo/3dips/seva/assets_demo_cli/dl3d140-165f5af8bfe32f70595a1c9393a6e442acf7af019998275144f605b89a306557" \
#                                                 --prompt "a high-quality, detailed, and professional image of a building" \
#                                                 --editing_prompt "make it look like it just rained" \
#                                                 --edited_prompt "a high-quality, detailed, and professional image of a vase with roses, on a round table in a garden, just after it rained" \
#                                                 --teacher_name "canny" \
#                                                 --student_name "seva" \
#                                                 --loss "sds_shared" \
#                                                 --lr 0.0 \
#                                                 --final_lr 0.0 \
#                                                 --n_warmup_steps 0 \
#                                                 --n_distill_per_timestep 1 \
#                                                 --n_distill_initial_timestep 1 \
#                                                 --distill_dt 40 \
#                                                 --sample_dt 50 \
#                                                 --n_teacher_iters 1 \
#                                                 --n_inv_iters 0 \
#                                                 --t_min 800 \
#                                                 --teacher_timestep_shape_factor 0.5 \
#                                                 --cfg_w 4.0 \
#                                                 --teacher_cfg 5.0 \
#                                                 --distillation_guidance_scale 1.0
