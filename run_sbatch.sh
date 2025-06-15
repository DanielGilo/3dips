#!/bin/bash
#SBATCH -c 10 # number of cores
#SBATCH --job-name=3dips
#SBATCH --gres=gpu:L40:1
###SBATCH --gres=gpu:A100:1
#SBATCH -o runs_output/slurm.%N.%j.out    # stdout goes here
#SBATCH -e runs_output/slurm.%N.%jerr.out   # stderr goes here
#SBATCH --requeue
##SBATCH --nodelist=bruno3
##SBATCH --partition=espresso
##SBATCH --account=espresso

#EXAMPLE OF  PYTHON JOB
#/home/danielgilo/miniconda3/envs/3dips/bin/python 2d_toy_exp.py --teacher_name "inpainting" --student_name "sd" --lr 1.0e-2 --num_iterations 10000
/home/danielgilo/miniconda3/envs/3dips/bin/python multiview_conditional_distillation.py --teacher_name "depth" --student_name "seva_frozen_mv_transformer" --loss "sds_shared" --lr 1e-4 --final_lr 5e-5 --n_warmup_steps 400 --n_distill_per_timestep 50 --n_timesteps 40 --conv_thresh 1.0e-2 --eta 1.0 --n_inv_iters 10 --t_min 20 --cfg_w 2.0  #--do_compile

