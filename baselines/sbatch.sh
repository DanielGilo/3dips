#!/bin/bash
#SBATCH --job-name=run_baselines
#SBATCH --output=run_baselines_%j.out
#SBATCH --error=run_baselines_%j.err
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=8

/home/danielgilo/miniconda3/envs/3dips/bin/python /home/danielgilo/3dips/baselines/run_baselines.py