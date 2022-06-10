#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=lotus_gpu
#SBATCH --partition=lotus_gpu
#SBATCH --job-name=lrovercast
#SBATCH --output=slurm-%j.out
#SBATCH --mem=30G
#SBATCH -t 0-12:00

echo "Running overcast with low resolution"

conda activate overcast

overcast \
  train \
    --job-dir output/ \
    --gpu-per-model .25 \
  jasmin-daily \
    --root  data/four_outputs_liqcf_pacific.csv \
  appended-treatment-transformer \
