#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=lotus_gpu
#SBATCH --partition=lotus_gpu
#SBATCH --job-name=hrovercast
#SBATCH --output=slurm-%j.out
#SBATCH --mem=30G
#SBATCH -t 0-12:00

echo "Running overcast with high resolution"

conda activate overcast

overcast \
  train \
    --job-dir output/ \
    --gpu-per-model .25 \
  jasmin-daily \
    --root  data/MERRA_25kmres_2003.csv \
  appended-treatment-transformer \
