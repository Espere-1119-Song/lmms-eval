#!/bin/bash
#SBATCH --job-name=cap_eval
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --mem=48G
#SBATCH --exclude=pika-h100-ash-ad2-091

echo "Job started at: $(date)"

CFG=./cfg/coco.yaml
python3 -m accelerate.commands.launch --num_processes=8 -m lmms_eval --config $CFG

echo "Job finished at: $(date)"