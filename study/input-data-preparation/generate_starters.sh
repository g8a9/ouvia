#!/bin/bash
#SBATCH --job-name=gen_starters
#SBATCH --output=./logs/%A.out
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=4
#SBATCH --qos=gpu-debug
#SBATCH --partition=a6000

module load cuda
source /mnt/home/giuseppe/mydata/venvs/gnt/bin/activate

# model="Qwen/Qwen3-235B-A22B"
model="Qwen/Qwen3-32B"

python src/generate_starters.py \
    --scenario "booking_a_taxi" \
    --model_name_or_path $model \
    --total_starters 50 \
    --enable_thinking