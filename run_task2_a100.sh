#!/bin/bash
#SBATCH --job-name=task2_effnet
#SBATCH --output=logs/task2_a100_%j.out
#SBATCH --error=logs/task2_a100_%j.err
#SBATCH --partition=unlimited
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source /common/home/psc84/anaconda3/etc/profile.d/conda.sh
conda activate PacMan

cd /common/users/psc84/GSOC_tests/03_eval_repo
mkdir -p logs

echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python task2_lens_finding/train.py \
  --data-dir     /common/users/psc84/GSOC_tests/03_eval_repo/task2_data \
  --epochs       30 \
  --batch-size   32 \
  --model        efficientnet \
  --lr-backbone  5e-5 \
  --lr-head      3e-4 \
  --focal-alpha  0.25 \
  --focal-gamma  2.5 \
  --num-workers  4 \
  --tta \
  --save-dir     /common/users/psc84/GSOC_tests/03_eval_repo/checkpoints/task2
