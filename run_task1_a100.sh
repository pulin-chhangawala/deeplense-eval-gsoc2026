#!/bin/bash
#SBATCH --job-name=task1_effnet
#SBATCH --output=logs/task1_a100_%j.out
#SBATCH --error=logs/task1_a100_%j.err
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

python task1_classification/train.py \
  --data-dir     /common/users/psc84/GSOC_tests/03_eval_repo/task1_data \
  --epochs       50 \
  --batch-size   64 \
  --model        efficientnet \
  --lr-backbone  5e-5 \
  --lr-head      3e-4 \
  --mixup-alpha  0.2 \
  --smoothing    0.05 \
  --num-workers  4 \
  --tta \
  --save-dir     /common/users/psc84/GSOC_tests/03_eval_repo/checkpoints/task1
