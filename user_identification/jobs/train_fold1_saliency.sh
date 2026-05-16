#!/bin/bash
#SBATCH -p gpubase_l40s_b4
#SBATCH -N 1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=labram_4s_fold1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/mojtabam/projects/aip-aghodsib/mojtabam/EEG   # or your project root

mkdir -p final_user_identification/4s/fold_1

# Non-interactive bash: avoid ~/.bashrc; use conda without `activate` (works with `set -u`).
export PATH="/home/mojtabam/miniconda3/bin:${PATH}"
conda run -n eeg_env --no-capture-output python saliency_analysis/main.py --checkpoint final_user_identification/4s/fold_1/best_model.pth --target_type user_identification --segment_length 4s --hdf5_dir ~/scratch/fold_1 --output_dir final_user_identification/4s/fold_1/saliency/
