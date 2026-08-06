#!/bin/bash
#SBATCH -A aip-aghodsib
#SBATCH -p gpubase_l40s_b4
#SBATCH -N 1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00

set -euo pipefail

: "${FOLD_IDX:?FOLD_IDX must be set (0-4)}"

PROJECT_ROOT="/project/6101794/mojtabam/EEG"
RUN_DIR="${PROJECT_ROOT}/final_user_identification_triplet/1s/fold_${FOLD_IDX}"
HDF5_DIR="${HOME}/scratch/1s/fold_${FOLD_IDX}"
TRAIN_LOG="${RUN_DIR}/train_fold${FOLD_IDX}.log"

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_DIR}"

export PATH="/home/mojtabam/miniconda3/bin:${PATH}"

echo "Job ID: ${SLURM_JOB_ID:-local}" | tee -a "${TRAIN_LOG}"
echo "Start: $(date)" | tee -a "${TRAIN_LOG}"
echo "Fold: ${FOLD_IDX}" | tee -a "${TRAIN_LOG}"
echo "Segment length: 1s" | tee -a "${TRAIN_LOG}"
echo "HDF5: ${HDF5_DIR}" | tee -a "${TRAIN_LOG}"
echo "Output: ${RUN_DIR}" | tee -a "${TRAIN_LOG}"

conda run -n eeg_env --no-capture-output python user_identification/train_labram_arcface.py \
  --hdf5_dir "${HDF5_DIR}" \
  --output_dir "${RUN_DIR}" \
  --segment_length 1s \
  --loss_type triplet \
  --pretrained_path LaBraM/checkpoints/labram-base.pth \
  --epochs 200 \
  --batch_size 512 \
  --triplet_classes_per_batch 32 \
  --triplet_samples_per_class 12 \
  --triplet_extra_negatives 128 \
  --triplet_margin 0.2 \
  --triplet_mining semi_hard \
  --early_stopping_patience 10 \
  --early_stopping_min_delta 1e-4 \
  --lr 5e-4 \
  --layer_decay 0.9 \
  --weight_decay 0.05 \
  --warmup_epochs 10 \
  --clip_grad 0.2 \
  --num_workers 8 \
  --save_ckpt_freq 5 \
  --seed 42 \
  2>&1 | tee -a "${TRAIN_LOG}"

echo "End: $(date)" | tee -a "${TRAIN_LOG}"
