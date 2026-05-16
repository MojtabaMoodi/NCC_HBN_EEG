#!/bin/bash
#SBATCH --account=aip-aghodsib
#SBATCH --partition=gpubase_l40s_b4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=labram_frozen
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Required env vars from sbatch --export:
#   SEGMENT_LENGTH: 1s | 2s | 4s
#   FOLD_IDX:       0..4
: "${SEGMENT_LENGTH:?SEGMENT_LENGTH must be set to 1s/2s/4s}"
: "${FOLD_IDX:?FOLD_IDX must be set to 0..4}"

case "${SEGMENT_LENGTH}" in
  1s|2s|4s) ;;
  *)
    echo "ERROR: SEGMENT_LENGTH must be one of: 1s, 2s, 4s. Got: ${SEGMENT_LENGTH}" >&2
    exit 1
    ;;
esac

case "${FOLD_IDX}" in
  0|1|2|3|4) ;;
  *)
    echo "ERROR: FOLD_IDX must be one of: 0,1,2,3,4. Got: ${FOLD_IDX}" >&2
    exit 1
    ;;
esac

if [[ "${SEGMENT_LENGTH}" == "4s" ]]; then
  BATCH_SIZE=384
else
  BATCH_SIZE=768
fi

PROJECT_ROOT="/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG"
HDF5_DIR="${HOME}/scratch/${SEGMENT_LENGTH}/fold_${FOLD_IDX}"
OUT_DIR="final_user_identification_frozen/${SEGMENT_LENGTH}/fold_${FOLD_IDX}"
PRETRAINED="LaBraM/checkpoints/labram-base.pth"

cd "${PROJECT_ROOT}"
mkdir -p logs "${OUT_DIR}"

export PATH="/home/mojtabam/miniconda3/bin:${PATH}"

echo "=========================================="
echo "LaBraM Frozen User-ID Training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Segment length: ${SEGMENT_LENGTH}"
echo "Fold: ${FOLD_IDX}"
echo "HDF5 dir: ${HDF5_DIR}"
echo "Output dir: ${OUT_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Start: $(date)"
echo "=========================================="

conda run -n eeg_env --no-capture-output \
  python user_identification/train_labram_arcface.py \
    --hdf5_dir "${HDF5_DIR}" \
    --output_dir "${OUT_DIR}" \
    --segment_length "${SEGMENT_LENGTH}" \
    --epochs 200 \
    --batch_size "${BATCH_SIZE}" \
    --seed 42 \
    --pretrained_path "${PRETRAINED}" \
    --freeze_backbone

echo "Completed: $(date)"
