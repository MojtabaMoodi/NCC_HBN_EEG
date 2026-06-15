#!/bin/bash
#SBATCH --account=aip-aghodsib
#SBATCH --partition=gpubase_l40s_b4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=labram_frozen_resume
#SBATCH --output=/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/final_logs_frozen/slurm/%x-%j.out
#SBATCH --error=/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/final_logs_frozen/slurm/%x-%j.err

set -euo pipefail

# Required via sbatch --export (see submit_labram_frozen_resume_*.sh):
#   TASK=age|gender|regression
#   SEGMENT=1s|2s|4s
#   START_EPOCH=<int>  next epoch index (0-based); see output/log.txt last "epoch" + 1
# Optional:
#   RESUME_CKPT=<path>  default: ${RUN_DIR}/output/checkpoint-best.pth
: "${TASK:?TASK must be set to age, gender, or regression}"
: "${SEGMENT:?SEGMENT must be set to 1s, 2s, or 4s}"
: "${START_EPOCH:?START_EPOCH must be set (next epoch to run, 0-based)}"

case "${TASK}" in
  age|gender|regression) ;;
  *)
    echo "ERROR: TASK must be age, gender, or regression. Got: ${TASK}" >&2
    exit 1
    ;;
esac

case "${SEGMENT}" in
  1s|2s|4s) ;;
  *)
    echo "ERROR: SEGMENT must be 1s, 2s, or 4s. Got: ${SEGMENT}" >&2
    exit 1
    ;;
esac

if ! [[ "${START_EPOCH}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: START_EPOCH must be a non-negative integer. Got: ${START_EPOCH}" >&2
  exit 1
fi

EEG_ROOT="${EEG_ROOT:-/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG}"
DATA_PATH="${DATA_PATH:-/home/mojtabam/scratch/processed_eeg_data_hdf5}"

if [[ "${TASK}" == "regression" ]]; then
  RUN_LABEL="labram_age_regression_frozen"
  JOB_SUFFIX="age_regression"
else
  RUN_LABEL="labram_${TASK}_frozen"
  JOB_SUFFIX="${TASK}"
fi

RUN_DIR="${EEG_ROOT}/final_logs_frozen/LaBraM/${RUN_LABEL}_${SEGMENT}"
RESUME_CKPT="${RESUME_CKPT:-${RUN_DIR}/output/checkpoint-best.pth}"
TRAIN_LOG="${RUN_DIR}/train_resume_slurm-${SLURM_JOB_ID:-local}.log"

mkdir -p "${RUN_DIR}/output" "${RUN_DIR}/logs" "${EEG_ROOT}/final_logs_frozen/slurm"

if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "ERROR: Resume checkpoint not found: ${RESUME_CKPT}" >&2
  exit 1
fi

echo "=========================================="
echo "SLURM Resume Job"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "TASK: ${TASK}  SEGMENT: ${SEGMENT}"
echo "RUN_DIR: ${RUN_DIR}"
echo "RESUME_CKPT: ${RESUME_CKPT}"
echo "START_EPOCH: ${START_EPOCH} (runs epochs ${START_EPOCH}..49)"
echo "Train log: ${TRAIN_LOG} (original train.log is never overwritten)"
echo "Start: $(date)"
echo "=========================================="

if command -v nvidia-smi &>/dev/null; then
  nvidia-smi
fi

cd "${EEG_ROOT}/LaBraM"
chmod +x runs/run_labram_frozen.sh

export DATA_PATH
export TASK
export SEGMENT

{
  echo ""
  echo "========== RESUME $(date) job=${SLURM_JOB_ID:-local} start_epoch=${START_EPOCH} =========="
  DATA_PATH="${DATA_PATH}" TASK="${TASK}" SEGMENT="${SEGMENT}" \
    bash runs/run_labram_frozen.sh \
      --output_dir "${RUN_DIR}/output" \
      --log_dir "${RUN_DIR}/logs" \
      --resume "${RESUME_CKPT}" \
      --start_epoch "${START_EPOCH}"
  echo "========== RESUME finished $(date) =========="
} 2>&1 | tee "${TRAIN_LOG}"

echo "Completed: $(date)"
