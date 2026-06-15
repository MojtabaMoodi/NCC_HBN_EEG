#!/bin/bash
#SBATCH --account=aip-aghodsib
#SBATCH --partition=gpubase_l40s_b4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=labram_frozen
#SBATCH --output=/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/final_logs_frozen/slurm/%x-%j.out
#SBATCH --error=/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/final_logs_frozen/slurm/%x-%j.err

set -euo pipefail

# Required via sbatch --export (see submit_labram_frozen_*.sh):
#   TASK=age|gender|regression
#   SEGMENT=1s|2s|4s
: "${TASK:?TASK must be set to age, gender, or regression}"
: "${SEGMENT:?SEGMENT must be set to 1s, 2s, or 4s}"

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
TRAIN_LOG="${RUN_DIR}/train_slurm-${SLURM_JOB_ID:-local}.log"
mkdir -p "${RUN_DIR}/output" "${RUN_DIR}/logs" "${EEG_ROOT}/final_logs_frozen/slurm"

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Job Name: ${SLURM_JOB_NAME:-labram_${JOB_SUFFIX}_frozen_${SEGMENT}}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "TASK: ${TASK}"
echo "SEGMENT: ${SEGMENT}"
echo "RUN_DIR: ${RUN_DIR}"
echo "DATA_PATH: ${DATA_PATH}"
echo "Train log: ${TRAIN_LOG}"
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

DATA_PATH="${DATA_PATH}" TASK="${TASK}" SEGMENT="${SEGMENT}" \
  bash runs/run_labram_frozen.sh \
    --output_dir "${RUN_DIR}/output" \
    --log_dir "${RUN_DIR}/logs" \
  2>&1 | tee "${TRAIN_LOG}"

echo "Completed: $(date)"
