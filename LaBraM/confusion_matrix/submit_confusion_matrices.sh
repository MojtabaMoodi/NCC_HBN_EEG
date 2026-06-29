#!/usr/bin/env bash
# Submit SLURM job to compute all LaBraM age/gender confusion matrices (1s, 2s, 4s).
#
# Usage (from anywhere):
#   bash LaBraM/confusion_matrix/submit_confusion_matrices.sh
#
# Optional env overrides:
#   EEG_ROOT, DATA_PATH, DEVICE
#   SLURM_ACCOUNT, SLURM_PARTITION, SLURM_TIME, SLURM_GRES, SLURM_MEM

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EEG_ROOT="${EEG_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SLURM_SCRIPT="${EEG_ROOT}/LaBraM/runs/slurm_jobs/run_confusion_matrices.slurm"
LOG_DIR="${EEG_ROOT}/LaBraM/confusion_matrix/logs"
DATA_PATH="${DATA_PATH:-/home/mojtabam/scratch/processed_eeg_data_hdf5}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${SLURM_SCRIPT}" ]]; then
  echo "Error: SLURM script not found: ${SLURM_SCRIPT}" >&2
  exit 1
fi

echo "Submitting LaBraM confusion matrix job"
echo "  EEG_ROOT: ${EEG_ROOT}"
echo "  DATA_PATH: ${DATA_PATH}"
echo "  DEVICE: ${DEVICE}"
echo "  SLURM script: ${SLURM_SCRIPT}"
echo "  Logs: ${LOG_DIR}/slurm-<jobid>.{out,err}"

JOB_ID="$(
  sbatch \
    --account="${SLURM_ACCOUNT:-aip-aghodsib}" \
    --partition="${SLURM_PARTITION:-gpubase_l40s_b4}" \
    --time="${SLURM_TIME:-6:00:00}" \
    --mem="${SLURM_MEM:-32G}" \
    --gres="${SLURM_GRES:-gpu:l40s:1}" \
    --job-name="labram_confusion_matrices" \
    --output="${LOG_DIR}/slurm-%j.out" \
    --error="${LOG_DIR}/slurm-%j.err" \
    --export=ALL,EEG_ROOT="${EEG_ROOT}",DATA_PATH="${DATA_PATH}",DEVICE="${DEVICE}" \
    "${SLURM_SCRIPT}" \
  | awk '{print $4}'
)"

echo "Submitted job ${JOB_ID}"
echo "  stdout: ${LOG_DIR}/slurm-${JOB_ID}.out"
echo "  stderr: ${LOG_DIR}/slurm-${JOB_ID}.err"
echo "  run log: ${LOG_DIR}/run_all_confusion_matrices_slurm-${JOB_ID}.log"
