#!/usr/bin/env bash
# Submit SLURM job: bootstrap CIs/p-values for all final_logs models (9 tasks × 5 backends).
#
# Usage (from anywhere):
#   bash eval_subject_bootstrap/submit_bootstrap_evaluation.sh
#
# Optional env overrides:
#   EEG_ROOT, HDF5_BASE_DIR, DEVICE, BATCH_SIZE, BOOTSTRAP_B, PERMUTATION_M, SEED,
#   CONFIDENCE, TASK_TYPE, REGRESSION_BASELINE_STAT
#   SLURM_ACCOUNT, SLURM_PARTITION, SLURM_TIME, SLURM_GRES, SLURM_MEM

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EEG_ROOT="${EEG_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SLURM_SCRIPT="${EEG_ROOT}/eval_subject_bootstrap/slurm_jobs/run_bootstrap_evaluation.slurm"
LOG_DIR="${EEG_ROOT}/eval_subject_bootstrap/logs"

HDF5_BASE_DIR="${HDF5_BASE_DIR:-/home/mojtabam/scratch/processed_eeg_data_hdf5}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BOOTSTRAP_B="${BOOTSTRAP_B:-2000}"
PERMUTATION_M="${PERMUTATION_M:-1000}"
SEED="${SEED:-123}"
CONFIDENCE="${CONFIDENCE:-0.95}"
TASK_TYPE="${TASK_TYPE:-both}"
REGRESSION_BASELINE_STAT="${REGRESSION_BASELINE_STAT:-median}"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${SLURM_SCRIPT}" ]]; then
  echo "Error: SLURM script not found: ${SLURM_SCRIPT}" >&2
  exit 1
fi

echo "Submitting bootstrap evaluation job (all tasks × 1s/2s/4s × cnn/resnet18/34/50/labram)"
echo "  EEG_ROOT: ${EEG_ROOT}"
echo "  HDF5_BASE_DIR: ${HDF5_BASE_DIR}"
echo "  DEVICE: ${DEVICE}"
echo "  BOOTSTRAP_B: ${BOOTSTRAP_B}"
echo "  Logs: ${LOG_DIR}/slurm-<jobid>.{out,err}"

JOB_ID="$(
  sbatch \
    --account="${SLURM_ACCOUNT:-aip-aghodsib}" \
    --partition="${SLURM_PARTITION:-gpubase_l40s_b4}" \
    --time="${SLURM_TIME:-3-00:00:00}" \
    --mem="${SLURM_MEM:-32G}" \
    --gres="${SLURM_GRES:-gpu:l40s:1}" \
    --exclude="${SLURM_EXCLUDE:-kn076}" \
    --job-name="bootstrap_eval" \
    --output="${LOG_DIR}/slurm-%j.out" \
    --error="${LOG_DIR}/slurm-%j.err" \
    --export=ALL,\
EEG_ROOT="${EEG_ROOT}",\
HDF5_BASE_DIR="${HDF5_BASE_DIR}",\
DEVICE="${DEVICE}",\
BATCH_SIZE="${BATCH_SIZE}",\
BOOTSTRAP_B="${BOOTSTRAP_B}",\
PERMUTATION_M="${PERMUTATION_M}",\
SEED="${SEED}",\
CONFIDENCE="${CONFIDENCE}",\
TASK_TYPE="${TASK_TYPE}",\
REGRESSION_BASELINE_STAT="${REGRESSION_BASELINE_STAT}" \
    "${SLURM_SCRIPT}" \
  | awk '{print $4}'
)"

echo "Submitted job ${JOB_ID}"
echo "  stdout: ${LOG_DIR}/slurm-${JOB_ID}.out"
echo "  stderr: ${LOG_DIR}/slurm-${JOB_ID}.err"
echo "  run log: ${LOG_DIR}/run_all_bootstrap_slurm-${JOB_ID}.log"
