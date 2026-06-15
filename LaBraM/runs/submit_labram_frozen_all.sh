#!/bin/bash
# Submit all 9 frozen LaBraM experiments (age, gender, regression × 1s, 2s, 4s).
#
# Usage (from repo root or LaBraM/runs):
#   bash LaBraM/runs/submit_labram_frozen_all.sh
#   bash LaBraM/runs/submit_labram_frozen_all.sh --dry-run
#
# Optional env overrides:
#   EEG_ROOT, DATA_PATH, SLURM_PARTITION, SLURM_ACCOUNT, SLURM_GRES
#   SLURM_TIME — if set, applies to all jobs (default: 3d for 1s, 2d for 2s/4s)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_jobs/run_labram_frozen.slurm"

EEG_ROOT="${EEG_ROOT:-/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG}"
DATA_PATH="${DATA_PATH:-/home/mojtabam/scratch/processed_eeg_data_hdf5}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-aip-aghodsib}"
SLURM_PARTITION="${SLURM_PARTITION:-gpubase_l40s_b4}"
SLURM_GRES="${SLURM_GRES:-gpu:l40s:1}"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

mkdir -p "${EEG_ROOT}/final_logs_frozen/slurm"

run_dir_for() {
  local task="$1"
  local seg="$2"
  if [[ "${task}" == "regression" ]]; then
    echo "${EEG_ROOT}/final_logs_frozen/LaBraM/labram_age_regression_frozen_${seg}"
  else
    echo "${EEG_ROOT}/final_logs_frozen/LaBraM/labram_${task}_frozen_${seg}"
  fi
}

job_name_for() {
  local task="$1"
  local seg="$2"
  if [[ "${task}" == "regression" ]]; then
    echo "labram_age_regression_frozen_${seg}"
  else
    echo "labram_${task}_frozen_${seg}"
  fi
}

mem_for_segment() {
  case "$1" in
    1s) echo "64G" ;;
    2s) echo "48G" ;;
    4s) echo "32G" ;;
    *) echo "32G" ;;
  esac
}

time_for_segment() {
  if [[ -n "${SLURM_TIME:-}" ]]; then
    echo "${SLURM_TIME}"
    return
  fi
  case "$1" in
    1s) echo "3-00:00:00" ;;
    2s|4s) echo "2-00:00:00" ;;
    *) echo "2-00:00:00" ;;
  esac
}

echo "Submitting frozen LaBraM jobs"
echo "  SLURM script: ${SLURM_SCRIPT}"
echo "  EEG_ROOT:     ${EEG_ROOT}"
echo "  DATA_PATH:    ${DATA_PATH}"
echo "  Partition:    ${SLURM_PARTITION}"
echo ""

SUBMITTED=()
for TASK in age gender regression; do
  for SEG in 1s 2s 4s; do
    RUN_DIR="$(run_dir_for "${TASK}" "${SEG}")"
    JOB_NAME="$(job_name_for "${TASK}" "${SEG}")"
    MEM="$(mem_for_segment "${SEG}")"
    TIME_LIMIT="$(time_for_segment "${SEG}")"
    mkdir -p "${RUN_DIR}/output" "${RUN_DIR}/logs"

    SBATCH_ARGS=(
      --account="${SLURM_ACCOUNT}"
      --partition="${SLURM_PARTITION}"
      --time="${TIME_LIMIT}"
      --mem="${MEM}"
      --gres="${SLURM_GRES}"
      --job-name="${JOB_NAME}"
      --output="${RUN_DIR}/slurm-%j.out"
      --error="${RUN_DIR}/slurm-%j.err"
      --export=ALL,TASK="${TASK}",SEGMENT="${SEG}",EEG_ROOT="${EEG_ROOT}",DATA_PATH="${DATA_PATH}"
    )

    if [[ "${DRY_RUN}" -eq 1 ]]; then
      echo "[dry-run] sbatch ${SBATCH_ARGS[*]} ${SLURM_SCRIPT}"
    else
      JOB_ID="$(sbatch "${SBATCH_ARGS[@]}" "${SLURM_SCRIPT}" | awk '{print $4}')"
      echo "Submitted ${JOB_NAME} -> job ${JOB_ID}"
      SUBMITTED+=("${JOB_ID}")
    fi
  done
done

echo ""
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry run complete (9 jobs would be submitted)."
else
  echo "Submitted ${#SUBMITTED[@]} jobs."
  echo "Monitor: squeue -u \$USER"
  echo "Per-run logs: final_logs_frozen/LaBraM/<run_name>/train.log"
  echo "Slurm stdout: final_logs_frozen/LaBraM/<run_name>/slurm-<jobid>.out"
fi
