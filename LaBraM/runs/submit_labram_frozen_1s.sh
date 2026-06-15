#!/bin/bash
# Submit frozen LaBraM jobs for 1s segments only (age, gender, regression).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_jobs/run_labram_frozen.slurm"
EEG_ROOT="${EEG_ROOT:-/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG}"
DATA_PATH="${DATA_PATH:-/home/mojtabam/scratch/processed_eeg_data_hdf5}"

for TASK in age gender regression; do
  if [[ "${TASK}" == "regression" ]]; then
    JOB_NAME="labram_age_regression_frozen_1s"
    RUN_DIR="${EEG_ROOT}/final_logs_frozen/LaBraM/labram_age_regression_frozen_1s"
  else
    JOB_NAME="labram_${TASK}_frozen_1s"
    RUN_DIR="${EEG_ROOT}/final_logs_frozen/LaBraM/labram_${TASK}_frozen_1s"
  fi
  mkdir -p "${RUN_DIR}/output" "${RUN_DIR}/logs"
  sbatch \
    --account="${SLURM_ACCOUNT:-aip-aghodsib}" \
    --partition="${SLURM_PARTITION:-gpubase_l40s_b4}" \
    --time="${SLURM_TIME:-3-00:00:00}" \
    --mem=64G \
    --gres="${SLURM_GRES:-gpu:l40s:1}" \
    --job-name="${JOB_NAME}" \
    --output="${RUN_DIR}/slurm-%j.out" \
    --error="${RUN_DIR}/slurm-%j.err" \
    --export=ALL,TASK="${TASK}",SEGMENT=1s,EEG_ROOT="${EEG_ROOT}",DATA_PATH="${DATA_PATH}" \
    "${SLURM_SCRIPT}"
done

echo "Submitted 3 frozen 1s jobs."
