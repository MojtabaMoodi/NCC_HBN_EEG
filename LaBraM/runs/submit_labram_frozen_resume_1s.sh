#!/bin/bash
# Resume incomplete frozen LaBraM 1s jobs (age, gender, age regression).
#
# checkpoint-best.pth stores weights only (no optimizer); --start_epoch skips
# finished epochs. START_EPOCH is read from the last line of output/log.txt.
#
# Usage:
#   bash LaBraM/runs/submit_labram_frozen_resume_1s.sh
#   bash LaBraM/runs/submit_labram_frozen_resume_1s.sh --dry-run
#
# Override a single run:
#   TASK=age START_EPOCH=26 bash -c '...'  # or edit and sbatch manually

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_jobs/run_labram_frozen_resume.slurm"
EEG_ROOT="${EEG_ROOT:-/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG}"
DATA_PATH="${DATA_PATH:-/home/mojtabam/scratch/processed_eeg_data_hdf5}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-aip-aghodsib}"
SLURM_PARTITION="${SLURM_PARTITION:-gpubase_l40s_b4}"
SLURM_TIME="${SLURM_TIME:-3-00:00:00}"
SLURM_GRES="${SLURM_GRES:-gpu:l40s:1}"
SLURM_MEM="${SLURM_MEM:-64G}"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

run_dir_for() {
  local task="$1"
  if [[ "${task}" == "regression" ]]; then
    echo "${EEG_ROOT}/final_logs_frozen/LaBraM/labram_age_regression_frozen_1s"
  else
    echo "${EEG_ROOT}/final_logs_frozen/LaBraM/labram_${task}_frozen_1s"
  fi
}

job_name_for() {
  local task="$1"
  if [[ "${task}" == "regression" ]]; then
    echo "labram_age_regression_frozen_1s_resume"
  else
    echo "labram_${task}_frozen_1s_resume"
  fi
}

next_start_epoch() {
  local log_file="$1"
  if [[ ! -f "${log_file}" ]]; then
    echo "ERROR: Missing metrics log: ${log_file}" >&2
    return 1
  fi
  python3 - "${log_file}" <<'PY'
import json
import sys

path = sys.argv[1]
last_epoch = None
with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        last_epoch = json.loads(line)["epoch"]
if last_epoch is None:
    raise SystemExit(f"No epochs found in {path}")
print(last_epoch + 1)
PY
}

echo "Resume frozen LaBraM 1s jobs"
echo "  SLURM script: ${SLURM_SCRIPT}"
echo ""

for TASK in age gender regression; do
  RUN_DIR="$(run_dir_for "${TASK}")"
  JOB_NAME="$(job_name_for "${TASK}")"
  LOG_FILE="${RUN_DIR}/output/log.txt"
  CKPT="${RUN_DIR}/output/checkpoint-best.pth"

  if [[ ! -f "${CKPT}" ]]; then
    echo "SKIP ${JOB_NAME}: no checkpoint at ${CKPT}" >&2
    continue
  fi

  START_EPOCH="$(next_start_epoch "${LOG_FILE}")"
  if [[ "${START_EPOCH}" -ge 50 ]]; then
    echo "SKIP ${JOB_NAME}: already at epoch ${START_EPOCH} (>= 50)" >&2
    continue
  fi

  mkdir -p "${RUN_DIR}/output" "${RUN_DIR}/logs"

  SBATCH_ARGS=(
    --account="${SLURM_ACCOUNT}"
    --partition="${SLURM_PARTITION}"
    --time="${SLURM_TIME}"
    --mem="${SLURM_MEM}"
    --gres="${SLURM_GRES}"
    --job-name="${JOB_NAME}"
    --output="${RUN_DIR}/slurm-resume-%j.out"
    --error="${RUN_DIR}/slurm-resume-%j.err"
    --export=ALL,TASK="${TASK}",SEGMENT=1s,START_EPOCH="${START_EPOCH}",EEG_ROOT="${EEG_ROOT}",DATA_PATH="${DATA_PATH}",RESUME_CKPT="${CKPT}"
  )

  echo "${JOB_NAME}: resume from epoch ${START_EPOCH} (${CKPT})"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "  [dry-run] sbatch ${SBATCH_ARGS[*]} ${SLURM_SCRIPT}"
  else
    JOB_ID="$(sbatch "${SBATCH_ARGS[@]}" "${SLURM_SCRIPT}" | awk '{print $4}')"
    echo "  -> submitted job ${JOB_ID}"
  fi
done

echo ""
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry run complete."
else
  echo "Monitor: squeue -u \$USER"
  echo "Resume training logs: final_logs_frozen/LaBraM/*/train_resume_slurm-<jobid>.log"
  echo "Epoch status: bash LaBraM/runs/check_frozen_1s_progress.sh"
fi
