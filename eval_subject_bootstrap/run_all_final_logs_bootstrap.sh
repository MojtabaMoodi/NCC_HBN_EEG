#!/usr/bin/env bash
#
# Run subject-level bootstrap evaluation for all final_logs tasks × window sizes.
#
# Required environment variables (no defaults):
#   EEG_ROOT         Path to EEG repository (contains final_logs/)
#   HDF5_BASE_DIR    Processed HDF5 directory (eeg_data_{train,test}_{1s,2s,4s}.h5)
#   DEVICE           cuda or cpu
#   BATCH_SIZE       Inference batch size
#
# Optional:
#   MANIFEST_DIR     Defaults to ${EEG_ROOT}/eval_subject_bootstrap/manifests
#   OUTPUT_ROOT      Defaults to ${EEG_ROOT}/eval_subject_bootstrap/outputs/final_logs
#   BOOTSTRAP_B      Number of bootstrap replicates (required if not set — pass explicitly)
#   PERMUTATION_M    Permutation test replicates for classification (0 to skip)
#   SEED             Random seed
#   CONFIDENCE       CI level (e.g. 0.95)
#   TASK_TYPE        HDF5 task groups: both | active | passive
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

required_vars=(EEG_ROOT HDF5_BASE_DIR DEVICE BATCH_SIZE BOOTSTRAP_B PERMUTATION_M SEED CONFIDENCE TASK_TYPE REGRESSION_BASELINE_STAT)
for var in "${required_vars[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "Error: set ${var} before running this script." >&2
    exit 1
  fi
done

MANIFEST_DIR="${MANIFEST_DIR:-${EEG_ROOT}/eval_subject_bootstrap/manifests}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EEG_ROOT}/eval_subject_bootstrap/outputs/final_logs}"

ALL_BACKENDS="cnn,resnet18,resnet34,resnet50,labram"
TASKS=(age_classification gender_classification age_regression)
WINDOWS=(1s 2s 4s)

mkdir -p "${OUTPUT_ROOT}"

run_one() {
  local task="$1"
  local window="$2"
  local out_dir="${OUTPUT_ROOT}/${task}_${window}"
  mkdir -p "${out_dir}"

  local extra_args=()
  if [[ "${task}" == "age_regression" ]]; then
    extra_args+=(--regression_baseline_stat "${REGRESSION_BASELINE_STAT}")
  fi

  echo "=== bootstrap ${task} ${window} ==="
  python "${SCRIPT_DIR}/run_bootstrap_evaluation.py" \
    --task "${task}" \
    --segment_length "${window}" \
    --backends "${ALL_BACKENDS}" \
    --hdf5_base_dir "${HDF5_BASE_DIR}" \
    --eeg_root "${EEG_ROOT}" \
    --manifest_dir "${MANIFEST_DIR}" \
    --task_type "${TASK_TYPE}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --bootstrap_B "${BOOTSTRAP_B}" \
    --seed "${SEED}" \
    --confidence "${CONFIDENCE}" \
    --permutation_M "${PERMUTATION_M}" \
    --out_dir "${out_dir}" \
    "${extra_args[@]}"
}

for task in "${TASKS[@]}"; do
  for window in "${WINDOWS[@]}"; do
    run_one "${task}" "${window}"
  done
done

echo "Done. Outputs under ${OUTPUT_ROOT}"
