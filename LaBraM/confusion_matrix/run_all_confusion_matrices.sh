#!/usr/bin/env bash
#
# Compute LaBraM age/gender confusion matrices for all window lengths (1s, 2s, 4s).
# Requires finetuned checkpoints under final_logs/LaBraM/labram_*/output/
#   {age,gender}_labram_{1s,2s,4s}_best.pth  (primary on disk)
#   checkpoint-best.pth                      (alternate LaBraM save name)
#
# Usage (from EEG project root):
#   bash LaBraM/confusion_matrix/run_all_confusion_matrices.sh
#
# Optional env:
#   DATA_PATH     HDF5 directory (required if not set below)
#   DEVICE        torch device (required if not set below)
#   PROJECT_ROOT  repo root (defaults to parent of LaBraM/)
#   TASK_TYPES    space-separated task filters (default: both active passive)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}"

if [[ -z "${DATA_PATH:-}" ]]; then
  echo "Error: set DATA_PATH to the processed HDF5 directory (e.g. /home/mojtabam/scratch/processed_eeg_data_hdf5)."
  exit 1
fi
if [[ -z "${DEVICE:-}" ]]; then
  echo "Error: set DEVICE (e.g. cuda or cpu)."
  exit 1
fi

RESULTS_DIR="${PROJECT_ROOT}/LaBraM/confusion_matrix/results"
mkdir -p "${RESULTS_DIR}"

TASK_TYPES="${TASK_TYPES:-both active passive}"

COMMON_ARGS=(
  --data_path "${DATA_PATH}"
  --aggregate_by_participant majority_vote
  --batch_size 64
  --device "${DEVICE}"
  --model labram_base_patch200_200
  --qkv_bias true
  --rel_pos_bias true
  --abs_pos_emb true
  --use_mean_pooling true
  --layer_scale_init_value 0.1
  --init_scale 0.001
  --drop 0.0
  --attn_drop_rate 0.0
  --drop_path 0.1
)

resolve_checkpoint() {
  local task="$1"
  local segment="$2"
  local out_dir="${PROJECT_ROOT}/final_logs/LaBraM/labram_${task}_${segment}/output"
  local named="${out_dir}/${task}_labram_${segment}_best.pth"
  local standard="${out_dir}/checkpoint-best.pth"

  if [[ -f "${named}" ]]; then
    echo "${named}"
    return 0
  fi
  if [[ -f "${standard}" ]]; then
    echo "${standard}"
    return 0
  fi

  echo "Error: checkpoint not found for ${task} ${segment}. Tried:" >&2
  echo "  ${named}" >&2
  echo "  ${standard}" >&2
  exit 1
}

output_json_name() {
  local task="$1"
  local segment="$2"
  local task_type="$3"

  if [[ "${task_type}" == "both" ]]; then
    echo "${task}_${segment}_confusion_matrix.json"
  else
    echo "${task}_${segment}_${task_type}_confusion_matrix.json"
  fi
}

run_one() {
  local task="$1"
  local segment="$2"
  local num_workers="$3"
  local task_type="$4"
  local dataset=""
  local checkpoint=""
  local output_json=""

  if [[ "${task}" == "age" ]]; then
    dataset="age_baseline"
  elif [[ "${task}" == "gender" ]]; then
    dataset="gender_baseline"
  else
    echo "Unknown task: ${task}"
    exit 1
  fi

  checkpoint="$(resolve_checkpoint "${task}" "${segment}")"
  output_json="$(output_json_name "${task}" "${segment}" "${task_type}")"

  echo "=== LaBraM ${task} ${segment} task_type=${task_type} (checkpoint: ${checkpoint}) ==="
  python LaBraM/eval_confusion_matrix.py \
    --checkpoint "${checkpoint}" \
    --dataset "${dataset}" \
    --segment_length "${segment}" \
    --task_type "${task_type}" \
    --num_workers "${num_workers}" \
    --output_json "${RESULTS_DIR}/${output_json}" \
    "${COMMON_ARGS[@]}"
}

for task_type in ${TASK_TYPES}; do
  run_one age 1s 0 "${task_type}"
  run_one age 2s 4 "${task_type}"
  run_one age 4s 4 "${task_type}"
  run_one gender 1s 0 "${task_type}"
  run_one gender 2s 4 "${task_type}"
  run_one gender 4s 4 "${task_type}"
done

echo "Done. Results in ${RESULTS_DIR}"
