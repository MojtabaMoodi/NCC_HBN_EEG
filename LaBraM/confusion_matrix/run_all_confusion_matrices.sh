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
#   DATA_PATH   HDF5 directory (required if not set below)
#   DEVICE      torch device (required if not set below)
#   PROJECT_ROOT  repo root (defaults to parent of LaBraM/)

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

COMMON_ARGS=(
  --data_path "${DATA_PATH}"
  --task_type both
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

run_one() {
  local task="$1"
  local segment="$2"
  local num_workers="$3"
  local dataset=""
  local checkpoint=""

  if [[ "${task}" == "age" ]]; then
    dataset="age_baseline"
  elif [[ "${task}" == "gender" ]]; then
    dataset="gender_baseline"
  else
    echo "Unknown task: ${task}"
    exit 1
  fi

  checkpoint="$(resolve_checkpoint "${task}" "${segment}")"

  echo "=== LaBraM ${task} ${segment} (checkpoint: ${checkpoint}) ==="
  python LaBraM/eval_confusion_matrix.py \
    --checkpoint "${checkpoint}" \
    --dataset "${dataset}" \
    --segment_length "${segment}" \
    --num_workers "${num_workers}" \
    --output_json "${RESULTS_DIR}/${task}_${segment}_confusion_matrix.json" \
    "${COMMON_ARGS[@]}"
}

run_one age 1s 0
run_one age 2s 4
run_one age 4s 4
run_one gender 1s 0
run_one gender 2s 4
run_one gender 4s 4

echo "Done. Results in ${RESULTS_DIR}"
