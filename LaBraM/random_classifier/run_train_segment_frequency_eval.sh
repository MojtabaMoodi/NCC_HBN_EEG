#!/usr/bin/env bash
#
# Evaluate random classifier baselines (train_segment_frequency) for age and gender
# at 1s / 2s / 4s. Writes JSON under LaBraM/random_classifier/results/.
#
# Usage (from EEG project root, eeg_env active):
#   bash LaBraM/random_classifier/run_train_segment_frequency_eval.sh
#
# Optional:
#   DATA_PATH=/path/to/hdf5 RANDOM_SEED=42 bash LaBraM/random_classifier/run_train_segment_frequency_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
DATA_PATH="${DATA_PATH:-/home/mojtabam/scratch/processed_eeg_data_hdf5}"
RANDOM_SEED="${RANDOM_SEED:-42}"
BATCH_SIZE=64

mkdir -p "${RESULTS_DIR}"
cd "${PROJECT_ROOT}"

run_one() {
  local dataset="$1"
  local segment="$2"
  local num_workers="$3"
  local prefix="$4"
  local out="${RESULTS_DIR}/${prefix}_${segment}_train_segment_frequency.json"

  echo ""
  echo "=== ${prefix} ${segment} -> ${out} ==="
  python LaBraM/eval_random_classifier.py \
    --dataset "${dataset}" \
    --segment_length "${segment}" \
    --data_path "${DATA_PATH}" \
    --task_type both \
    --random_seed "${RANDOM_SEED}" \
    --random_strategy train_segment_frequency \
    --aggregate_by_participant majority_vote \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${num_workers}" \
    --output_json "${out}"
}

for segment in 1s 2s 4s; do
  if [[ "${segment}" == "1s" ]]; then
    workers=0
  else
    workers=4
  fi
  run_one age_baseline "${segment}" "${workers}" age
  run_one gender_baseline "${segment}" "${workers}" gender
done

echo ""
echo "Done. Results in: ${RESULTS_DIR}"
