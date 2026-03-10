#!/usr/bin/env bash
#
# Run CNN evaluation with majority vote (no retraining).
# Loads saved checkpoints and reports segment-level, participant (mean prob), and participant (majority vote) accuracy.
#
# Usage (from EEG project root):
#   ./CNN/run_eval_majority_vote.sh [--mode 2s] [--results_dir DIR] [--checkpoint_dir DIR] [--data_path DIR]
#
# Options (or pass directly to main.py via extra args):
#   --target         Task: gender | age | all (default: all). Use gender or age to evaluate only that task.
#   --mode           Segment length: 1s, 2s, 4s (default: 4s)
#   --results_dir    Where to save evaluation outputs (default: experiment_results)
#   --checkpoint_dir Directory containing <experiment_name>_best.pth files (default: results_dir/checkpoints)
#   --data_path      Path to multipart HDF5 data (e.g. ~/scratch/processed_eeg_data_hdf5)
#
# Examples:
#   ./CNN/run_eval_majority_vote.sh --target gender --mode 2s --results_dir ./eval_out --checkpoint_dir /path/to/ckpts --data_path ~/scratch/processed_eeg_data_hdf5
#   ./CNN/run_eval_majority_vote.sh --target age --mode 2s
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MODE="${MODE:-4s}"
RESULTS_DIR="${RESULTS_DIR:-experiment_results}"

echo "Running CNN evaluation with majority vote (eval only, no training)."
echo "  Mode: $MODE  |  Results dir: $RESULTS_DIR"
echo "  Pass --checkpoint_dir /path/to/ckpts --data_path /path/to/hdf5 to override defaults."
echo ""

python CNN/main.py \
  --mode "$MODE" \
  --eval_only \
  --aggregate_by_participant majority_vote \
  --results_dir "$RESULTS_DIR" \
  "$@"
