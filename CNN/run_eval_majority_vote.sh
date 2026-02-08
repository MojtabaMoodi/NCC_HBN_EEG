#!/usr/bin/env bash
#
# Run CNN evaluation with majority vote (no retraining).
# Loads saved checkpoints and reports:
#   - Segment-level (per-window) accuracy
#   - Participant-level mean probability accuracy
#   - Participant-level majority vote accuracy
#
# Usage:
#   From EEG project root:  ./CNN/run_eval_majority_vote.sh
#   From CNN directory:     ./run_eval_majority_vote.sh
#
# Override results directory (where checkpoints live):
#   ./CNN/run_eval_majority_vote.sh --results_dir CNN_4s_majority_vote
#
# Override segment length (must match the trained checkpoints):
#   ./CNN/run_eval_majority_vote.sh --mode 4s --results_dir experiment_results
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root = parent of CNN
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Defaults (override with env or pass as args, e.g. --results_dir my_dir)
MODE="${MODE:-4s}"
RESULTS_DIR="${RESULTS_DIR:-experiment_results}"

echo "Running CNN evaluation with majority vote (eval only, no training)."
echo "  Mode: $MODE  |  Results dir: $RESULTS_DIR"
echo "  Override: $0 --mode 4s --results_dir /path/to/checkpoints"
echo ""

python CNN/main.py \
  --mode "$MODE" \
  --eval_only \
  --aggregate_by_participant majority_vote \
  --results_dir "$RESULTS_DIR" \
  "$@"
