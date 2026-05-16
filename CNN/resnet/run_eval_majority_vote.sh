#!/usr/bin/env bash
#
# Run ResNet evaluation with majority vote (no retraining).
# Loads saved checkpoints from --results_dir and reports segment-level and
# participant-level (mean prob + majority vote) metrics.
#
# Usage (from EEG project root):
#   ./CNN/resnet/run_eval_majority_vote.sh [--mode 2s] [--resnet_type 34] [--results_dir DIR]
#
# Examples:
#   ./CNN/resnet/run_eval_majority_vote.sh --mode 2s --resnet_type 34 --results_dir experiment_results
#   ./CNN/resnet/run_eval_majority_vote.sh --mode 4s --resnet_type 18 --results_dir final_logs/RESNET
#
# Gender:
#   python -m CNN.resnet.run_resnet_gender --mode 2s --resnet_type 34 --eval_only --aggregate_by_participant majority_vote --results_dir experiment_results
# Age:
#   python -m CNN.resnet.run_resnet_age --mode 2s --resnet_type 34 --eval_only --aggregate_by_participant majority_vote --results_dir experiment_results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

MODE="${MODE:-2s}"
RESNET_TYPE="${RESNET_TYPE:-34}"
RESULTS_DIR="${RESULTS_DIR:-experiment_results}"
TASK="${TASK:-gender}"  # gender or age

while [[ $# -gt 0 ]]; do
  case $1 in
    --mode) MODE="$2"; shift 2 ;;
    --resnet_type) RESNET_TYPE="$2"; shift 2 ;;
    --results_dir) RESULTS_DIR="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "ResNet evaluation with majority vote (eval only)."
echo "  Mode: $MODE  |  ResNet: $RESNET_TYPE  |  Results dir: $RESULTS_DIR  |  Task: $TASK"
echo ""

if [[ "$TASK" == "age" ]]; then
  python -m CNN.resnet.run_resnet_age \
    --mode "$MODE" \
    --resnet_type "$RESNET_TYPE" \
    --eval_only \
    --aggregate_by_participant majority_vote \
    --results_dir "$RESULTS_DIR"
else
  python -m CNN.resnet.run_resnet_gender \
    --mode "$MODE" \
    --resnet_type "$RESNET_TYPE" \
    --eval_only \
    --aggregate_by_participant majority_vote \
    --results_dir "$RESULTS_DIR"
fi
