#!/usr/bin/env bash
#
# Run LaBraM evaluation only with majority vote.
# Loads the best checkpoint and reports segment-level, participant (mean prob), and participant (majority vote) accuracy.
#
# Task (gender vs age): set --dataset and --nb_classes to match your trained model:
#   Gender prediction:  --dataset gender_baseline --nb_classes 1  (default)
#   Age prediction:     --dataset age_classification --nb_classes 3
#
# Usage (from LaBraM directory or project root):
#   ./LaBraM/run_eval_majority_vote.sh --output_dir /path/to/your/run [--dataset gender_baseline|age_classification] [--nb_classes 1|3]
#   ./LaBraM/run_eval_majority_vote.sh --resume /path/to/checkpoint-best.pth --output_dir ./out --dataset age_classification --nb_classes 3 --data_path ~/scratch/processed_eeg_data_hdf5

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-}"
DATASET="${DATASET:-gender_baseline}"
SEGMENT_LENGTH="${SEGMENT_LENGTH:-2s}"
NB_CLASSES="${NB_CLASSES:-1}"
PRETRAINED_PATH="${PRETRAINED_PATH:-$SCRIPT_DIR/checkpoints/labram-base.pth}"

RESUME=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --segment_length) SEGMENT_LENGTH="$2"; shift 2 ;;
    --nb_classes) NB_CLASSES="$2"; shift 2 ;;
    --data_path) DATA_PATH="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$OUTPUT_DIR" && -z "$RESUME" ]]; then
  echo "Error: set OUTPUT_DIR (directory containing checkpoint-best.pth) or --resume /path/to/checkpoint-best.pth."
  echo "Example: OUTPUT_DIR=./outputs/gender_baseline_2s $0"
  echo "Or: $0 --output_dir ./outputs/gender_baseline_2s"
  echo "Or: $0 --resume /path/to/checkpoint-best.pth --output_dir ./eval_out"
  exit 1
fi
[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="."

if [[ -z "$RESUME" && ! -f "$OUTPUT_DIR/checkpoint-best.pth" ]]; then
  echo "Warning: $OUTPUT_DIR/checkpoint-best.pth not found. Will try loading anyway (e.g. checkpoint.pth with --auto_resume)."
fi

echo "LaBraM evaluation with majority vote (eval only)."
echo "  Task: $DATASET (nb_classes=$NB_CLASSES)  |  Output dir: $OUTPUT_DIR  |  Checkpoint: ${RESUME:-$OUTPUT_DIR/checkpoint-best.pth}"
echo "  Segment length: $SEGMENT_LENGTH"
echo ""

DATA_PATH_ARGS=()
[[ -n "${DATA_PATH:-}" ]] && DATA_PATH_ARGS=(--data_path "$DATA_PATH")
RESUME_ARGS=()
[[ -n "$RESUME" ]] && RESUME_ARGS=(--resume "$RESUME")

python run_class_finetuning.py \
  --eval \
  --aggregate_by_participant majority_vote \
  --output_dir "$OUTPUT_DIR" \
  --dataset "$DATASET" \
  --segment_length "$SEGMENT_LENGTH" \
  --nb_classes "$NB_CLASSES" \
  --finetune "$PRETRAINED_PATH" \
  --model labram_base_patch200_200 \
  --abs_pos_emb \
  --qkv_bias \
  --use_mean_pooling \
  --auto_resume \
  "${RESUME_ARGS[@]}" \
  "${DATA_PATH_ARGS[@]}" \
  "$@"
