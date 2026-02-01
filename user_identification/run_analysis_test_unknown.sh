#!/bin/bash
# Run confidence analysis on test and unknown splits.
# Use after training: point CHECKPOINT to your best_model.pth and set OUTPUT_DIR for analysis results.
#
# Usage:
#   ./user_identification/run_analysis_test_unknown.sh
#   CHECKPOINT=/path/to/best_model.pth OUTPUT_DIR=/path/to/analysis ./user_identification/run_analysis_test_unknown.sh
#   CHECKPOINT=/path/to/best_model.pth HDF5_DIR=/path/to/hdf5 OUTPUT_DIR=/path/to/analysis ./user_identification/run_analysis_test_unknown.sh

set -e

# Path to trained checkpoint (required if not default)
CHECKPOINT="${CHECKPOINT:-}"
# HDF5 directory (preprocessed user-identification data)
HDF5_DIR="${HDF5_DIR:-/home/mojtabam/scratch/processed_eeg_data_user_identification}"
# Where to save analysis results (plots, stats)
OUTPUT_DIR="${OUTPUT_DIR:-./confidence_analysis_test_unknown}"
# Segment length (must match training)
SEGMENT_LENGTH="${SEGMENT_LENGTH:-4s}"

cd "$(dirname "$0")/.."

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: CHECKPOINT=/path/to/best_model.pth [OUTPUT_DIR=...] [HDF5_DIR=...] $0"
    echo "  CHECKPOINT   Path to best_model.pth (or checkpoint_epoch_X.pth)"
    echo "  OUTPUT_DIR  Where to save analysis results (default: ./confidence_analysis_test_unknown)"
    echo "  HDF5_DIR    HDF5 data directory (default: $HDF5_DIR)"
    echo "  SEGMENT_LENGTH  Segment length, e.g. 4s (default: 4s)"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "=========================================="
echo "Confidence analysis: test + unknown"
echo "=========================================="
echo "Checkpoint:  $CHECKPOINT"
echo "HDF5 dir:    $HDF5_DIR"
echo "Output dir:  $OUTPUT_DIR"
echo "Segment:     $SEGMENT_LENGTH"
echo "=========================================="

python user_identification/analyze_confidence.py \
    --checkpoint "$CHECKPOINT" \
    --hdf5_dir "$HDF5_DIR" \
    --segment_length "$SEGMENT_LENGTH" \
    --split both \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 128 \
    --num_workers 4

echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "  test/    - test split plots and user_confidence_statistics.json"
echo "  unknown/ - unknown (held-out) users confidence distribution"
