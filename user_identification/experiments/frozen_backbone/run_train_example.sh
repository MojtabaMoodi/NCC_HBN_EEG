#!/usr/bin/env bash
# Example: frozen LaBraM backbone + trainable embedding projection (before ArcFace).
# Run from anywhere; switches to the EEG project root before invoking Python.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

HDF5_DIR="${HDF5_DIR:?Set HDF5_DIR to your preprocessed user-identification HDF5 directory}"
PRETRAINED_PATH="${PRETRAINED_PATH:?Set PRETRAINED_PATH to the LaBraM pretrain checkpoint (see LaBraM/checkpoints)}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_user_id_frozen_backbone}"

exec python user_identification/train_labram_arcface.py \
  --hdf5_dir "$HDF5_DIR" \
  --pretrained_path "$PRETRAINED_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --segment_length 4s \
  --freeze_backbone \
  "$@"
