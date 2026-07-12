#!/bin/bash
# Complete script to run evaluation on GPU server
# This script does everything: generates manifests and runs evaluation

set -e  # Exit on error

echo "=========================================="
echo "EEG Subject-Level Bootstrap Evaluation"
echo "=========================================="
echo ""

# Configuration
HDF5_DIR="data_processing/processed_eeg_data_hdf5_no_compression"
SEGMENT_LENGTH="4s"
TASK_TYPE="both"
OUTPUT_DIR="eval_subject_bootstrap/outputs/age_evaluation_4s_$(date +%Y%m%d_%H%M%S)"
MANIFEST_DIR="eval_subject_bootstrap/manifests"
CONFIG_FILE="eval_subject_bootstrap/config.yaml"

# Create directories
mkdir -p "$MANIFEST_DIR"
mkdir -p "$(dirname "$OUTPUT_DIR")"

echo "Step 1: Generating manifests..."
echo "-----------------------------------"

# Generate train manifest
TRAIN_MANIFEST="$MANIFEST_DIR/train_manifest_${SEGMENT_LENGTH}.csv"
if [ ! -f "$TRAIN_MANIFEST" ]; then
    echo "Generating train manifest..."
    python eval_subject_bootstrap/generate_manifests.py \
        --hdf5_dir "$HDF5_DIR" \
        --split train \
        --segment_length "$SEGMENT_LENGTH" \
        --task_type "$TASK_TYPE" \
        --output "$TRAIN_MANIFEST"
    echo "✓ Train manifest created: $TRAIN_MANIFEST"
else
    echo "✓ Train manifest already exists: $TRAIN_MANIFEST"
fi

# Generate test manifest
TEST_MANIFEST="$MANIFEST_DIR/test_manifest_${SEGMENT_LENGTH}.csv"
if [ ! -f "$TEST_MANIFEST" ]; then
    echo "Generating test manifest..."
    python eval_subject_bootstrap/generate_manifests.py \
        --hdf5_dir "$HDF5_DIR" \
        --split test \
        --segment_length "$SEGMENT_LENGTH" \
        --task_type "$TASK_TYPE" \
        --output "$TEST_MANIFEST"
    echo "✓ Test manifest created: $TEST_MANIFEST"
else
    echo "✓ Test manifest already exists: $TEST_MANIFEST"
fi

echo ""
echo "Step 2: Running evaluation..."
echo "-----------------------------------"

# Check if GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || {
    echo "Warning: Could not check CUDA availability. Continuing anyway..."
}

# Run evaluation
python eval_subject_bootstrap/run_bootstrap_evaluation.py \
    --train_manifest "$TRAIN_MANIFEST" \
    --test_manifest "$TEST_MANIFEST" \
    --models_config "$CONFIG_FILE" \
    --hdf5_base_dir "$HDF5_DIR" \
    --out_dir "$OUTPUT_DIR" \
    --batch_size 64 \
    --num_workers 8 \
    --device cuda \
    --bootstrap_B 2000 \
    --permutation_M 1000 \
    --seed 123 \
    --primary_metric balanced_accuracy \
    --save_npz

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "Key output files:"
echo "  - results.json: Complete results with bootstrap CIs"
echo "  - summary.csv: Summary table"
echo "  - bootstrap_deltas_*.csv: Bootstrap distributions"
echo ""
