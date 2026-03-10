#!/bin/bash
# Run saliency analysis in batch mode (all best models)
# This script can be run locally or submitted as a SLURM job
#
# Usage (local):
#   bash run_saliency_batch.sh
#
# Usage (SLURM):
#   sbatch run_saliency_batch.sh
#
# Usage (interactive SLURM session):
#   salloc --gres=gpu:4 --time=24:00:00 --mem=64G bash run_saliency_batch.sh
#
# To use your own list of checkpoint paths (one per line in a file):
#   CHECKPOINT_LIST=my_checkpoints.txt SEGMENT_LENGTH=4s OUTPUT_DIR=results bash run_saliency_batch.sh

# Initialize conda if available
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true

# Activate conda environment if it exists
if command -v conda &> /dev/null; then
    conda activate eeg_env 2>/dev/null || echo "Warning: Could not activate eeg_env conda environment"
fi

# Set working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Saliency Analysis - Batch Mode"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo "Working directory: $(pwd)"
echo ""

# Print GPU information if available
if command -v nvidia-smi &> /dev/null; then
    echo "=========================================="
    echo "GPU Information"
    echo "=========================================="
    nvidia-smi
    echo ""
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
    echo "=========================================="
    echo ""
fi

# Default configuration
HDF5_DIR="${HDF5_DIR:-data_processing/processed_eeg_data_hdf5_no_compression}"
OUTPUT_DIR="${OUTPUT_DIR:-saliency_results}"
DEVICE="${DEVICE:-cuda}"
NUM_GPUS="${NUM_GPUS:-1}"
# NOTE: Using vanilla_gradients for faster testing (~50x faster than integrated_gradients)
# Vanilla gradients: Fast, good for testing and quick results
# Integrated gradients: More accurate but ~50x slower (50 steps per sample)
# To switch back to integrated gradients: export METHOD="integrated_gradients"
METHOD="${METHOD:-vanilla_gradients}"  # Fast default for testing
BATCH_SIZE="${BATCH_SIZE:-128}"
# IMPORTANT: For proper aggregation across all participants, set MAX_SAMPLES="" to process all samples
# Limiting samples (e.g., 1000) is useful for quick testing but may not represent all participants
MAX_SAMPLES="${MAX_SAMPLES:-}"  # Default: process all samples. Set to a number (e.g., 1000) to limit for testing
SPLIT="${SPLIT:-test}"
TASK_TYPE="${TASK_TYPE:-both}"
NUM_STEPS="${NUM_STEPS:-50}"
TOP_K="${TOP_K:-10}"
# Optional: process only these checkpoints (file with one path per line). Requires SEGMENT_LENGTH.
CHECKPOINT_LIST="${CHECKPOINT_LIST:-}"
SEGMENT_LENGTH="${SEGMENT_LENGTH:-}"

# Build command
CMD="python3 saliency_analysis/main.py \
    --batch \
    --hdf5_dir $HDF5_DIR \
    --device $DEVICE \
    --num_gpus $NUM_GPUS \
    --method $METHOD \
    --batch_size $BATCH_SIZE \
    --output_dir $OUTPUT_DIR \
    --split $SPLIT \
    --task_type $TASK_TYPE \
    --num_steps $NUM_STEPS \
    --top_k $TOP_K"

# Add max_samples if specified
if [ -n "$MAX_SAMPLES" ] && [ "$MAX_SAMPLES" != "none" ] && [ "$MAX_SAMPLES" != "None" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

# Add checkpoint list and segment length if using custom checkpoint list
if [ -n "$CHECKPOINT_LIST" ]; then
    if [ -z "$SEGMENT_LENGTH" ]; then
        echo "Error: SEGMENT_LENGTH is required when using CHECKPOINT_LIST (e.g. export SEGMENT_LENGTH=4s)"
        exit 1
    fi
    CMD="$CMD --checkpoint_list $CHECKPOINT_LIST --segment_length $SEGMENT_LENGTH"
fi

# Print configuration
echo "Configuration:"
echo "  HDF5 directory: $HDF5_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Method: $METHOD"
echo "  Batch size: $BATCH_SIZE"
echo "  Max samples: ${MAX_SAMPLES:-all (recommended for proper aggregation)}"
if [ -n "$MAX_SAMPLES" ] && [ "$MAX_SAMPLES" != "none" ] && [ "$MAX_SAMPLES" != "None" ]; then
    echo "  ⚠️  WARNING: Processing only $MAX_SAMPLES samples may not include all participants"
    echo "     For proper aggregation, set MAX_SAMPLES=\"\" to process all test samples"
fi
echo "  Split: $SPLIT"
echo "  Task type: $TASK_TYPE"
echo "  Integrated gradients steps: $NUM_STEPS"
echo "  Top K channels: $TOP_K"
[ -n "$CHECKPOINT_LIST" ] && echo "  Checkpoint list: $CHECKPOINT_LIST (segment_length: $SEGMENT_LENGTH)"
echo ""

# Run the command
echo "=========================================="
echo "Starting saliency computation..."
echo "=========================================="
echo ""

eval $CMD

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Saliency computation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "Saliency computation failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
