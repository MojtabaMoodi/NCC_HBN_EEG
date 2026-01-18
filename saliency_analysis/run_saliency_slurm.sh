#!/bin/bash
# SLURM script to run saliency computation in batch mode (all best models)
# 
# Usage:
#   sbatch run_saliency_slurm.sh
#
# Or for interactive SLURM session:
#   salloc --gres=gpu:4 --time=24:00:00 --mem=64G bash run_saliency_slurm.sh
#
# Configuration:
#   - Adjust --gres=gpu:N to change number of GPUs
#   - Adjust --time and --mem based on your needs
#   - Modify the Python command arguments below as needed

#SBATCH --job-name=saliency_batch
#SBATCH --output=saliency_batch_%j.out
#SBATCH --error=saliency_batch_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:4  # Request 4 GPUs (adjust as needed)
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1

# Initialize conda
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true

# Activate conda environment
if command -v conda &> /dev/null; then
    conda activate eeg_env 2>/dev/null || echo "Warning: Could not activate eeg_env conda environment"
fi

# Set working directory
cd /home/mojtabam/projects/aip-aghodsib/mojtabam/EEG

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo ""

# Print GPU information
echo "=========================================="
echo "GPU Information"
echo "=========================================="
nvidia-smi
echo ""
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
echo "Number of GPUs requested: 4"
echo "=========================================="
echo ""

# Configuration
# You can modify these values or set them via environment variables before submitting
HDF5_DIR="${HDF5_DIR:-data_processing/processed_eeg_data_hdf5_no_compression}"
OUTPUT_DIR="${OUTPUT_DIR:-saliency_results}"
DEVICE="${DEVICE:-cuda}"
NUM_GPUS="${NUM_GPUS:-4}"
# NOTE: Using vanilla_gradients for faster testing (~50x faster than integrated_gradients)
# Vanilla gradients: Fast, good for testing and quick results
# Integrated gradients: More accurate but ~50x slower (50 steps per sample)
# To switch back to integrated gradients: export METHOD="integrated_gradients"
METHOD="${METHOD:-vanilla_gradients}"  # Fast default for testing
BATCH_SIZE="${BATCH_SIZE:-32}"
# IMPORTANT: For proper aggregation across all participants, set MAX_SAMPLES="" to process all samples
# Limiting samples (e.g., 1000) is useful for quick testing but may not represent all participants
MAX_SAMPLES="${MAX_SAMPLES:-}"  # Default: process all samples. Set to a number (e.g., 1000) to limit for testing
SPLIT="${SPLIT:-test}"  # Options: train, val, test
TASK_TYPE="${TASK_TYPE:-both}"  # Options: active, passive, both
NUM_STEPS="${NUM_STEPS:-50}"  # Steps for integrated gradients
TOP_K="${TOP_K:-10}"  # Top K channels to highlight

# Print configuration
echo "=========================================="
echo "Configuration"
echo "=========================================="
echo "HDF5 directory: $HDF5_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Number of GPUs: $NUM_GPUS"
echo "Method: $METHOD"
echo "Batch size: $BATCH_SIZE"
echo "Max samples per model: ${MAX_SAMPLES:-all (recommended for proper aggregation)}"
if [ -n "$MAX_SAMPLES" ] && [ "$MAX_SAMPLES" != "none" ] && [ "$MAX_SAMPLES" != "None" ]; then
    echo "  ⚠️  WARNING: Processing only $MAX_SAMPLES samples may not include all participants"
    echo "     For proper aggregation, set MAX_SAMPLES=\"\" to process all test samples"
fi
echo "Split: $SPLIT"
echo "Task type: $TASK_TYPE"
echo "Integrated gradients steps: $NUM_STEPS"
echo "Top K channels: $TOP_K"
echo "=========================================="
echo ""

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

# Run saliency computation
echo "=========================================="
echo "Starting saliency computation..."
echo "=========================================="
echo "Command: $CMD"
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
