#!/bin/bash
# Run saliency analysis for a single model and generate the topography map.
# Saliency maps and topography are written to --output_dir (default: saliency_results).
#
# HOW TO USE — provide the path to your best model (.pth):
#
#   bash saliency_analysis/run_saliency_single.sh \
#       --checkpoint /path/to/your/best_model.pth \
#       --target_type gender \
#       --segment_length 4s \
#       --output_dir /path/to/save/results
#
# Or with environment variables (e.g. in SLURM or a wrapper script):
#
#   export CHECKPOINT=/path/to/your/best_model.pth
#   export OUTPUT_DIR=/path/to/save/results
#   bash saliency_analysis/run_saliency_single.sh --target_type gender --segment_length 4s
#
# Required:
#   --checkpoint <path>   Path to the best model checkpoint (.pth)
#   --target_type        gender | age | combined
#   --segment_length     1s | 2s | 4s  (must match the data the model was trained on)
#
# Output (in --output_dir):
#   topography_<method>.png              Topography (scalp) map
#   saliency_results_<method>.npz        Saliency data
#   average_importance_<method>.png      Bar chart of channel importance
#   ... and other plots (see README)

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
echo "Saliency Analysis - Single Model"
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

# Default configuration (can be overridden by command line arguments or env)
# Use same HDF5 dir as training (e.g. export HDF5_DIR=~/scratch/processed_eeg_data_hdf5)
DEFAULT_HDF5_DIR="data_processing/processed_eeg_data_hdf5_no_compression"
DEFAULT_OUTPUT_DIR="saliency_results"
DEFAULT_DEVICE="cuda"
DEFAULT_NUM_GPUS="1"
# NOTE: Using vanilla_gradients for faster testing (~50x faster than integrated_gradients)
# Change to 'integrated_gradients' for more accurate results (but much slower)
DEFAULT_METHOD="vanilla_gradients"  # Changed from integrated_gradients for faster testing
DEFAULT_BATCH_SIZE="32"
DEFAULT_SPLIT="test"
DEFAULT_TASK_TYPE="both"
DEFAULT_NUM_STEPS="50"
DEFAULT_TOP_K="10"

# Parse command line arguments (CHECKPOINT, OUTPUT_DIR, HDF5_DIR can be set via environment)
CHECKPOINT="${CHECKPOINT:-}"
TARGET_TYPE=""
SEGMENT_LENGTH=""
HDF5_DIR="${HDF5_DIR:-$DEFAULT_HDF5_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
DEVICE="$DEFAULT_DEVICE"
NUM_GPUS="$DEFAULT_NUM_GPUS"
METHOD="$DEFAULT_METHOD"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
MAX_SAMPLES=""
SPLIT="$DEFAULT_SPLIT"
TASK_TYPE="$DEFAULT_TASK_TYPE"
NUM_STEPS="$DEFAULT_NUM_STEPS"
TOP_K="$DEFAULT_TOP_K"
NO_PLOTS=""

# Simple argument parser
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --target_type)
            TARGET_TYPE="$2"
            shift 2
            ;;
        --segment_length)
            SEGMENT_LENGTH="$2"
            shift 2
            ;;
        --hdf5_dir)
            HDF5_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --task_type)
            TASK_TYPE="$2"
            shift 2
            ;;
        --num_steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --no_plots)
            NO_PLOTS="--no_plots"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Passing all remaining arguments to Python script..."
            break
            ;;
    esac
done

# Validate required arguments
if [ -z "$CHECKPOINT" ] || [ -z "$TARGET_TYPE" ] || [ -z "$SEGMENT_LENGTH" ]; then
    echo "Error: Missing required arguments"
    echo ""
    echo "Required: checkpoint path, --target_type, --segment_length"
    echo "  --checkpoint <path>        Path to best model .pth (or set CHECKPOINT env var)"
    echo "  --target_type <type>       Target type (gender, age, combined)"
    echo "  --segment_length <length>  Segment length (1s, 2s, 4s)"
    echo ""
    echo "Optional (or set OUTPUT_DIR / HDF5_DIR env vars):"
    echo "  --output_dir <dir>         Where to save saliency and topography (default: $DEFAULT_OUTPUT_DIR)"
    echo "  --hdf5_dir <dir>           HDF5 data directory (default: $DEFAULT_HDF5_DIR)"
    echo "  --device <device>          Device (auto, cuda, cpu) (default: $DEFAULT_DEVICE)"
    echo "  --num_gpus <n>             Number of GPUs (default: $DEFAULT_NUM_GPUS)"
    echo "  --method <method>          Method (vanilla_gradients, integrated_gradients, both) (default: $DEFAULT_METHOD)"
    echo "  --batch_size <n>           Batch size (default: $DEFAULT_BATCH_SIZE)"
    echo "  --max_samples <n>          Max samples to process (default: all)"
    echo "  --split <split>            Data split (train, val, test) (default: $DEFAULT_SPLIT)"
    echo "  --task_type <type>         Task type (active, passive, both) (default: $DEFAULT_TASK_TYPE)"
    echo "  --num_steps <n>            Steps for integrated gradients (default: $DEFAULT_NUM_STEPS)"
    echo "  --top_k <n>                Top K channels to highlight (default: $DEFAULT_TOP_K)"
    echo "  --no_plots                 Skip generating plots"
    echo ""
    exit 1
fi

# Build command (quote paths so paths with spaces work)
CMD="python3 saliency_analysis/main.py \
    --checkpoint \"$CHECKPOINT\" \
    --target_type $TARGET_TYPE \
    --segment_length $SEGMENT_LENGTH \
    --hdf5_dir \"$HDF5_DIR\" \
    --device $DEVICE \
    --num_gpus $NUM_GPUS \
    --method $METHOD \
    --batch_size $BATCH_SIZE \
    --output_dir \"$OUTPUT_DIR\" \
    --split $SPLIT \
    --task_type $TASK_TYPE \
    --num_steps $NUM_STEPS \
    --top_k $TOP_K"

# Add optional arguments
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

if [ -n "$NO_PLOTS" ]; then
    CMD="$CMD $NO_PLOTS"
fi

# Print configuration
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Target type: $TARGET_TYPE"
echo "  Segment length: $SEGMENT_LENGTH"
echo "  HDF5 directory: $HDF5_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Method: $METHOD"
echo "  Batch size: $BATCH_SIZE"
echo "  Max samples: ${MAX_SAMPLES:-all}"
echo "  Split: $SPLIT"
echo "  Task type: $TASK_TYPE"
echo "  Integrated gradients steps: $NUM_STEPS"
echo "  Top K channels: $TOP_K"
echo "  Generate plots: $([ -z "$NO_PLOTS" ] && echo "yes" || echo "no")"
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
