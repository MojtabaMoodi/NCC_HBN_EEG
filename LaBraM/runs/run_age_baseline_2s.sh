#!/bin/bash
# LaBraM Fine-tuning Script for age_baseline (2s segments)
# Generated automatically - do not edit manually

# Initialize conda FIRST (before loading modules that might override Python)
# Source bashrc first to ensure conda is available
if [ -f ~/.bashrc ]; then
    source ~/.bashrc 2>/dev/null || true
fi

# Initialize conda - use full path if available
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
fi

# Activate conda environment BEFORE loading modules
if command -v conda &> /dev/null; then
    # Try eeg_env first, then base
    if conda env list 2>/dev/null | grep -q "eeg_env"; then
        conda activate eeg_env
    else
        conda activate base 2>/dev/null || true
    fi
fi

# Load required modules for Compute Canada/HPC environment
# Strategy: Load Python module temporarily to get system libraries (libcpupower.so.0, etc.)
# Then explicitly use conda's Python to override the module Python
if command -v module &> /dev/null; then
    # Load Python module to get system libraries (needed for numpy C extensions)
    # We'll override the Python path later to use conda's Python
    module load python/3.10 2>/dev/null || true
    # Load CUDA modules (needed for GPU)
    module load cuda/12.6 2>/dev/null || true
    # Try different cudnn versions (version may vary by system)
    module load cudnn/8.9.7 2>/dev/null || \
    module load cudnn/9.0 2>/dev/null || \
    module load cudnn 2>/dev/null || true
fi

# Override Python to use conda's Python (even though we loaded Python module for libraries)
# This ensures we use conda's Python and packages, but with system libraries available
CONDA_PYTHON="$HOME/miniconda3/envs/eeg_env/bin/python"
if [ -f "$CONDA_PYTHON" ]; then
    export PATH="$HOME/miniconda3/envs/eeg_env/bin:$PATH"
    alias python="$CONDA_PYTHON" 2>/dev/null || true
fi

# Verify we're using conda Python
PYTHON_PATH=$(which python 2>/dev/null || echo "not found")
if [[ "$PYTHON_PATH" == *"conda"* ]] || [[ "$PYTHON_PATH" == *"miniconda"* ]] || [[ "$PYTHON_PATH" == *"anaconda"* ]] || [[ "$PYTHON_PATH" == *"eeg_env"* ]]; then
    echo "Using conda Python: $PYTHON_PATH"
else
    echo "Warning: Not using conda Python. Current: $PYTHON_PATH"
    # Force use conda Python
    if [ -f "$CONDA_PYTHON" ]; then
        export PATH="$HOME/miniconda3/envs/eeg_env/bin:$PATH"
        PYTHON_PATH="$CONDA_PYTHON"
    fi
fi

# Check numpy import (system libraries should now be available from Python module)
if ! "$PYTHON_PATH" -c "import numpy" 2>/dev/null; then
    echo "Warning: numpy import failed even with system libraries loaded."
    echo "Attempting to reinstall numpy..."
    "$PYTHON_PATH" -m pip install --force-reinstall --no-cache-dir numpy 2>/dev/null || true
    
    # Check again
    if ! "$PYTHON_PATH" -c "import numpy" 2>/dev/null; then
        echo "Error: numpy still cannot be imported."
        echo "Current Python: $PYTHON_PATH"
        echo "This may require reinstalling numpy from PyPI instead of Compute Canada wheelhouse."
        echo "Try: $PYTHON_PATH -m pip install --no-binary numpy numpy"
        exit 1
    fi
fi

# Set environment variables for single GPU training (no distributed)
# Unset any existing distributed variables to avoid conflicts
unset RANK
unset WORLD_SIZE
unset LOCAL_RANK
unset MASTER_ADDR
unset MASTER_PORT
unset SLURM_PROCID

# Parse script arguments: pass --output_dir DIR and/or --log_dir DIR when running
OUTPUT_DIR_OVERRIDE=""
LOG_DIR_OVERRIDE=""
OTHER_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --output_dir) OUTPUT_DIR_OVERRIDE="$2"; shift 2 ;;
    --log_dir)    LOG_DIR_OVERRIDE="$2"; shift 2 ;;
    *)            OTHER_ARGS+=("$1"); shift ;;
  esac
done
set -- "${OTHER_ARGS[@]}"

# Dataset configuration
DATASET="age_baseline"
NB_CLASSES=3

# LaBraM fine-tuning hyperparameters (from paper)
# Note: Batch size automatically adjusted based on segment length and dataset type:
#   - 1024 for baseline experiments (1s) - increased from 512
#   - 512 for baseline experiments (2s) - 2x reduction from 1024 for 2s segments
#   - 256 for baseline experiments (4s) - increased from 128
#   - 128 for cross-task CV (1s, eval uses 1.5x = 192) - increased from 64
#   - 16 for cross-task CV with 4s segments (eval uses 1.5x = 24) - increased from 8
EPOCHS=50
BATCH_SIZE=512
LEARNING_RATE=0.0005
WEIGHT_DECAY=0.05
WARMUP_EPOCHS=5
MIN_LR=1e-5
DROP_PATH=0.1
SMOOTHING=0.0
LAYER_DECAY=0.65
CLIP_GRAD=3.0
LAYER_SCALE_INIT_VALUE=0.1
MODEL_EMA_DECAY=0.996

# Data configuration
SEGMENT_LENGTH="2s"

# Model configuration
MODEL="labram_base_patch200_200"
PRETRAINED_PATH="/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/LaBraM/checkpoints/labram-base.pth"

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/${DATASET}_${SEGMENT_LENGTH}}"
LOG_DIR="${LOG_DIR:-./logs/${DATASET}_${SEGMENT_LENGTH}}"
[[ -n "${OUTPUT_DIR_OVERRIDE:-}" ]] && OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
[[ -n "${LOG_DIR_OVERRIDE:-}" ]] && LOG_DIR="$LOG_DIR_OVERRIDE"

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "LaBraM Fine-tuning for age_baseline"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Segment Length: $SEGMENT_LENGTH"
echo "Output Dir: $OUTPUT_DIR"
echo "Model: $MODEL"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "=========================================="

# Change to LaBraM directory
cd "$(dirname "$0")/.."

# Use conda Python explicitly (ensures we use conda's Python even if module Python is loaded)
PYTHON_CMD="${PYTHON_PATH:-python}"
if [ -f "$HOME/miniconda3/envs/eeg_env/bin/python" ]; then
    PYTHON_CMD="$HOME/miniconda3/envs/eeg_env/bin/python"
fi

# Run the fine-tuning (handles CV automatically)
# Unbuffered output so progress appears in log file when stdout is redirected
export PYTHONUNBUFFERED=1
# Note: Task-type evaluation (active/passive) is automatically performed every 5 epochs
# and at the end of training. Results are saved in the output directory.
"$PYTHON_CMD" run_class_finetuning.py \
    --model $MODEL \
    --finetune $PRETRAINED_PATH \
    --dataset $DATASET \
    --nb_classes $NB_CLASSES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_epochs $WARMUP_EPOCHS \
    --min_lr $MIN_LR \
    --drop_path $DROP_PATH \
    --smoothing $SMOOTHING \
    --segment_length $SEGMENT_LENGTH \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --abs_pos_emb \
    --qkv_bias \
    --use_mean_pooling \
    --layer_decay $LAYER_DECAY \
    --layer_scale_init_value $LAYER_SCALE_INIT_VALUE \
    --opt_betas 0.9 0.98 \
    --opt_eps 1e-8 \
    --momentum 0.9 \
    --clip_grad $CLIP_GRAD \
    --model_ema \
    --model_ema_decay $MODEL_EMA_DECAY \
    --save_ckpt \
    --auto_resume \
    --seed 42 \
    --pin_mem \
    --num_workers 10 \
    --aggregate_by_participant majority_vote \
    "$@"

echo "=========================================="
echo "Fine-tuning completed for age_baseline"
echo "Results saved to: $OUTPUT_DIR"
echo "Logs saved to: $LOG_DIR"

