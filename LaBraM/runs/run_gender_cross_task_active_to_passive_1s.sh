#!/bin/bash
# LaBraM Fine-tuning Script for gender_cross_task_active_to_passive (1s segments)
# Generated automatically - do not edit manually

# ⚠️  WARNING: Cross-task experiments are currently DISABLED
# Cross-task functionality has been commented out in labram_dataset.py
# This script will fail until cross-task is re-enabled for HDF5 format
# See labram_dataset.py for details

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
DATASET="gender_cross_task_active_to_passive"
NB_CLASSES=1

# LaBraM fine-tuning hyperparameters (from paper)
# Note: Batch size automatically adjusted based on segment length and dataset type:
#   - 1024 for baseline experiments (1s) - increased from 512
#   - 128 for cross-task CV (1s, eval uses 1.5x = 192) - increased from 64
#   - 256 for 4s segments - increased from 128
#   - 16 for cross-task CV with 4s segments (eval uses 1.5x = 24) - increased from 8
EPOCHS=50
BATCH_SIZE=1024
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

# Cross-task parameters
TRAIN_TASK_TYPE="active"
VAL_TEST_TASK_TYPE="passive"

# Data configuration
SEGMENT_LENGTH="1s"

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
echo "LaBraM Fine-tuning for gender_cross_task_active_to_passive"
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

# Run the fine-tuning (handles CV automatically)
# Unbuffered output so progress appears in log file when stdout is redirected
export PYTHONUNBUFFERED=1
python run_class_finetuning.py \
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
    --num_workers 10

echo "=========================================="
echo "Fine-tuning completed for gender_cross_task_active_to_passive"
echo "Results saved to: $OUTPUT_DIR"
echo "Logs saved to: $LOG_DIR"
