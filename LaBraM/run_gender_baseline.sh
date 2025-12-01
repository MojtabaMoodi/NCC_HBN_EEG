#!/bin/bash

# LaBraM Fine-tuning Script for Gender Classification (Baseline)
# Based on LaBraM paper hyperparameters

# Set up environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Set up single GPU distributed training
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# Dataset configuration
DATASET="gender_baseline"
SEGMENT_LENGTH="1s"  # Use 1s segments for gender classification
DATA_PATH="/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/data_processing/processed_eeg_data_1s_segments"

# Output directory
OUTPUT_DIR="./outputs/gender_baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Model configuration (LaBraM-Base)
MODEL="labram_base_patch200_200"
INPUT_SIZE=200
NB_CLASSES=1  # Binary classification with single output (like TUAB)

# Training hyperparameters (from LaBraM paper)
BATCH_SIZE=512
EPOCHS=50
LEARNING_RATE=5e-4
MIN_LR=1e-6
WARMUP_EPOCHS=5
WEIGHT_DECAY=0.05
DROP_PATH=0.1
LAYER_DECAY=0.65  # Layer-wise learning rate decay for Base model

# Model architecture
DROP_RATE=0.0
ATTN_DROP_RATE=0.0
LAYER_SCALE_INIT_VALUE=0.1
QKV_BIAS=true
REL_POS_BIAS=true
ABS_POS_EMB=false
USE_MEAN_POOLING=true

# Optimizer settings
OPTIMIZER="adamw"
CLIP_GRAD=3.0

# Training settings
UPDATE_FREQ=1
SAVE_CKPT_FREQ=5
NUM_WORKERS=8
PIN_MEM=true

# Evaluation settings
DISABLE_EVAL_DURING_FINETUNING=false

# Model EMA
MODEL_EMA=true
MODEL_EMA_DECAY=0.996

# Logging
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "LaBraM Gender Classification Fine-tuning"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Segment Length: $SEGMENT_LENGTH"
echo "Data Path: $DATA_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Model: $MODEL"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "=========================================="

# Run fine-tuning
python run_class_finetuning.py \
    --dataset "$DATASET" \
    --segment_length "$SEGMENT_LENGTH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --model "$MODEL" \
    --input_size "$INPUT_SIZE" \
    --nb_classes "$NB_CLASSES" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LEARNING_RATE" \
    --min_lr "$MIN_LR" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --weight_decay "$WEIGHT_DECAY" \
    --drop_path "$DROP_PATH" \
    --layer_decay "$LAYER_DECAY" \
    --drop "$DROP_RATE" \
    --attn_drop_rate "$ATTN_DROP_RATE" \
    --layer_scale_init_value "$LAYER_SCALE_INIT_VALUE" \
    --qkv_bias \
    --abs_pos_emb \
    --use_mean_pooling \
    --opt "$OPTIMIZER" \
    --opt_betas 0.9 0.999 \
    --clip_grad "$CLIP_GRAD" \
    --update_freq "$UPDATE_FREQ" \
    --save_ckpt_freq "$SAVE_CKPT_FREQ" \
    --num_workers "$NUM_WORKERS" \
    --pin_mem \
    --model_ema \
    --model_ema_decay "$MODEL_EMA_DECAY" \
    --save_ckpt \
    --auto_resume \
    --seed 42 \
    --world_size 1 \
    --local_rank -1

echo "=========================================="
echo "Fine-tuning completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "Logs saved to: $LOG_DIR"
echo "=========================================="
