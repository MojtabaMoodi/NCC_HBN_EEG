#!/usr/bin/env python3
"""
Generate run scripts for all valid dataset names.

This script creates individual shell scripts for each experiment type,
following LaBraM paper hyperparameters for fine-tuning.
"""

import os
from dataset_config import VALID_DATASET_NAMES, DATASET_TYPE_CONFIGS, get_dataset_type_and_params

def generate_run_script(dataset_name, segment_length):
    """Generate a run script for a specific dataset and segment length."""
    
    # Get dataset configuration
    dataset_type, kwargs = get_dataset_type_and_params(dataset_name)
    config = DATASET_TYPE_CONFIGS[dataset_type]
    
    # Determine if this is a cross-validation or cross-task experiment
    is_cv = 'cv' in dataset_name.lower()
    is_cross_task = 'cross_task' in dataset_name.lower()
    is_cross_task_cv = 'cross_task_cv' in dataset_name.lower()
    
    # Check if this experiment type is disabled
    is_disabled = is_cv or is_cross_task or is_cross_task_cv
    
    # Adjust batch size to avoid OOM:
    # - 4s segments are 4x larger in memory
    # - 2s segments are 2x larger in memory
    # - Cross-task CV loads data from both task types (more memory)
    # Increased batch sizes for better training efficiency
    if segment_length == "4s" and is_cross_task_cv:
        batch_size = 16  # Increased from 8: Most memory intensive (eval uses 1.5x = 24)
    elif segment_length == "4s":
        batch_size = 256  # Increased from 128: 4x reduction from 1024 for 4s segments
    elif segment_length == "2s":
        batch_size = 512  # 2x reduction from 1024 for 2s segments
    elif is_cross_task_cv:
        batch_size = 128  # Increased from 64: Cross-task CV with reasonable batch size
    elif is_cv:
        batch_size = 512  # Increased from 256: Regular CV: load 5 folds into memory during prep
    else:
        batch_size = 1024  # Increased from 512: Baseline experiments for better efficiency
    
    # Build checkpoint arguments  
    if is_cross_task_cv:
        # Enable checkpoint saving with cache clearing to prevent OOM
        ckpt_args = " \\\n    --save_ckpt \\\n    --save_ckpt_freq 999"
        eval_flag = ""
    else:
        ckpt_args = " \\\n    --save_ckpt \\\n    --auto_resume"
        eval_flag = ""
    
    # Add warning for disabled experiments
    warning = ""
    if is_disabled:
        if is_cv and not is_cross_task:
            warning = """
# ⚠️  WARNING: Cross-validation experiments are currently DISABLED
# Cross-validation functionality has been commented out in labram_dataset.py
# This script will fail until cross-validation is re-enabled for HDF5 format
# See labram_dataset.py for details
"""
        elif is_cross_task:
            warning = """
# ⚠️  WARNING: Cross-task experiments are currently DISABLED
# Cross-task functionality has been commented out in labram_dataset.py
# This script will fail until cross-task is re-enabled for HDF5 format
# See labram_dataset.py for details
"""
    
    # Script content
    script_content = f"""#!/bin/bash
# LaBraM Fine-tuning Script for {dataset_name} ({segment_length} segments)
# Generated automatically - do not edit manually{warning}

# Set environment variables for single GPU training (no distributed)
# Unset any existing distributed variables to avoid conflicts
unset RANK
unset WORLD_SIZE
unset LOCAL_RANK
unset MASTER_ADDR
unset MASTER_PORT
unset SLURM_PROCID

# Dataset configuration
DATASET="{dataset_name}"
NB_CLASSES={config['nb_classes']}

# LaBraM fine-tuning hyperparameters (from paper)
# Note: Batch size automatically adjusted based on segment length and dataset type:
#   - 1024 for baseline experiments (1s) - increased from 512
#   - 512 for baseline experiments (2s)
#   - 256 for baseline experiments (4s) - increased from 128
#   - 128 for cross-task CV (1s, eval uses 1.5x = 192) - increased from 64
#   - 16 for cross-task CV with 4s segments (eval uses 1.5x = 24) - increased from 8
# Note: Data is loaded from HDF5 format (processed_eeg_data_hdf5)
EPOCHS={20 if is_cv else 50}
BATCH_SIZE={batch_size}
LEARNING_RATE={2e-4 if is_cross_task_cv else 5e-4}
WEIGHT_DECAY=0.05
WARMUP_EPOCHS=5
MIN_LR=1e-5
DROP_PATH=0.1
SMOOTHING=0.0
LAYER_DECAY=0.65
CLIP_GRAD={10.0 if is_cross_task_cv else 3.0}
LAYER_SCALE_INIT_VALUE=0.1
MODEL_EMA_DECAY=0.996
"""

    # Add cross-validation specific parameters
    if is_cv:
        n_folds = kwargs.get('n_folds', 5)
        script_content += f"""
# Cross-validation parameters
N_FOLDS={n_folds}
"""
    
    # Add cross-task specific parameters
    if 'cross_task' in dataset_name.lower():
        train_task_type = kwargs.get('train_task_type', 'active')
        val_test_task_type = kwargs.get('val_test_task_type', 'passive')
        script_content += f"""
# Cross-task parameters
TRAIN_TASK_TYPE="{train_task_type}"
VAL_TEST_TASK_TYPE="{val_test_task_type}"
"""
    
    # Add segment length parameter
    script_content += f"""
# Data configuration
SEGMENT_LENGTH="{segment_length}"

# Model configuration
MODEL="labram_base_patch200_200"
PRETRAINED_PATH="/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/LaBraM/checkpoints/labram-base.pth"

# Output configuration
OUTPUT_DIR="./outputs/${{DATASET}}_${{SEGMENT_LENGTH}}"
LOG_DIR="./logs/${{DATASET}}_${{SEGMENT_LENGTH}}"

# Create output directories
mkdir -p "${{OUTPUT_DIR}}"
mkdir -p "${{LOG_DIR}}"

echo "=========================================="
echo "LaBraM Fine-tuning for {dataset_name}"
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
python run_class_finetuning.py \\
    --model $MODEL \\
    --finetune $PRETRAINED_PATH \\
    --dataset $DATASET \\
    --nb_classes $NB_CLASSES \\
    --epochs $EPOCHS \\
    --batch_size $BATCH_SIZE \\
    --lr $LEARNING_RATE \\
    --weight_decay $WEIGHT_DECAY \\
    --warmup_epochs $WARMUP_EPOCHS \\
    --min_lr $MIN_LR \\
    --drop_path $DROP_PATH \\
    --smoothing $SMOOTHING \\
    --segment_length $SEGMENT_LENGTH \\
    --output_dir $OUTPUT_DIR \\
    --log_dir $LOG_DIR \\
    --abs_pos_emb \\
    --qkv_bias \\
    --use_mean_pooling \\
    --layer_decay $LAYER_DECAY \\
    --layer_scale_init_value $LAYER_SCALE_INIT_VALUE \\
    --opt_betas 0.9 0.98 \\
    --opt_eps 1e-8 \\
    --momentum 0.9 \\
    --clip_grad $CLIP_GRAD \\
    --model_ema \\
    --model_ema_decay $MODEL_EMA_DECAY{ckpt_args} \\
    --seed 42 \\
    --pin_mem \\
    --num_workers 10{eval_flag}

echo "=========================================="
echo "Fine-tuning completed for {dataset_name}"
echo "Results saved to: $OUTPUT_DIR"
echo "Logs saved to: $LOG_DIR"
"""

    return script_content

def main():
    """Generate all run scripts."""
    
    print("Generating run scripts for all valid dataset names...")
    print("Note: Cross-validation and cross-task experiments are currently disabled")
    print("      They will be marked with warnings in the generated scripts")
    
    segment_lengths = ["1s", "2s", "4s"]  # Support all segment lengths
    total_scripts = 0
    
    for dataset_name in VALID_DATASET_NAMES:
        for segment_length in segment_lengths:
            print(f"Generating script for: {dataset_name} ({segment_length})")
            
            # Generate script content
            script_content = generate_run_script(dataset_name, segment_length)
            
            # Write script file
            script_filename = f"runs/run_{dataset_name}_{segment_length}.sh"
            with open(script_filename, 'w') as f:
                f.write(script_content)
            
            # Make script executable
            os.chmod(script_filename, 0o755)
            
            print(f"  Created: {script_filename}")
            total_scripts += 1
    
    print(f"\nGenerated {total_scripts} run scripts in the 'runs' directory")
    print("\nTo run an experiment:")
    print("  ./runs/run_gender_baseline_1s.sh")
    print("  ./runs/run_gender_baseline_2s.sh")
    print("  ./runs/run_gender_baseline_4s.sh")
    print("  ./runs/run_age_baseline_1s.sh")
    print("  etc.")
    print("\n⚠️  Note: CV and cross-task scripts are disabled and will fail.")
    print("   Only baseline experiments are currently supported with HDF5 format.")

if __name__ == "__main__":
    main()
