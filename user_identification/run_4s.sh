#!/bin/bash
# Direct script to run user identification training for 4s segments using LaBraM with ArcFace
# This can be run interactively or submitted as a SLURM job

# Initialize conda
source ~/.bashrc || true
eval "$(conda shell.bash hook)" || true

# Activate eeg_env conda environment
conda activate eeg_env

# Change to project root directory
cd /home/mojtabam/projects/aip-aghodsib/mojtabam/EEG

# Run the LaBraM user identification training script with ArcFace
echo "=========================================="
echo "Starting LaBraM User Identification Training: 4s mode with ArcFace"
echo "=========================================="
echo ""

# Check if running in SLURM batch job (not interactive session)
# Interactive SLURM sessions should use torchrun, not srun
if [ -n "$SLURM_JOB_ID" ] && [ -n "$SLURM_NTASKS" ] && [ "$SLURM_NTASKS" != "" ]; then
    # SLURM batch job - set up distributed training with srun
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_PORT=29500
    
    # Calculate WORLD_SIZE: if SLURM_NTASKS is not set, use ntasks-per-node * nnodes
    if [ -z "$SLURM_NTASKS" ] || [ "$SLURM_NTASKS" = "" ]; then
        export WORLD_SIZE=$((SLURM_NTASKS_PER_NODE * SLURM_NNODES))
    else
        export WORLD_SIZE=$SLURM_NTASKS
    fi
    
    # Ensure WORLD_SIZE is set (fallback to 4 if still empty)
    if [ -z "$WORLD_SIZE" ] || [ "$WORLD_SIZE" = "" ]; then
        export WORLD_SIZE=4
    fi
    
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    
    # Use srun for distributed training
    srun python user_identification/train_labram_arcface.py \
        --hdf5_dir /home/mojtabam/scratch/processed_eeg_data_user_identification \
        --segment_length 4s \
        --pretrained_path LaBraM/checkpoints/labram-base.pth \
        --epochs 50 \
        --lr 5e-4 \
        --batch_size 192 \
        --num_workers 8 \
        --layer_decay 0.9 \
        --weight_decay 0.05 \
        --warmup_epochs 10 \
        --clip_grad 0.2 \
        --arcface_margin 0.5 \
        --arcface_scale 256.0 \
        --output_dir ./results_user_id_4s_labram_arcface \
        --save_ckpt_freq 5 \
        --seed 42 \
        --distributed
else
    # Interactive session (SLURM or non-SLURM) - use torchrun
    # For multi-GPU, set CUDA_VISIBLE_DEVICES and use torchrun
    if [ -n "$CUDA_VISIBLE_DEVICES" ] && [ $(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l) -gt 1 ]; then
        # Multiple GPUs available - use torchrun for distributed training
        num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
        echo "Using torchrun with $num_gpus GPUs (interactive session)"
        torchrun --nproc_per_node=$num_gpus user_identification/train_labram_arcface.py \
            --hdf5_dir /home/mojtabam/scratch/processed_eeg_data_user_identification \
            --segment_length 4s \
            --pretrained_path LaBraM/checkpoints/labram-base.pth \
            --epochs 50 \
            --lr 5e-4 \
            --batch_size 192 \
            --num_workers 8 \
            --layer_decay 0.9 \
            --weight_decay 0.05 \
            --warmup_epochs 10 \
            --clip_grad 0.2 \
            --arcface_margin 0.5 \
            --arcface_scale 256.0 \
            --output_dir ./results_user_id_4s_labram_arcface \
            --save_ckpt_freq 5 \
            --seed 42 \
            --distributed
    else
        # Single GPU
        echo "Using single GPU"
        python user_identification/train_labram_arcface.py \
            --hdf5_dir /home/mojtabam/scratch/processed_eeg_data_user_identification \
            --segment_length 4s \
            --pretrained_path LaBraM/checkpoints/labram-base.pth \
            --epochs 50 \
            --lr 5e-4 \
            --batch_size 192 \
            --num_workers 8 \
            --layer_decay 0.9 \
            --weight_decay 0.05 \
            --warmup_epochs 10 \
            --clip_grad 0.2 \
            --arcface_margin 0.5 \
            --arcface_scale 256.0 \
            --output_dir ./results_user_id_4s_labram_arcface \
            --save_ckpt_freq 5 \
            --seed 42
    fi
fi

echo ""
echo "=========================================="
echo "Training completed"
echo "=========================================="
