#!/bin/bash
# Direct script to run user identification training for 4s segments using LaBraM with ArcFace
# Supports single-GPU and multi-GPU (interactive or SLURM).
#
# Single GPU (default when only one GPU visible):
#   ./user_identification/run_4s.sh
#   NUM_GPUS=1 ./user_identification/run_4s.sh   # force single GPU
#
# Multiple GPUs (auto-detect from CUDA_VISIBLE_DEVICES, or set NUM_GPUS):
#   NUM_GPUS=4 ./user_identification/run_4s.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./user_identification/run_4s.sh
#
# Save results to a different location:
#   OUTPUT_DIR=/path/to/my_run ./user_identification/run_4s.sh

# Optional: directory where checkpoints and logs are saved (default: ./results_user_id_4s_labram_arcface)
OUTPUT_DIR="${OUTPUT_DIR:-./results_user_id_4s_labram_arcface}"
# Optional: number of GPUs to use (1 = single-GPU, no distributed; 2+ = multi-GPU with torchrun). Unset = auto-detect.
NUM_GPUS="${NUM_GPUS:-}"

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
echo "Output directory: $OUTPUT_DIR"
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
    
    # Check if best_model.pth exists to resume training
    if [ -f "$OUTPUT_DIR/best_model.pth" ]; then
        echo "Found best_model.pth - resuming training from checkpoint"
        RESUME_ARG="--resume best_model.pth"
    else
        RESUME_ARG=""
        echo "No checkpoint found - starting training from scratch"
    fi
    
    # Use srun for distributed training
    srun python user_identification/train_labram_arcface.py \
        --hdf5_dir /home/mojtabam/scratch/processed_eeg_data_user_identification \
        --segment_length 4s \
        --pretrained_path LaBraM/checkpoints/labram-base.pth \
        --epochs 200 \
        --lr 5e-4 \
        --batch_size 192 \
        --num_workers 8 \
        --layer_decay 0.9 \
        --weight_decay 0.05 \
        --warmup_epochs 10 \
        --clip_grad 0.2 \
        --arcface_margin 0.5 \
        --arcface_scale 256.0 \
        --output_dir "$OUTPUT_DIR" \
        --save_ckpt_freq 5 \
        --early_stopping_patience 10 \
        --early_stopping_min_delta 0.0001 \
        --seed 42 \
        --distributed \
        $RESUME_ARG
else
    # Interactive session (SLURM salloc or local): single-GPU or multi-GPU via NUM_GPUS / CUDA_VISIBLE_DEVICES
    use_multi_gpu=false
    if [ -n "$NUM_GPUS" ]; then
        if [ "$NUM_GPUS" -eq 1 ]; then
            use_multi_gpu=false
        else
            use_multi_gpu=true
            num_gpus=$NUM_GPUS
        fi
    else
        # Auto-detect: multi-GPU only if CUDA_VISIBLE_DEVICES has more than one GPU
        if [ -n "$CUDA_VISIBLE_DEVICES" ] && [ $(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l) -gt 1 ]; then
            use_multi_gpu=true
            num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
        else
            use_multi_gpu=false
        fi
    fi

    if [ "$use_multi_gpu" = true ]; then
        # Multi-GPU: torchrun sets RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, LOCAL_RANK
        echo "Using torchrun with $num_gpus GPUs"
        if [ -f "$OUTPUT_DIR/best_model.pth" ]; then
            echo "Found best_model.pth - resuming training from checkpoint"
            RESUME_ARG="--resume best_model.pth"
        else
            RESUME_ARG=""
            echo "No checkpoint found - starting training from scratch"
        fi
        torchrun --nproc_per_node=$num_gpus user_identification/train_labram_arcface.py \
            --hdf5_dir /home/mojtabam/scratch/processed_eeg_data_user_identification \
            --segment_length 4s \
            --pretrained_path LaBraM/checkpoints/labram-base.pth \
            --epochs 200 \
            --lr 5e-4 \
            --batch_size 192 \
            --num_workers 8 \
            --layer_decay 0.9 \
            --weight_decay 0.05 \
            --warmup_epochs 10 \
            --clip_grad 0.2 \
            --arcface_margin 0.5 \
            --arcface_scale 256.0 \
            --output_dir "$OUTPUT_DIR" \
            --save_ckpt_freq 5 \
            --seed 42 \
            --distributed \
            $RESUME_ARG
    else
        # Single GPU: unset distributed env vars so the script does not init process group
        unset RANK WORLD_SIZE MASTER_ADDR MASTER_PORT LOCAL_RANK
        echo "Using single GPU"
        if [ -f "$OUTPUT_DIR/best_model.pth" ]; then
            echo "Found best_model.pth - resuming training from checkpoint"
            RESUME_ARG="--resume best_model.pth"
        else
            RESUME_ARG=""
            echo "No checkpoint found - starting training from scratch"
        fi
        python user_identification/train_labram_arcface.py \
            --hdf5_dir /home/mojtabam/scratch/processed_eeg_data_user_identification \
            --segment_length 4s \
            --pretrained_path LaBraM/checkpoints/labram-base.pth \
            --epochs 200 \
            --lr 5e-4 \
            --batch_size 192 \
            --num_workers 8 \
            --layer_decay 0.9 \
            --weight_decay 0.05 \
            --warmup_epochs 10 \
            --clip_grad 0.2 \
            --arcface_margin 0.5 \
            --arcface_scale 256.0 \
            --output_dir "$OUTPUT_DIR" \
            --save_ckpt_freq 5 \
            --seed 42 \
            $RESUME_ARG
    fi
fi

echo ""
echo "=========================================="
echo "Training completed"
echo "=========================================="
