#!/bin/bash
#SBATCH --account=aip-aghodsib
#SBATCH --partition=gpubase_l40s_b5
#SBATCH --time=4-10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:4
#SBATCH --job-name=user_id_4s
#SBATCH --output=./logs/slurm/%x-%j.out
#SBATCH --error=./logs/slurm/%x-%j.err

# Job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="
echo ""

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo ""

# Print system information
echo "System Information:"
echo "Hostname: $(hostname)"
echo "CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo ""

# Load required modules
module load python/3.10
module load cuda/12.6
module load cudnn/8.9.7

# Initialize conda
source ~/.bashrc || true
eval "$(conda shell.bash hook)" || true

# Activate eeg_env conda environment
conda activate eeg_env

# Change to project root directory
cd /home/mojtabam/projects/aip-aghodsib/mojtabam/EEG

# Create logs directory if it doesn't exist
mkdir -p ./logs/slurm

# Run the LaBraM user identification training script with ArcFace
echo "=========================================="
echo "Starting LaBraM User Identification Training: 4s mode with ArcFace"
echo "=========================================="
echo ""
echo "Command: python user_identification/train_labram_arcface.py --hdf5_dir /home/mojtabam/scratch/processed_eeg_data_user_identification --segment_length 4s --pretrained_path LaBraM/checkpoints/labram-base.pth --epochs 200 --lr 5e-4 --batch_size 192 --output_dir ./results_user_id_4s_labram_arcface"
echo ""

# Set up distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Calculate WORLD_SIZE: if SLURM_NTASKS is not set, use ntasks-per-node * nnodes
if [ -z "$SLURM_NTASKS" ] || [ "$SLURM_NTASKS" = "" ]; then
    # Calculate from ntasks-per-node and number of nodes
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

# Debug: Print environment variables
echo "Distributed Training Environment:"
echo "  MASTER_ADDR=$MASTER_ADDR"
echo "  MASTER_PORT=$MASTER_PORT"
echo "  WORLD_SIZE=$WORLD_SIZE"
echo "  RANK=$RANK"
echo "  LOCAL_RANK=$LOCAL_RANK"
echo "  SLURM_NTASKS=$SLURM_NTASKS"
echo "  SLURM_NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE"
echo "  SLURM_NNODES=$SLURM_NNODES"
echo ""

# Run with distributed training
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
    --output_dir ./results_user_id_4s_labram_arcface \
    --save_ckpt_freq 5 \
    --seed 42 \
    --distributed

echo ""
echo "=========================================="
echo "Job completed: User Identification 4s training"
echo "End Time: $(date)"
echo "=========================================="
