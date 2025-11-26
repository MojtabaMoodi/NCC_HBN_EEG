#!/bin/bash
#SBATCH --account=aip-aghodsib
#SBATCH --partition=gpubase_l40s_b5
#SBATCH --time=4-10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:4
#SBATCH --job-name=cnn_1s
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

# Change to CNN directory
cd /home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/CNN

# Create logs directory if it doesn't exist
mkdir -p ./logs/slurm

# Run the training script
echo "=========================================="
echo "Starting CNN Training: 1s mode"
echo "=========================================="
echo ""
echo "Command: python main.py --mode 1s --epochs 50 --learning_rate 0.0001 --num_gpus 4 --results_dir results_1s --reports_dir report_1s"
echo ""

python main.py --mode 1s --epochs 50 --learning_rate 0.0001 --num_gpus 4 --results_dir results_1s --reports_dir report_1s

echo ""
echo "=========================================="
echo "Job completed: CNN 1s training"
echo "End Time: $(date)"
echo "=========================================="
