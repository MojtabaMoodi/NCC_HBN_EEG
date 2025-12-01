#!/bin/bash
#SBATCH --account=aip-aghodsib
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=labram_multi_output_cross_task_cv_active_to_passive_gender_stratified_1s
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

# Activate labram conda environment
conda activate labram

# Change to LaBraM directory (runs directory is in LaBraM/runs)
cd "$(dirname "$0")/.."

# Run the training script
echo "=========================================="
echo "Starting training: run_multi_output_cross_task_cv_active_to_passive_gender_stratified_1s.sh"
echo "=========================================="
echo ""

bash runs/run_multi_output_cross_task_cv_active_to_passive_gender_stratified_1s.sh

echo ""
echo "=========================================="
echo "Job completed: run_multi_output_cross_task_cv_active_to_passive_gender_stratified_1s.sh"
echo "End Time: $(date)"
echo "=========================================="
