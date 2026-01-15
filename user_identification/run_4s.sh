#!/bin/bash
# Direct script to run user identification training for 4s segments
# This can be run interactively or submitted as a SLURM job

# Initialize conda
source ~/.bashrc || true
eval "$(conda shell.bash hook)" || true

# Activate eeg_env conda environment
conda activate eeg_env

# Change to project root directory
cd /home/mojtabam/projects/aip-aghodsib/mojtabam/EEG

# Run the user identification training script
echo "=========================================="
echo "Starting User Identification Training: 4s mode"
echo "=========================================="
echo ""

python user_identification/main.py \
    --mode 4s \
    --model labram \
    --epochs 50 \
    --learning_rate 0.0001 \
    --num_gpus 4 \
    --results_dir results_user_id_4s_labram \
    --reports_dir reports_user_id_4s_labram \
    --hdf5_dir /home/mojtabam/scratch/processed_eeg_data_user_identification \
    --labram_checkpoint LaBraM/checkpoints/labram-base.pth

echo ""
echo "=========================================="
echo "Training completed"
echo "=========================================="
