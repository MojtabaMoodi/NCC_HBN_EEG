#!/bin/bash
# SLURM script to run saliency computation on GPU
# Usage: sbatch run_saliency_slurm.sh
# Or: salloc --gres=gpu:4 --time=24:00:00 --mem=64G bash run_saliency_slurm.sh

#SBATCH --job-name=saliency_computation
#SBATCH --output=saliency_%j.out
#SBATCH --error=saliency_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:4  # Request 4 GPUs (adjust as needed)
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1

# Activate conda environment if needed
# conda activate eeg_env

# Set working directory
cd /home/mojtabam/projects/aip-aghodsib/mojtabam/EEG

# Print GPU information
echo "=========================================="
echo "GPU Information"
echo "=========================================="
nvidia-smi
echo ""
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
echo "Number of GPUs requested: 4"
echo "=========================================="
echo ""

# Run saliency computation
# Options:
#   --device cuda: Use CUDA (auto-detects GPUs)
#   --num_gpus 4: Use 4 GPUs (will use DataParallel)
#   --max_samples: Limit samples per model (None = all)
#   --method: integrated_gradients, vanilla_gradients, or both (to compare)
#   --batch_size: Batch size for data loading

python3 saliency_analysis/main.py \
    --batch \
    --hdf5_dir data_processing/processed_eeg_data_hdf5_no_compression \
    --device cuda \
    --num_gpus 4 \
    --method both \
    --batch_size 32 \
    --output_dir saliency_results \
    --max_samples 1000  # Adjust or remove to process all samples

echo ""
echo "=========================================="
echo "Saliency computation completed!"
echo "=========================================="
