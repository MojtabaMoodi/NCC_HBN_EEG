#!/bin/bash
#SBATCH --account=aip-aghodsib
#SBATCH --partition=gpubase_l40s_b4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --job-name=labram_confusion_matrices
#SBATCH --output=/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/LaBraM/confusion_matrix/logs/slurm-%j.out
#SBATCH --error=/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/LaBraM/confusion_matrix/logs/slurm-%j.err

set -euo pipefail

EEG_ROOT="${EEG_ROOT:-/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG}"
DATA_PATH="${DATA_PATH:-/home/mojtabam/scratch/processed_eeg_data_hdf5}"
DEVICE="${DEVICE:-cuda}"

LOG_DIR="${EEG_ROOT}/LaBraM/confusion_matrix/logs"
RUN_LOG="${LOG_DIR}/run_all_confusion_matrices_slurm-${SLURM_JOB_ID:-local}.log"
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Job Name: ${SLURM_JOB_NAME:-labram_confusion_matrices}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "EEG_ROOT: ${EEG_ROOT}"
echo "DATA_PATH: ${DATA_PATH}"
echo "DEVICE: ${DEVICE}"
echo "Run log: ${RUN_LOG}"
echo "Start: $(date)"
echo "=========================================="

if command -v nvidia-smi &>/dev/null; then
  nvidia-smi
fi

set +u
if [ -f ~/.bashrc ]; then
  source ~/.bashrc 2>/dev/null || true
fi
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
  eval "$(conda shell.bash hook)" 2>/dev/null || true
fi
if command -v conda &> /dev/null; then
  if conda env list 2>/dev/null | grep -q "eeg_env"; then
    conda activate eeg_env
  else
    echo "ERROR: conda env eeg_env not found." >&2
    exit 1
  fi
fi

if command -v module &> /dev/null; then
  module load python/3.10 2>/dev/null || true
  module load cuda/12.6 2>/dev/null || true
  module load cudnn/8.9.7 2>/dev/null || \
  module load cudnn/9.0 2>/dev/null || \
  module load cudnn 2>/dev/null || true
fi
set -u

CONDA_PYTHON="$HOME/miniconda3/envs/eeg_env/bin/python"
if [ -f "$CONDA_PYTHON" ]; then
  export PATH="$HOME/miniconda3/envs/eeg_env/bin:$PATH"
fi

cd "${EEG_ROOT}"
export PYTHONPATH="${EEG_ROOT}/LaBraM:${EEG_ROOT}:${EEG_ROOT}/data_processing${PYTHONPATH:+:${PYTHONPATH}}"
chmod +x LaBraM/confusion_matrix/run_all_confusion_matrices.sh

export DATA_PATH
export DEVICE
export PROJECT_ROOT="${EEG_ROOT}"

DATA_PATH="${DATA_PATH}" DEVICE="${DEVICE}" PROJECT_ROOT="${EEG_ROOT}" \
  bash LaBraM/confusion_matrix/run_all_confusion_matrices.sh \
  2>&1 | tee "${RUN_LOG}"

echo "Completed: $(date)"
