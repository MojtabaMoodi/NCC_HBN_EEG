#!/bin/bash
#SBATCH -A aip-aghodsib
#SBATCH -p gpubase_l40s_b4
#SBATCH -N 1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=triplet_4s_eval
#SBATCH --output=final_user_identification_triplet/slurm-%x-%j.out
#SBATCH --error=final_user_identification_triplet/slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/project/6101794/mojtabam/EEG"
FINAL_ROOT="${PROJECT_ROOT}/final_user_identification_triplet"
HDF5_ROOT_4S="${HOME}/scratch/4s"
LOG_DIR="${FINAL_ROOT}"
PRED_DIR="${FINAL_ROOT}/test_predictions"
EVAL_LOG="${FINAL_ROOT}/eval_run_4s.log"

cd "${PROJECT_ROOT}"
mkdir -p "${FINAL_ROOT}" "${PRED_DIR}"

export PATH="/home/mojtabam/miniconda3/bin:${PATH}"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start: $(date)"
echo "Final root: ${FINAL_ROOT}"
echo "HDF5 root (4s): ${HDF5_ROOT_4S}"

conda run -n eeg_env --no-capture-output python user_identification/run_all_folds_active_passive_test_unknown_eval.py \
  --final_root "${FINAL_ROOT}" \
  --hdf5_root_4s "${HDF5_ROOT_4S}" \
  --n_folds 5 \
  --device cuda \
  --log_dir "${LOG_DIR}" \
  --save_predictions_dir "${PRED_DIR}" \
  2>&1 | tee "${EVAL_LOG}"

echo "End: $(date)"
echo "Aggregate summary: ${FINAL_ROOT}/active_passive_test_unknown_eval_summary.json"
echo "Segment summary log: ${LOG_DIR}/eval_run_4s.log"
