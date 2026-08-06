#!/bin/bash
set -euo pipefail

cd /project/6101794/mojtabam/EEG

SCRIPT="user_identification/jobs/slurm_train_triplet_2s_fold_resume.slurm"

for fold in 0 1 2 3 4; do
  RUN_DIR="final_user_identification_triplet/2s/fold_${fold}"
  mkdir -p "${RUN_DIR}"

  latest_ckpt=$(ls -t "${RUN_DIR}"/checkpoint_epoch_*.pth 2>/dev/null | head -1 || true)
  if [ -n "${latest_ckpt}" ]; then
    RESUME_CKPT=$(basename "${latest_ckpt}")
  else
    RESUME_CKPT="best_model.pth"
  fi

  echo "Submitting fold ${fold} resume from ${RESUME_CKPT}"
  sbatch \
    --export=ALL,FOLD_IDX="${fold}",RESUME_CKPT="${RESUME_CKPT}" \
    --job-name="triplet_2s_f${fold}_resume" \
    --output="${RUN_DIR}/slurm-%x-%j.out" \
    --error="${RUN_DIR}/slurm-%x-%j.err" \
    "${SCRIPT}"
done
