#!/bin/bash
set -euo pipefail

cd /project/6101794/mojtabam/EEG

SCRIPT="user_identification/jobs/slurm_train_triplet_4s_fold.slurm"

for fold in 1 2 3 4; do
  sbatch \
    --export=ALL,FOLD_IDX="${fold}" \
    --job-name="triplet_4s_f${fold}" \
    --output="final_user_identification_triplet/4s/fold_${fold}/slurm-%x-%j.out" \
    --error="final_user_identification_triplet/4s/fold_${fold}/slurm-%x-%j.err" \
    "${SCRIPT}"
done
