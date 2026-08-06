#!/bin/bash
set -euo pipefail

cd /project/6101794/mojtabam/EEG

SCRIPT="user_identification/jobs/slurm_train_triplet_1s_fold.slurm"

for fold in 0 1 2 3 4; do
  mkdir -p "final_user_identification_triplet/1s/fold_${fold}"
  sbatch \
    --export=ALL,FOLD_IDX="${fold}" \
    --job-name="triplet_1s_f${fold}" \
    --output="final_user_identification_triplet/1s/fold_${fold}/slurm-%x-%j.out" \
    --error="final_user_identification_triplet/1s/fold_${fold}/slurm-%x-%j.err" \
    "${SCRIPT}"
done
