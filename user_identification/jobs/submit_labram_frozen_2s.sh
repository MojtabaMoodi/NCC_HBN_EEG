#!/bin/bash
set -euo pipefail

cd /home/mojtabam/projects/aip-aghodsib/mojtabam/EEG

SCRIPT="user_identification/jobs/slurm_train_labram_frozen_fold.slurm"

for fold in 0 1 2 3 4; do
  sbatch --export=ALL,SEGMENT_LENGTH=2s,FOLD_IDX="${fold}" \
    --job-name="labram_frozen_2s_f${fold}" \
    "${SCRIPT}"
done
