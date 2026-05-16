#!/bin/bash
set -euo pipefail

cd /home/mojtabam/projects/aip-aghodsib/mojtabam/EEG

for segment in 1s 2s 4s; do
  for fold in 0 1 2 3 4; do
    sbatch --export=ALL,SEGMENT_LENGTH="${segment}",FOLD_IDX="${fold}" \
      --job-name="labram_frozen_${segment}_f${fold}" \
      user_identification/jobs/slurm_train_labram_frozen_fold.slurm
  done
done
