#!/bin/bash
set -euo pipefail

cd /project/6101794/mojtabam/EEG
mkdir -p final_user_identification_triplet

sbatch user_identification/jobs/slurm_triplet_4s_open_set_eval.slurm
