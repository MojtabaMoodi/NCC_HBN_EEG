# LaBraM confusion matrices (age / gender)

Load **saved finetuned checkpoints** (no retraining) and compute confusion matrices on the
**test** split with the same participant aggregation as LaBraM eval
(`--aggregate_by_participant majority_vote`).

Checkpoints from training (under `final_logs/LaBraM/labram_{age,gender}_{1s,2s,4s}/output/`):

```
{age,gender}_labram_{1s,2s,4s}_best.pth   # primary name on disk
checkpoint-best.pth                       # alternate LaBraM save name
```

Implementation:

| File | Role |
|------|------|
| `../eval_confusion_matrix.py` | CLI (checkpoint + test loader + save JSON) |
| `../classification_eval_utils.py` | Shared inference, aggregation, confusion-matrix helpers |

## Output JSON

Each run writes **`LaBraM/confusion_matrix/results/{age\|gender}_{1s\|2s\|4s}_confusion_matrix.json`**
with:

- `test.confusion_matrices.segment` — one row/column per class on **segments**
- `test.confusion_matrices.participant_mean_prob` — participant-level (mean probability)
- `test.confusion_matrices.participant_majority_vote` — participant-level (**reported metric**)

Class labels: gender `{0, 1}`; age `{0, 1, 2}`.

## Model flags

Architecture flags (`--model`, `--qkv_bias`, …) must **match the trained checkpoint**.
Values below match `final_logs/LaBraM/labram_*` training logs.

## Run all age + gender (interactive)

From the **EEG project root**, with `eeg_env` active and checkpoints present:

```bash
export DATA_PATH=/home/mojtabam/scratch/processed_eeg_data_hdf5
export DEVICE=cuda
bash LaBraM/confusion_matrix/run_all_confusion_matrices.sh
```

Required env for the batch script:

| Variable | Example |
|----------|---------|
| `DATA_PATH` | `/home/mojtabam/scratch/processed_eeg_data_hdf5` |
| `DEVICE` | `cuda` |
| `PROJECT_ROOT` | path to EEG repo if not inferred from script location |

## SLURM (cluster)

Submit one GPU job that runs all six evals sequentially (age/gender × 1s/2s/4s):

```bash
bash LaBraM/confusion_matrix/submit_confusion_matrices.sh
```

Optional overrides before submit:

| Variable | Default |
|----------|---------|
| `EEG_ROOT` | repo root (inferred from script path) |
| `DATA_PATH` | `/home/mojtabam/scratch/processed_eeg_data_hdf5` |
| `DEVICE` | `cuda` |
| `SLURM_ACCOUNT` | `aip-aghodsib` |
| `SLURM_PARTITION` | `gpubase_l40s_b4` |
| `SLURM_TIME` | `6:00:00` |
| `SLURM_GRES` | `gpu:l40s:1` |
| `SLURM_MEM` | `32G` |

Logs and outputs:

| Path | Contents |
|------|----------|
| `LaBraM/confusion_matrix/logs/slurm-<jobid>.out` | SLURM stdout |
| `LaBraM/confusion_matrix/logs/slurm-<jobid>.err` | SLURM stderr |
| `LaBraM/confusion_matrix/logs/run_all_confusion_matrices_slurm-<jobid>.log` | Full tee of batch script |
| `LaBraM/confusion_matrix/results/*.json` | Confusion matrix JSON (one per task × segment) |

SLURM entry points:

| File | Role |
|------|------|
| `submit_confusion_matrices.sh` | `sbatch` wrapper |
| `../runs/slurm_jobs/run_confusion_matrices.slurm` | Job body (conda `eeg_env`, GPU, runs batch script) |

Check job status:

```bash
squeue -u "$USER" -n labram_confusion_matrices
tail -f LaBraM/confusion_matrix/logs/slurm-<jobid>.out
```

## Individual command template

Use `--num_workers 0` for **1s**, `4` for **2s** and **4s** (matches LaBraM eval).

```bash
python LaBraM/eval_confusion_matrix.py \
  --checkpoint final_logs/LaBraM/labram_{age|gender}_{1s|2s|4s}/output/{age|gender}_labram_{1s|2s|4s}_best.pth \
  --dataset {age_baseline|gender_baseline} \
  --segment_length {1s|2s|4s} \
  --data_path /home/mojtabam/scratch/processed_eeg_data_hdf5 \
  --task_type both \
  --aggregate_by_participant majority_vote \
  --batch_size 64 \
  --num_workers {0|4} \
  --device cuda \
  --model labram_base_patch200_200 \
  --qkv_bias true \
  --rel_pos_bias true \
  --abs_pos_emb true \
  --use_mean_pooling true \
  --layer_scale_init_value 0.1 \
  --init_scale 0.001 \
  --drop 0.0 \
  --attn_drop_rate 0.0 \
  --drop_path 0.1 \
  --output_json LaBraM/confusion_matrix/results/{age|gender}_{1s|2s|4s}_confusion_matrix.json
```

Example — age, 4s:

```bash
python LaBraM/eval_confusion_matrix.py \
  --checkpoint final_logs/LaBraM/labram_age_4s/output/age_labram_4s_best.pth \
  --dataset age_baseline \
  --segment_length 4s \
  --data_path /home/mojtabam/scratch/processed_eeg_data_hdf5 \
  --task_type both \
  --aggregate_by_participant majority_vote \
  --batch_size 64 \
  --num_workers 4 \
  --device cuda \
  --model labram_base_patch200_200 \
  --qkv_bias true \
  --rel_pos_bias true \
  --abs_pos_emb true \
  --use_mean_pooling true \
  --layer_scale_init_value 0.1 \
  --init_scale 0.001 \
  --drop 0.0 \
  --attn_drop_rate 0.0 \
  --drop_path 0.1 \
  --output_json LaBraM/confusion_matrix/results/age_4s_confusion_matrix.json
```

Example — gender, 1s (`--num_workers 0`):

```bash
python LaBraM/eval_confusion_matrix.py \
  --checkpoint final_logs/LaBraM/labram_gender_1s/output/gender_labram_1s_best.pth \
  --dataset gender_baseline \
  --segment_length 1s \
  --data_path /home/mojtabam/scratch/processed_eeg_data_hdf5 \
  --task_type both \
  --aggregate_by_participant majority_vote \
  --batch_size 64 \
  --num_workers 0 \
  --device cuda \
  --model labram_base_patch200_200 \
  --qkv_bias true \
  --rel_pos_bias true \
  --abs_pos_emb true \
  --use_mean_pooling true \
  --layer_scale_init_value 0.1 \
  --init_scale 0.001 \
  --drop 0.0 \
  --attn_drop_rate 0.0 \
  --drop_path 0.1 \
  --output_json LaBraM/confusion_matrix/results/gender_1s_confusion_matrix.json
```
