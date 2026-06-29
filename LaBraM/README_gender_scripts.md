# LaBraM Fine-Tuning Scripts (Project Extensions)

This document describes how to run **project-specific** LaBraM fine-tuning on gender, age, combined, and multi-output tasks. The upstream paper README is in [README.md](README.md).

## Layout

| Path | Purpose |
|------|---------|
| `run_class_finetuning.py` | Main training and evaluation entry point |
| `dataset_config.py` | Dataset names, class counts, default HDF5 paths |
| `runs/run_<experiment>_<segment>.sh` | SLURM/HPC-ready launchers (conda `eeg_env`, single GPU) |
| `run_gender_baseline.sh` | Legacy single-script gender baseline (prefer `runs/`) |
| `run_eval_majority_vote.sh` | Eval-only with participant-level aggregation |
| `report_generator.py` | Summaries from `outputs/*/log.txt` |
| [REPORT_GENERATION_PROCESS.md](REPORT_GENERATION_PROCESS.md) | Report pipeline details |
| `freeze_backbone.py` | Linear-probe helper (`apply_finetune_freeze`) |
| `runs/run_labram_frozen.sh` | Frozen backbone for age / gender / regression |

## Frozen backbone (linear probe)

Train only the task head while keeping pretrained LaBraM weights fixed:

```bash
cd LaBraM
TASK=age SEGMENT=1s bash runs/run_labram_frozen.sh \
  --output_dir ../final_logs_frozen/LaBraM/labram_age_frozen_1s/output \
  --log_dir ../final_logs_frozen/LaBraM/labram_age_frozen_1s/logs
```

Or pass flags directly to `run_class_finetuning.py`:

```bash
python run_class_finetuning.py ... --freeze_backbone --unfreeze_last_n_blocks 0
```

- `--unfreeze_last_n_blocks N` also unfreezes the last `N` transformer blocks (default `0` = head only).
- Logs record `freeze_backbone`, `n_parameters`, `n_parameters_total`, and `task_type_metrics` (active/passive).
- Aggregated reports: see [final_logs_frozen/note.txt](../final_logs_frozen/note.txt).

## Data

HDF5 multipart files: `eeg_data_{train,val,test}_{1s|2s|4s}_part*.h5`.

Default directory (override with `--data_path` or env `EEG_HDF5_DIR`):

- `~/scratch/processed_eeg_data_hdf5` (see `dataset_config.DATA_PATHS`)

Preprocessing: [data_processing/README.md](../data_processing/README.md).

## Datasets (`--dataset`)

Names match the CNN experiment framework (see `dataset_config.VALID_DATASET_NAMES`). Examples:

| Task | Example `--dataset` | `--nb_classes` |
|------|---------------------|----------------|
| Gender (binary, BCE) | `gender_baseline` | `1` |
| Age (3-class) | `age_baseline` or `age_classification` | `3` |
| Age regression | `age_regression_baseline` | `1` |
| Combined gender+age | `combined_baseline` | `6` |
| Multi-output heads | `multi_output_baseline` | `2` |

Cross-validation and cross-task variants use the same naming as CNN (e.g. `gender_cv_gender_stratified`, `age_cross_task_active_to_passive`). CV runs create `fold_0/`, `fold_1/`, … under `--output_dir`.

## Run scripts (`runs/`)

Generated launchers live under `LaBraM/runs/`. Run from **project root** or **LaBraM/**:

```bash
# Gender baseline, 1s (writes ./outputs/gender_baseline_1s by default)
bash LaBraM/runs/run_gender_baseline_1s.sh

# Custom output directory
bash LaBraM/runs/run_gender_baseline_4s.sh --output_dir /path/to/my_run

# Age baseline, 2s
bash LaBraM/runs/run_age_baseline_2s.sh
```

**Segment coverage:**

- **1s and 4s:** full suite (baseline, CV, cross-task, cross-task CV) for gender, age, combined, multi-output
- **2s:** `run_gender_baseline_2s.sh`, `run_age_baseline_2s.sh` only

**Typical hyperparameters in `runs/` scripts** (see script header for exact values):

- Gender/age baseline 1s: `epochs=50`, `batch_size=1024`, `lr=5e-4`, EMA enabled
- Gender/age baseline 4s: `batch_size=256` (smaller than 1s due to longer windows)
- Pretrained weights: `LaBraM/checkpoints/labram-base.pth`

Scripts accept `--output_dir` and `--log_dir`; remaining CLI flags are forwarded to `run_class_finetuning.py`.

## Direct Python usage

From `LaBraM/` (or project root with `python LaBraM/run_class_finetuning.py`):

```bash
cd LaBraM
python run_class_finetuning.py \
  --model labram_base_patch200_200 \
  --finetune checkpoints/labram-base.pth \
  --dataset gender_baseline \
  --nb_classes 1 \
  --segment_length 1s \
  --data_path ~/scratch/processed_eeg_data_hdf5 \
  --output_dir ./outputs/gender_baseline_1s \
  --epochs 50 \
  --batch_size 1024 \
  --abs_pos_emb --qkv_bias --use_mean_pooling \
  --auto_resume --save_ckpt
```

**Age regression:**

```bash
python run_class_finetuning.py \
  --dataset age_regression_baseline \
  --nb_classes 1 \
  --segment_length 4s \
  --data_path ~/scratch/processed_eeg_data_hdf5 \
  --output_dir ./outputs/age_regression_baseline_4s \
  ... # same model/pretrained flags as above
```

**Useful flags:** `--class_weight_power` (gender imbalance), `--early_stopping_patience` / `--early_stopping_min_delta` (val accuracy for classification, val MAE for regression), `--eval` (evaluation only).

## Evaluation with participant aggregation

Re-run test metrics without training (segment-level, participant mean-probability, participant majority vote):

```bash
# Wrapper (from LaBraM/ or project root)
./LaBraM/run_eval_majority_vote.sh \
  --output_dir ./outputs/gender_baseline_2s \
  --dataset gender_baseline --nb_classes 1 --segment_length 2s

# Or directly
python run_class_finetuning.py --eval \
  --aggregate_by_participant majority_vote \
  --output_dir ./outputs/gender_baseline_2s \
  --dataset gender_baseline --nb_classes 1 \
  --segment_length 2s --auto_resume \
  --data_path ~/scratch/processed_eeg_data_hdf5
```

For age classification use `--dataset age_classification --nb_classes 3`.

## Random classifier baseline

Segment-level random predictions on the test split, with the same participant
aggregation as model eval. Results and commands live under
**`LaBraM/random_classifier/`** (see that README).

Quick run (all age/gender × 1s/2s/4s, `train_segment_frequency`):

```bash
bash LaBraM/random_classifier/run_train_segment_frequency_eval.sh
```

Outputs: `LaBraM/random_classifier/results/{age,gender}_{1s,2s,4s}_train_segment_frequency.json`

## Confusion matrices (finetuned LaBraM)

Load saved finetuned weights (e.g. `{task}_labram_{segment}_best.pth`; no retraining) and compute segment- and
participant-level confusion matrices on the test split. See **`LaBraM/confusion_matrix/`**.

Interactive batch (requires `DATA_PATH` and `DEVICE`):

```bash
export DATA_PATH=/home/mojtabam/scratch/processed_eeg_data_hdf5
export DEVICE=cuda
bash LaBraM/confusion_matrix/run_all_confusion_matrices.sh
```

SLURM (all age/gender × 1s/2s/4s):

```bash
bash LaBraM/confusion_matrix/submit_confusion_matrices.sh
```

Outputs: `LaBraM/confusion_matrix/results/{age,gender}_{1s,2s,4s}_confusion_matrix.json`
(primary table: `test.confusion_matrices.participant_majority_vote`).

## Outputs

```
outputs/
├── gender_baseline_1s/
│   ├── checkpoint-best.pth
│   ├── checkpoint.pth
│   ├── log.txt              # JSON lines, one object per epoch
│   └── ...
├── gender_cv_gender_stratified_4s/
│   ├── fold_0/log.txt
│   ├── fold_1/log.txt
│   └── ...
```

TensorBoard logs go to `--log_dir` (default `./logs/<dataset>_<segment>` in run scripts).

## Reports

```bash
cd LaBraM
python report_generator.py 1s    # or 2s, 4s, or no arg for all
```

See [REPORT_GENERATION_PROCESS.md](REPORT_GENERATION_PROCESS.md). Reports are written under `reports_{segment}/` by default.

## Environment

Use conda env **`eeg_env`** (see run scripts). Official LaBraM env setup is in [README.md](README.md).

## Monitoring

```bash
tensorboard --logdir=LaBraM/logs/gender_baseline_1s
tail -f LaBraM/outputs/gender_baseline_1s/log.txt
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA OOM | Lower `--batch_size` in the run script or CLI |
| HDF5 not found | Set `--data_path` or `EEG_HDF5_DIR` |
| Wrong task/classes | Match `--dataset` and `--nb_classes` to training |
| Resume | `--auto_resume` or `--resume path/to/checkpoint.pth` |
