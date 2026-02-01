# User Identification Module

This module provides tools for **user identification** using **LaBraM with ArcFace**: training, evaluation, and **open-set** analysis (known vs unknown participants). It also includes an alternative entry point for CNN and LaBraM experiments via `main.py`.

## Structure

| File | Description |
|------|-------------|
| `train_labram_arcface.py` | LaBraM + ArcFace training (primary training script) |
| `run_4s.sh` | Bash wrapper for 4s training: single-GPU or multi-GPU (interactive/SLURM) |
| `run_4s.slurm` | SLURM batch script for cluster runs |
| `labram_model.py` | LaBraM wrapper for user identification |
| `labram_trainer.py` | Training loop and ArcFace integration |
| `analyze_confidence.py` | Confidence and OOD analysis: test/unknown splits, open-set threshold tuning |
| `run_analysis_test_unknown.sh` | Wrapper to run analysis on test + unknown with `--split both` |
| `main.py` | Alternative: run CNN and/or LaBraM experiments via CNN experiment pipeline |
| `validate.py` | Validation script to verify user-identification data setup |

## Training (LaBraM + ArcFace)

The primary training flow uses LaBraM with ArcFace loss. Data must be preprocessed for user identification (see `data_processing`); HDF5 directory must contain `participant_id_to_class_idx.json`.

### Run training (recommended: use the wrapper)

```bash
# Default: single GPU, output to ./results_user_id_4s_labram_arcface
./user_identification/run_4s.sh

# Save to a custom directory
OUTPUT_DIR=/path/to/my_run ./user_identification/run_4s.sh

# Single GPU (explicit)
NUM_GPUS=1 ./user_identification/run_4s.sh

# Multi-GPU (e.g. 4 GPUs)
NUM_GPUS=4 ./user_identification/run_4s.sh
# Or rely on auto-detect from CUDA_VISIBLE_DEVICES:
CUDA_VISIBLE_DEVICES=0,1,2,3 ./user_identification/run_4s.sh
```

**Resume:** If `OUTPUT_DIR/best_model.pth` exists, the script resumes from it automatically.

**Environment variables (run_4s.sh):**

| Variable | Default | Description |
|----------|--------|-------------|
| `OUTPUT_DIR` | `./results_user_id_4s_labram_arcface` | Where checkpoints and logs are saved |
| `NUM_GPUS` | (auto) | `1` = single-GPU; `2+` = multi-GPU with `torchrun`. Unset = auto-detect from `CUDA_VISIBLE_DEVICES` |

**Single-GPU vs multi-GPU:**  
- Single-GPU: no distributed init; all distributed env vars are unset.  
- Multi-GPU: `torchrun --nproc_per_node=N`; requires `--distributed` (script adds it when `NUM_GPUS > 1`).  
- SLURM batch jobs use `srun` and set `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`, etc.

### Run training script directly

```bash
python user_identification/train_labram_arcface.py \
  --hdf5_dir /path/to/processed_eeg_data_user_identification \
  --segment_length 4s \
  --pretrained_path LaBraM/checkpoints/labram-base.pth \
  --epochs 200 \
  --lr 5e-4 \
  --batch_size 192 \
  --num_workers 8 \
  --output_dir /path/to/output \
  --save_ckpt_freq 5 \
  --seed 42
# Multi-GPU: add --distributed and run with torchrun
```

**Key arguments:** `--hdf5_dir` (required), `--segment_length` (1s/2s/4s), `--output_dir`, `--resume` (e.g. `best_model.pth`), `--distributed` (only for multi-GPU). ArcFace: `--arcface_margin`, `--arcface_scale`. Early stopping: `--early_stopping_patience`, `--early_stopping_min_delta`.

## Confidence and Open-Set Analysis

After training, use `analyze_confidence.py` to evaluate on **test** (known users) and **unknown** (held-out users), and to tune an **open-set** threshold so the system can both identify known users and reject unknown users.

### Run analysis (test + unknown)

```bash
# Use wrapper (set CHECKPOINT; optional: OUTPUT_DIR, HDF5_DIR, SEGMENT_LENGTH)
CHECKPOINT=/path/to/best_model.pth ./user_identification/run_analysis_test_unknown.sh
CHECKPOINT=/path/to/best_model.pth OUTPUT_DIR=./my_analysis ./user_identification/run_analysis_test_unknown.sh

# Or call the script directly
python user_identification/analyze_confidence.py \
  --checkpoint /path/to/best_model.pth \
  --hdf5_dir /path/to/processed_eeg_data_user_identification \
  --segment_length 4s \
  --split both \
  --output_dir ./confidence_analysis_test_unknown
```

**Analysis script arguments:**

| Argument | Default | Description |
|----------|--------|-------------|
| `--checkpoint` | (required) | Path to trained checkpoint (e.g. `best_model.pth`) |
| `--hdf5_dir` | (see script) | HDF5 directory (user-identification preprocessed) |
| `--segment_length` | `4s` | `1s`, `2s`, or `4s` (must match training) |
| `--split` | `test` | `train`, `val`, `test`, `unknown`, or `both` |
| `--output_dir` | `./confidence_analysis` | Base directory for all outputs |
| `--batch_size` | 128 | Inference batch size |
| `--num_workers` | 4 | DataLoader workers |
| `--device` | `cuda` | `cuda` or `cpu` |

**`--split both`** runs on test and unknown; when the checkpoint uses ArcFace, it also computes feature-space OOD (distance to known-user centroids), **known-user score** (1/(1+OOD distance)), and **open-set threshold** tuning.

### Analysis outputs (when `--split both`)

Outputs are written under `output_dir`:

| Path | Description |
|------|-------------|
| `test/` | Test split (known users): correct vs incorrect confidence, per-user stats, plots |
| `test/user_confidence_statistics.json` | Per-user mean/min/max confidence (correct and incorrect) |
| `test/confidence_distributions_test.png` | Confidence histograms (correct vs incorrect) |
| `unknown/` | Unknown (held-out) users: no identity labels |
| `unknown/unknown_confidence_summary.json` | Aggregate stats: **known_user_score**, OOD distance, softmax confidence |
| `unknown/confidence_distribution_unknown.png` | Softmax confidence distribution (closed-world; often high for unknown) |
| `unknown/known_user_score_unknown.png` | Known-user score distribution (low = more unknown; use this to flag unknowns) |
| `open_set_summary.json` | Best threshold on known-user score and **open-set accuracy** (correct known + correct unknown rejections) |
| `ood_centroids.npz` | Cached known-user centroids (from training set) for OOD distance |

**Metrics:**

- **Known-user score** = 1/(1 + OOD distance). Low for unknown users, high for known users. Use this (not softmax) to decide “known vs unknown”.
- **Open-set:** Reject as “unknown” when known_user_score < threshold. Threshold is tuned to maximize open-set accuracy on test+unknown.

### Example results (from current runs)

With a LaBraM+ArcFace checkpoint and `--split both`:

- **Open-set accuracy** (best threshold on known-user score): ~94.5% (e.g. threshold 0.85).
- **Unknown users:** known_user_score mean ~0.59; softmax confidence mean ~0.96 (closed-world, so high for unknowns is expected).
- **Test (known users):** high closed-set accuracy; per-user confidence stats in `test/user_confidence_statistics.json`.

## Validating Setup

```bash
python user_identification/validate.py \
  --hdf5_dir /path/to/processed_eeg_data_user_identification
```

## Alternative: CNN / LaBraM experiments (main.py)

For running CNN and/or LaBraM experiments through the CNN experiment pipeline (different from the ArcFace training above):

```bash
python user_identification/main.py \
  --mode 4s \
  --model both \
  --labram_checkpoint /path/to/pretrained_labram.pth \
  --epochs 50 \
  --learning_rate 0.0001 \
  --num_gpus 4 \
  --hdf5_dir /path/to/processed_eeg_data_user_identification
```

Arguments: `--mode` (1s/2s/4s), `--model` (cnn/labram/both), `--labram_checkpoint`, `--epochs`, `--learning_rate`, `--num_gpus`, `--hdf5_dir`, `--results_dir`, `--reports_dir`.

## Notes

- LaBraM training in `run_4s.sh` uses a pre-trained LaBraM backbone (`--pretrained_path`); the checkpoint saves the full model and ArcFace state for analysis.
- Unknown users require preprocessing with “consider unknown” (held-out participants); otherwise the unknown split is empty.
- For saliency/interpretability on **CNN** age/gender models, use the `saliency_analysis` module; user identification (LaBraM) is not supported there.
