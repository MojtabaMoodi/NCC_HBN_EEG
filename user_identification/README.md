# User Identification Module

This module provides tools for **user identification**: the model learns which **participant (user)** an EEG segment belongs to. It supports **LaBraM with ArcFace** (training, evaluation, open-set analysis with known vs unknown participants), **CNN or ResNet with ArcFace** (`cnn_resnet_arcface/train_user_identification_backbone.py`), and an alternative **CNN experiment pipeline** via `main.py` (gender/age and related tasks).

**Run from project root:** All scripts and wrappers assume the EEG project root as the current working directory (e.g. `cd /path/to/EEG` before running). The shell wrappers set this explicitly; when calling Python scripts directly, run them from the project root so imports and paths resolve correctly. **Usage steps:** [Prepare data → Validate → Train → (optional) Analyze](#usage-steps).

**Requirements:** Conda env `eeg_env`, LaBraM (with pretrained checkpoint under `LaBraM/checkpoints/`), and `data_processing` with HDF5 data prepared for user identification (including `participant_id_to_class_idx.json`). See `data_processing/README.md` for preprocessing.

**Data split:** Unlike gender/age (participant-level split), user identification splits **per participant at sample level** (e.g. 70% train, 15% val, 15% test per participant) so that all participants appear in all splits. Preprocessing: `data_processing/preprocess_user_identification.py`; use `--consider_unknown` to create an unknown split for open-set evaluation.

## Quick reference

| What | Where |
|------|--------|
| Training (LaBraM + ArcFace) | `run_4s.sh` or `train_labram_arcface.py` |
| **5-fold CV** (disjoint unknown, mean ± std) | `run_5fold_cv.py` (data: `preprocess_user_identification.py --n_folds 5`); optional `--train_script` for CNN/ResNet ArcFace; per-fold `fold_k/train_fold_k.log`; use `--aggregate_only` to aggregate results from folds trained on separate machines |
| **CNN / ResNet + ArcFace (user identification)** | `cnn_resnet_arcface/train_user_identification_backbone.py`; active/passive eval: `cnn_resnet_arcface/eval_active_passive_cnn_resnet.py` |
| Test active/passive accuracy (per fold or all folds) | `eval_active_passive_test.py` |
| Full eval runner (test active/passive + unknown/open-set, folds/segments, logs) | `run_all_folds_active_passive_test_unknown_eval.py` |
| Confidence / open-set analysis | `analyze_confidence.py` (e.g. `--split both`) or `run_analysis_test_unknown.sh` |
| CNN/LaBraM experiments (non-ArcFace) | `main.py` |
| Validate HDF5 setup | `validate.py` |
| Registration / open-world eval (unknown-only) | `registering_users/` — see [registering_users/README.md](registering_users/README.md) |
| Preprocessing | `data_processing/preprocess_user_identification.py` |

## Usage steps

Run all commands from the **project root** (e.g. `cd /path/to/EEG`).

### 1. Prepare data

User-identification data must be preprocessed so each participant’s samples are split 70% train / 15% val / 15% test, and (optionally) some participants are held out as “unknown” for open-set evaluation.

**Single run (one train/val/test + optional unknown):**
```bash
python data_processing/preprocess_user_identification.py \
  --data_root /path/to/preprocessed_new \
  --output_dir /path/to/processed_eeg_data_user_identification \
  --segment_length 4s \
  --random_seed 42
# Optional: add --consider_unknown --unknown_ratio 0.2 for open-set evaluation
```

**5-fold (disjoint unknown sets for mean ± std reporting):**
```bash
python data_processing/preprocess_user_identification.py \
  --data_root /path/to/preprocessed_new \
  --output_dir /path/to/user_id_5fold \
  --n_folds 5 \
  --segment_length 4s \
  --random_seed 42
```
This creates `output_dir/fold_0` … `output_dir/fold_4`; in each fold a different 20% of participants are unknown.

**5-fold in parallel (same partition as above):** use `--fold_index k` with the same `--data_root`, `--output_dir`, `--n_folds`, `--random_seed`, and `--segment_length` as a full run. Each job writes only `output_dir/fold_k/`. Example (SLURM array 0–4):

```bash
python data_processing/preprocess_user_identification.py \
  --data_root /path/to/preprocessed_new \
  --output_dir /path/to/user_id_5fold \
  --n_folds 5 \
  --fold_index $SLURM_ARRAY_TASK_ID \
  --segment_length 4s \
  --random_seed 42
```

**Check fold outputs (no HDF5 I/O):** from project root, `python data_processing/verify_user_id_fold_partition.py --output_dir /path/to/user_id_5fold --n_folds 5` — reads each `fold_k/participant_id_to_class_idx.json` and `unknown_participant_ids.json` and checks disjoint unknowns, same universe per fold, and full cover. Optional: add `--data_root ... --random_seed 42` to assert JSON unknown sets match a recomputed partition from source data.

### 2. Validate setup (optional but recommended)

```bash
python user_identification/validate.py --hdf5_dir /path/to/processed_eeg_data_user_identification
```
For 5-fold data, validate one fold: `--hdf5_dir /path/to/user_id_5fold/fold_0`.

### 3. Train

**Single run:**
```bash
# Easiest: use wrapper (output under ./results_user_id_4s_labram_arcface)
OUTPUT_DIR=/path/to/my_run ./user_identification/run_4s.sh

# Or call the training script directly (set paths explicitly)
python user_identification/train_labram_arcface.py \
  --hdf5_dir /path/to/processed_eeg_data_user_identification \
  --output_dir /path/to/my_run \
  --segment_length 4s --epochs 200 --batch_size 192 --seed 42
```

**5-fold CV (train one model per fold, then get mean ± std):**
```bash
python user_identification/run_5fold_cv.py \
  --hdf5_root /path/to/user_id_5fold \
  --output_root /path/to/results_5fold \
  --n_folds 5 \
  --segment_length 4s --epochs 200 --batch_size 192 --seed 42
```
Summary is written to `output_root/cv_summary.json`. Each fold’s training subprocess stdout/stderr is saved under `output_root/fold_k/train_fold_k.log`.

**Optional `train_script`:** by default `run_5fold_cv.py` invokes `train_labram_arcface.py`. For CNN or ResNet + ArcFace user identification, pass e.g. `--train_script user_identification/cnn_resnet_arcface/train_user_identification_backbone.py` plus that script’s flags (`--backbone`, etc.); unknown CLI tokens are forwarded to the training script, and `--hdf5_dir` / `--output_dir` are set per fold automatically. See [CNN and ResNet ArcFace user identification](#cnn-and-resnet-arcface-user-identification).

### 4. Optional: open-set analysis

If you used `--consider_unknown` (or 5-fold data), you can evaluate on test + unknown and tune an open-set threshold:

```bash
CHECKPOINT=/path/to/best_model.pth \
  HDF5_DIR=/path/to/processed_eeg_data_user_identification \
  ./user_identification/run_analysis_test_unknown.sh
# Or: python user_identification/analyze_confidence.py --checkpoint ... --hdf5_dir ... --split both --output_dir ...
```

See [Confidence and Open-Set Analysis](#confidence-and-open-set-analysis) for outputs and metrics.

### CNN and ResNet ArcFace user identification

Train from scratch with the **CNN experiment pipeline** (`CNN.experiment`) and ArcFace (`train_user_identification_backbone.py`). This is separate from `train_labram_arcface.py`; checkpoints are **not** interchangeable (raw EEG vs LaBraM scaling).

**5-fold orchestration** (same layout as LaBraM: `hdf5_root/fold_0` …, `output_root/fold_k`):

```bash
python user_identification/run_5fold_cv.py \
  --hdf5_root /path/to/user_id_5fold \
  --output_root /path/to/cnn_or_resnet_results \
  --n_folds 5 \
  --train_script user_identification/cnn_resnet_arcface/train_user_identification_backbone.py \
  --segment_length 4s --epochs 200 --seed 42 --backbone resnet34
```

**Artifacts per fold** (`output_dir` = e.g. `.../fold_k/`):

| File / directory | Purpose |
|------------------|--------|
| `results.json` | `val_accuracy` / `test_accuracy` (percent) for `run_5fold_cv.py` aggregation |
| `best_model.pth` | Best checkpoint |
| `{experiment}_task_active/`, `{experiment}_task_passive/` | Compact `*_user_identification_metrics.json` per task (no giant `*_predictions.json`; those are skipped for this target to avoid multi‑GB JSON) |
| `val_active_passive_metrics.json` | Validation active/passive metrics from `eval_active_passive_cnn_resnet.py --split val` (run automatically after training) |
| `user_identification_active_passive_fold_metrics.json` | Structured summary: val/test active & passive metrics, paths to task metric files |
| `user_identification_fold_metrics.log` | Same metrics in plain text |

**Standalone eval** on val or test: `python user_identification/cnn_resnet_arcface/eval_active_passive_cnn_resnet.py --checkpoint ... --hdf5_dir ... --segment_length ... --split test` (or `--split val`). Multi-fold: use `--cv_dir`, `--hdf5_root`, `--n_folds` as in `--help`.

### 5. Evaluate active vs passive tasks (test + unknown, fold-wise + summary)

Use this when you want:
- **Per-fold test accuracy** for `active` and `passive` tasks separately, and
- **Per-fold open-set metrics** (`test + unknown`) for `active` and `passive` separately,
- plus a single **cross-fold summary JSON**.

```bash
python user_identification/run_all_folds_active_passive_test_unknown_eval.py \
  --final_root final_user_identification \
  --hdf5_root_4s /path/to/hdf5_parent_with_fold_dirs \
  --n_folds 5 \
  --device cuda \
  --log_dir final_user_identification
```

Expected HDF5 layout for each segment length root:
- `/path/to/hdf5_parent_with_fold_dirs/fold_0`
- ...
- `/path/to/hdf5_parent_with_fold_dirs/fold_4`

Each `fold_k` must contain matching segment files (e.g. `eeg_data_*_4s*.h5`) and `participant_id_to_class_idx.json`.

**Outputs (example for 4s):**
- Per-fold test active/passive accuracy JSON: `final_user_identification/4s/fold_k/eval_task_type_test_accuracy.json`
- Per-fold unknown/open-set outputs:
  - `final_user_identification/4s/fold_k/eval_test_unknown_active/...`
  - `final_user_identification/4s/fold_k/eval_test_unknown_passive/...`
- All-fold summary: `final_user_identification/active_passive_test_unknown_eval_summary.json`
- Logs (when `--log_dir` is set):
  - `final_user_identification/eval_run_4s_fold0.log` ... `eval_run_4s_fold4.log`
  - `final_user_identification/eval_run_4s.log`

**1s checkpoints:** pass the parent of `fold_0` … `fold_4` for 1s data, e.g.:

```bash
python user_identification/run_all_folds_active_passive_test_unknown_eval.py \
  --final_root final_user_identification \
  --hdf5_root_1s "${HOME}/scratch" \
  --n_folds 5 \
  --device cuda \
  --log_dir final_user_identification
```

Omit any `--hdf5_root_*` you do not have; each segment length is evaluated only if its root is set (see `run_all_folds_active_passive_test_unknown_eval.py --help`).

### 6. Example: 1s five-fold LaBraM + ArcFace (`final_user_identification`)

End-to-end commands for **1s** segments, **20% unknown per fold**, HDF5 under `~/scratch/fold_k`, and checkpoints under `final_user_identification/fold_k` (same training and aggregate layout as `final_user_identification/note.txt` in the repo).

**Layout:** `run_5fold_cv.py --aggregate_only` reads `final_user_identification/fold_k/results.json`. The script `run_all_folds_active_passive_test_unknown_eval.py` looks for `best_model.pth` under `final_user_identification/1s/fold_k/`. After steps 2–3, link the flat fold dirs into `1s/` before step 5 (or train directly into `final_user_identification/1s/fold_k` and use `--output_root final_user_identification/1s` in step 3):

```bash
mkdir -p final_user_identification/1s
for k in 0 1 2 3 4; do ln -sfn "../fold_${k}" "final_user_identification/1s/fold_${k}"; done
```

**1. Prepare 5-fold data (20% unknown per fold)**

```bash
python data_processing/preprocess_user_identification.py \
  --data_root ~/scratch/preprocessed_new \
  --output_dir ~/scratch \
  --n_folds 5 \
  --segment_length 1s \
  --random_seed 42
```

Or run one fold per job (same `--data_root`, `--output_dir`, `--n_folds`, `--segment_length`, `--random_seed`; only writes `output_dir/fold_k/`):

```bash
python data_processing/preprocess_user_identification.py \
  --data_root ~/scratch/preprocessed_new \
  --output_dir ~/scratch \
  --n_folds 5 \
  --fold_index 4 \
  --segment_length 1s \
  --random_seed 42
```

**2. Train each fold**

```bash
python user_identification/train_labram_arcface.py \
  --hdf5_dir ~/scratch/fold_0 \
  --output_dir final_user_identification/fold_0 \
  --segment_length 1s --epochs 200 --batch_size 768 --seed 42

python user_identification/train_labram_arcface.py \
  --hdf5_dir ~/scratch/fold_1 \
  --output_dir final_user_identification/fold_1 \
  --segment_length 1s --epochs 200 --batch_size 768 --seed 42 \
  --resume final_user_identification/fold_1/best_model.pth

python user_identification/train_labram_arcface.py \
  --hdf5_dir ~/scratch/fold_2 \
  --output_dir final_user_identification/fold_2 \
  --segment_length 1s --epochs 200 --batch_size 768 --seed 42 \
  --resume final_user_identification/fold_2/best_model.pth

python user_identification/train_labram_arcface.py \
  --hdf5_dir ~/scratch/fold_3 \
  --output_dir final_user_identification/fold_3 \
  --segment_length 1s --epochs 200 --batch_size 768 --seed 42 \
  --resume final_user_identification/fold_3/best_model.pth

python user_identification/train_labram_arcface.py \
  --hdf5_dir ~/scratch/fold_4 \
  --output_dir final_user_identification/fold_4 \
  --segment_length 1s --epochs 200 --batch_size 768 --seed 42 \
  --resume final_user_identification/fold_4/best_model.pth
```

**3. Collect results and aggregate**

```bash
python user_identification/run_5fold_cv.py \
  --output_root final_user_identification \
  --n_folds 5 \
  --aggregate_only
```

**4. Saliency map (example: fold 3)**

```bash
conda deactivate && conda activate eeg_env && \
  python saliency_analysis/main.py \
  --checkpoint final_user_identification/fold_3/best_model.pth \
  --target_type user_identification \
  --segment_length 1s \
  --hdf5_dir ~/scratch/fold_3 \
  --output_dir final_user_identification/fold_3/saliency/
```

**5. Accuracy per task type (active / passive / unknown eval)**

```bash
python user_identification/run_all_folds_active_passive_test_unknown_eval.py \
  --final_root final_user_identification \
  --hdf5_root_1s "${HOME}/scratch" \
  --n_folds 5 \
  --device cuda \
  --log_dir final_user_identification
```

## Structure

| File | Description |
|------|-------------|
| `train_labram_arcface.py` | LaBraM + ArcFace training (primary training script) |
| `run_5fold_cv.py` | Run N-fold CV: train one model per fold on data from `--n_folds` preprocessing, then report mean ± std of val/test accuracy and save `cv_summary.json`. Optional `--train_script` (default: `train_labram_arcface.py`). Writes `fold_k/train_fold_k.log` per fold. Use `--aggregate_only` to only aggregate existing fold results (e.g. after training each fold on a separate machine). |
| `eval_active_passive_test.py` | Evaluate saved checkpoint(s) on **test** split with `task_type=active` and `task_type=passive` separately; writes per-fold metrics and mean ± std summary. |
| `run_all_folds_active_passive_test_unknown_eval.py` | Orchestrates fold/segment evaluation: per-fold test active/passive accuracy + per-fold `analyze_confidence.py --split both` for active/passive; writes aggregate summary JSON and optional split logs. |
| `run_4s.sh` | Bash wrapper for 4s training: single-GPU or multi-GPU (interactive or SLURM batch). HDF5 path and `cd` to project root are hardcoded; for custom paths use the direct Python command or edit the script. |
| `run_4s.slurm` | SLURM batch script for cluster runs (output_dir and hdf5_dir are fixed in the script; no resume by default—edit to add `--resume` or `OUTPUT_DIR` if needed). |
| `labram_model.py` | LaBraM wrapper for user identification |
| `labram_trainer.py` | Training loop and ArcFace integration |
| `analyze_confidence.py` | Confidence and OOD analysis: test/unknown splits, open-set threshold tuning |
| `run_analysis_test_unknown.sh` | Wrapper to run analysis on test + unknown (`--split both`). Requires `CHECKPOINT`; optional: `OUTPUT_DIR`, `HDF5_DIR`, `SEGMENT_LENGTH`. |
| `cnn_resnet_arcface/train_user_identification_backbone.py` | CNN or ResNet18/34/50 + ArcFace for user identification (compact metrics JSON; see [CNN and ResNet ArcFace user identification](#cnn-and-resnet-arcface-user-identification)). |
| `cnn_resnet_arcface/eval_active_passive_cnn_resnet.py` | Evaluate saved CNN/ResNet user-ID checkpoints on val or test, active and passive. |
| `main.py` | Alternative: run CNN and/or LaBraM experiments via the CNN experiment pipeline (not ArcFace). |
| `validate.py` | Validation script: checks mapping, class coverage, and label consistency. Uses `--hdf5_dir` (script has a default path; override as needed). |

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

**Key arguments:** `--hdf5_dir` (required), `--segment_length` (1s/2s/4s), `--output_dir` (default when run directly: `./results_labram_arcface`; `run_4s.sh` uses `OUTPUT_DIR`), `--resume` (e.g. `best_model.pth`), `--distributed` (only for multi-GPU). ArcFace: `--arcface_margin`, `--arcface_scale`. Early stopping: `--early_stopping_patience`, `--early_stopping_min_delta`. Optimizer: `--lr`, `--layer_decay`, `--weight_decay`, `--warmup_epochs`, `--clip_grad`. Run `python user_identification/train_labram_arcface.py --help` for full list.

### 5-fold cross-validation (disjoint unknown participants)

To report mean ± std across 5 folds with **no overlap** in unknown participants (each fold holds out a different 20% as unknown):

1. **Preprocess with 5 folds** (from project root):
   ```bash
   python data_processing/preprocess_user_identification.py \
     --data_root /path/to/preprocessed_new \
     --output_dir /path/to/user_id_5fold \
     --n_folds 5 \
     --segment_length 4s \
     --random_seed 42
   ```
   This creates `output_dir/fold_0`, ..., `output_dir/fold_4`; in fold *k*, the *k*-th group of participants is unknown and the rest are known.

2. **Run 5-fold training and aggregation** (single machine):
   ```bash
   python user_identification/run_5fold_cv.py \
     --hdf5_root /path/to/user_id_5fold \
     --output_root /path/to/results_5fold \
     --n_folds 5 \
     --segment_length 4s --epochs 200 --batch_size 192 --seed 42
   ```
   Each fold is trained with `train_labram_arcface.py` unless you pass `--train_script`; val/test accuracy are read from each fold's `results.json` and summarized as mean ± std. Summary is written to `output_root/cv_summary.json`. Training logs: `output_root/fold_k/train_fold_k.log`.

3. **Distributed: train each fold on a separate machine**, then aggregate once:
   - Prepare data once (step 1 above); copy or share `user_id_5fold/fold_0` … `fold_4` as needed.
   - On the machine for fold *k*, run (from project root):
     ```bash
     python user_identification/train_labram_arcface.py \
       --hdf5_dir /path/to/fold_k \
       --output_dir /path/to/results_5fold/fold_k \
       --segment_length 4s --epochs 200 --batch_size 192 --seed 42
     ```
     Use the same training args for all folds so results are comparable.
   - After all folds are done, copy `results_5fold/fold_0` … `fold_4` (each must contain `results.json`) to one `output_root`, then run:
     ```bash
     python user_identification/run_5fold_cv.py \
       --output_root /path/to/results_5fold \
       --n_folds 5 \
       --aggregate_only
     ```
   This writes `output_root/cv_summary.json` from the existing `results.json` files without running training.

## How unknown participants are handled

**Unknown participants** are users that the model has never seen during training. The pipeline handles them as follows.

1. **Preprocessing (data_processing)**  
   When building user-identification HDF5 data, run preprocessing with **`--consider_unknown`** (e.g. in `data_processing/preprocess_user_identification.py`). A fraction of participants (e.g. `--unknown_ratio 0.2`) are held out: they are **not** in the train/val/test splits and do **not** appear in `participant_id_to_class_idx.json`. Their segments are written to a separate **unknown** split (e.g. `eeg_data_unknown_4s*.h5`). At load time, unknown samples get a sentinel label (e.g. `-1`) so the model never sees their identity during training.

2. **Training**  
   Training uses only **known** users (train/val/test). The model and ArcFace head are fit to recognize these identities. Unknown users are never used in training.

3. **Analysis and open-set behavior**  
   After training, run analysis with `--split both` (test + unknown). For each sample we get:
   - **Embeddings** from the model (ArcFace backbone).
   - **Known-user centroids**: mean embedding per known class, computed on the training set.
   - **OOD distance**: minimum L2 distance from the sample’s embedding to any known-user centroid (high = more “out-of-distribution”).
   - **Known-user score** = 1/(1 + OOD distance): low for unknown participants, high for known ones. This score is used (instead of softmax) to decide “known vs unknown”.
   - **Open-set threshold**: we tune a threshold on the known-user score so that “predict unknown” when score < threshold. The threshold is chosen to maximize **open-set accuracy** (correctly identifying known users + correctly rejecting unknown users) on the combined test+unknown set.

So: unknown participants are **created at preprocessing** (held-out users, unknown split), **ignored at training**, and **detected at analysis** via feature-space OOD and the known-user score, with a tunable threshold for open-set recognition.

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
| `--hdf5_dir` | script default | HDF5 directory (user-identification preprocessed); run with `--help` to see the default path |
| `--segment_length` | `4s` | `1s`, `2s`, or `4s` (must match training) |
| `--split` | `test` | `train`, `val`, `test`, `unknown`, or `both` |
| `--task_type` | `both` | `both`, `active`, or `passive` (filters train/test/unknown datasets by task type) |
| `--output_dir` | `./confidence_analysis` | Base directory for all outputs (wrapper `run_analysis_test_unknown.sh` uses `./confidence_analysis_test_unknown`) |
| `--batch_size` | 128 | Inference batch size |
| `--num_workers` | 4 | DataLoader workers |
| `--device` | `cuda` | `cuda` or `cpu` |

**`--split both`** runs on test and unknown; when the checkpoint uses ArcFace, it also computes feature-space OOD (distance to known-user centroids), **known-user score** (1/(1+OOD distance)), and **open-set threshold** tuning.

**Split semantics:**
- `test` = known participants only
- `unknown` = held-out participants only (`-1` label)
- open-set summary (in `open_set_summary.json`) is computed on **test + unknown**

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
# From project root; --hdf5_dir has a script default (see --help)
python user_identification/validate.py --hdf5_dir /path/to/processed_eeg_data_user_identification
```

`validate.py` checks that all participants are in the mapping, all classes appear in training data, and labels match. Override `--hdf5_dir` if your data is not at the script’s default path.

## Alternative: CNN / LaBraM experiments (main.py)

For running CNN and/or LaBraM experiments through the **CNN experiment pipeline** (standard cross-entropy, not ArcFace):

```bash
# From project root
python user_identification/main.py \
  --mode 4s \
  --model both \
  --labram_checkpoint /path/to/pretrained_labram.pth \
  --epochs 50 \
  --learning_rate 0.0001 \
  --num_gpus 4 \
  --hdf5_dir /path/to/processed_eeg_data_user_identification
```

**Arguments:** `--mode` (1s/2s/4s), `--model` (cnn/labram/both), `--labram_checkpoint`, `--epochs`, `--learning_rate`, `--num_gpus`, `--hdf5_dir`, `--results_dir`, `--reports_dir`, `--batch_size` (default: None → segment-length based: 256 for 1s, 192 for 2s, 128 for 4s), `--random_seed`. LaBraM without `--labram_checkpoint` prompts to continue (not recommended).

## Notes

- LaBraM training in `run_4s.sh` uses a pre-trained LaBraM backbone (`--pretrained_path`); the checkpoint saves the full model and ArcFace state for analysis.
- Unknown participants are created at preprocessing: run `data_processing/preprocess_user_identification.py` with **`--consider_unknown`** (and optionally `--unknown_ratio`); otherwise the unknown split is empty. See [How unknown participants are handled](#how-unknown-participants-are-handled) above.
- For **saliency and topography** on user identification models (LaBraM + ArcFace), use the `saliency_analysis` module with `--target_type user_identification`. Run from project root: `python saliency_analysis/main.py --checkpoint <path/to/best_model.pth> --target_type user_identification --segment_length 4s --hdf5_dir <same_as_training_fold> --output_dir <output_dir>`. Use the same `--hdf5_dir` as for training that fold (directory containing `participant_id_to_class_idx.json` and HDF5 files). See `saliency_analysis/README.md` and `saliency_analysis/TOPOGRAPHY_GUIDE.md`.
