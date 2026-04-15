# Registration Evaluation (Unknown-Only)

This directory contains scripts for evaluating **user registration and open-world identification** using **only unknown-user data** from each fold.

The current protocol does **not** use train/val/test-known user data for decision logic. It uses unknown users only, enrolls a subset, and evaluates:

- classification of enrolled users by identity on unseen probe EEG
- rejection of non-enrolled users as `UNKNOWN`

---

## What This Module Does

For a given fold and task (`active` or `passive`):

1. Load unknown-split EEG samples.
2. Split unknown participants into:
   - **registered** users (enrolled)
   - **unregistered** users (must be labeled `UNKNOWN`)
3. For each registered user, split their segments into:
   - **enrollment** segments (build prototype)
   - **probe** segments (evaluation)
4. Build one L2-normalized prototype per registered user.
5. Evaluate:
   - **Closed-set registered classification** (registered probe only)
   - **Open-world K+1 classification** (K registered IDs + 1 `UNKNOWN`)

In open-world mode, prediction is:

- nearest registered prototype gives a candidate ID
- if min distance to prototypes is larger than a threshold, output `UNKNOWN`

---

## Files in This Folder

- `run_registration_eval.py`  
  Single-fold evaluation (core logic).

- `run_all_folds_registration_eval.py`  
  Run evaluation across folds and aggregate metrics.

- `run_registration_grid_search.py`  
  Grid search over threshold/objective-related hyperparameters.

- `unknown_hdf5_index.py`  
  Utilities to index unknown HDF5 samples and split participants.

- `registration_map_dataset.py`  
  Map-style dataset that loads samples referenced by unknown-split indices.

---

## Data and Checkpoint Requirements

### HDF5 layout

`--hdf5_dir` should point to a fold directory like:

- `${HOME}/scratch/fold_0`

It must contain unknown-split HDF5 files expected by `EEGDataLoader._get_unknown_hdf5_file_paths`, such as:

- `eeg_data_unknown_<segment>.h5`
- or partitioned variants `eeg_data_unknown_<segment>_part*.h5`

### Checkpoints

For all-fold scripts, checkpoints are resolved strictly as:

- `<cv_dir>/fold_k/best_model.pth`

Both all-fold scripts enforce this and fail fast if missing.

---

## Core Metrics Reported

From `run_registration_eval.py` output JSON:

1. `registered_user_classification_probe_only`
   - closed-set ID quality among registered users only
   - metrics: accuracy, balanced accuracy, macro-F1, MCC

2. `open_world_user_classification`
   - K+1 classification (registered IDs + `UNKNOWN`)
   - includes:
     - selected threshold and metrics
     - reference threshold maximizing plain accuracy
     - assignment breakdown counts
     - unknown rejection rate

### Key fields to watch

- `unknown_recall`  
  Fraction of unregistered samples correctly labeled `UNKNOWN`.

- `mean_registered_recall`  
  Mean per-registered-user recall (identity retention).

- `f1_macro`  
  Macro-F1 over K+1 classes.

---

## Threshold Selection Criteria

Supported `--threshold_criterion` values:

- `f1_macro`
- `balanced_accuracy`
- `accuracy`
- `mcc`
- `weighted_recall_balance`
- `harmonic_recall_balance`

### Joint optimization criteria

- `weighted_recall_balance`  
  `unknown_recall_weight * unknown_recall + (1 - unknown_recall_weight) * mean_registered_recall`

- `harmonic_recall_balance`  
  Harmonic mean of `unknown_recall` and `mean_registered_recall` (penalizes imbalance).

Use `--unknown_recall_weight` in `[0,1]` with `weighted_recall_balance`.

---

## Usage

Run from repository root.

### 1) Single-fold evaluation

```bash
python user_identification/registering_users/run_registration_eval.py \
  --checkpoint final_user_identification/4s/fold_0/best_model.pth \
  --hdf5_dir "${HOME}/scratch/fold_0" \
  --segment_length 4s \
  --task_type active \
  --output_dir registering_users_results/fold_0_active \
  --register_fraction_of_unknown 0.5 \
  --enrollment_fraction_per_user 0.5 \
  --threshold_criterion weighted_recall_balance \
  --unknown_recall_weight 0.5 \
  --threshold_grid_points 501 \
  --device cuda
```

### 2) All-fold evaluation

```bash
python user_identification/registering_users/run_all_folds_registration_eval.py \
  --cv_dir final_user_identification/4s \
  --hdf5_root "${HOME}/scratch" \
  --segment_length 4s \
  --task_type active \
  --n_folds 5 \
  --output_root registering_users_results/all_folds_active_4s \
  --threshold_criterion weighted_recall_balance \
  --unknown_recall_weight 0.5 \
  --threshold_grid_points 501 \
  --device cuda
```

### 3) Grid search for better operating points

```bash
python user_identification/registering_users/run_registration_grid_search.py \
  --cv_dir final_user_identification/4s \
  --hdf5_root "${HOME}/scratch" \
  --segment_length 4s \
  --task_type active \
  --n_folds 5 \
  --output_root registering_users_results/grid_search_active_4s \
  --grid_enrollment_fractions "0.5,0.6,0.7" \
  --grid_unknown_recall_weights "0.4,0.5,0.6,0.7" \
  --grid_register_fractions "0.5" \
  --grid_threshold_criteria "weighted_recall_balance,harmonic_recall_balance" \
  --objective_unknown_weight 0.5 \
  --threshold_grid_points 501 \
  --device cuda
```

---

## Output Files

### `run_registration_eval.py`

Under `--output_dir`:

- `registration_open_set_report.json`  
  Full report with selected threshold, metrics, breakdown.

- `registration_eval_arrays.npz`  
  Numeric arrays (prototypes, distances, predictions, threshold sweep).

### `run_all_folds_registration_eval.py`

Under `--output_root`:

- per-fold outputs under `fold_k/`
- `registration_eval_cv_summary.json` with mean/std aggregates

### `run_registration_grid_search.py`

Under `--output_root`:

- one folder per config
- `grid_search_ranked_results.json` with ranked configs and best config

---

## Practical Tuning Guidance

If you want better unknown rejection:

- increase `unknown_recall_weight` (for `weighted_recall_balance`)
- consider stricter threshold criteria

If you want better registered-user retention/ID:

- decrease `unknown_recall_weight`
- increase `enrollment_fraction_per_user` (while keeping enough probe samples)

For deployment-quality selection:

- tune on all folds, not only one fold
- choose threshold criterion based on target trade-off, not plain accuracy alone

---

## Notes and Constraints

- Fractions must be in valid ranges:
  - `register_fraction_of_unknown`: `(0,1)`
  - `enrollment_fraction_per_user`: `(0,1)`
  - `unknown_recall_weight`: `[0,1]`
- `threshold_grid_points` must be `>= 11`
- At least 2 unknown participants are required per fold split
- Each registered participant must have at least 2 segments to split enrollment/probe

