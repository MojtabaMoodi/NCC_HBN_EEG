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
- if the **reject signal** is worse than a tuned threshold, output `UNKNOWN`

By default the reject signal is **raw min L2 distance** \(d\) to the nearest prototype. For parity with `analyze_confidence.py` open-set analysis, you can instead use **`known_user_score = 1/(1+d)`** on that same \(d\) and sweep thresholds in score space (`--threshold_on known_user_score`). The map \(d \mapsto 1/(1+d)\) is strictly monotone, so accept/reject decisions are equivalent to choosing a distance cutoff; only the reported threshold units and the discrete grid points differ.

---

## Files in This Folder

- `run_registration_eval.py`  
  Single-fold evaluation (core logic).

- `run_all_folds_registration_eval.py`  
  Run evaluation across folds and aggregate metrics.

- `run_registration_grid_search.py`  
  Grid search over threshold/objective-related hyperparameters.

- `run_registration_pipeline.py`  
  Runs **active** and **passive** grid searches, writes a **manifest** with best configs, then (by default) runs **all-fold** evaluation for each task using those best settings.

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
  --threshold_on known_user_score \
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
  --threshold_on known_user_score \
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
  --threshold_on known_user_score \
  --device cuda
```

#### Ranking tie-breaks

`grid_search_ranked_results.json` lists configs sorted by descending **objective score**
(`objective_unknown_weight * unknown_recall_mean + (1 - w) * mean_registered_recall_mean`).
When that score ties, ranking uses **higher `open_world_f1_macro_mean`**, then **higher `open_world_mcc_mean`**.

#### Optional disk cleanup after a grid

To keep only the top few config directories (plus `grid_search_ranked_results.json`):

```bash
python user_identification/registering_users/run_registration_grid_search.py \
  ...same args as above... \
  --prune_configs \
  --prune_keep_top_k 3
```

This deletes sibling `enroll_*` directories that are not in the top **K** after ranking. Use with care.

### 4) Full pipeline (active + passive grid, manifest, final all-fold eval)

This automates the recommended workflow:

1. Grid search **active** → `grid_search_active/`
2. Grid search **passive** → `grid_search_passive/`
3. Write `registration_pipeline_manifest.json` (best configs, paths, and shell-ready final-eval commands)
4. Run `run_all_folds_registration_eval.py` for **active** and **passive** using each task’s best hyperparameters

```bash
python user_identification/registering_users/run_registration_pipeline.py \
  --cv_dir final_user_identification/4s \
  --hdf5_root "${HOME}/scratch" \
  --segment_length 4s \
  --n_folds 5 \
  --output_root registering_users_results/pipeline_4s_active_passive \
  --objective_unknown_weight 0.5 \
  --threshold_on known_user_score \
  --device cuda
```

Useful flags:

- `--prune_after_grid` — forward `--prune_configs` to each grid run (keeps top `--prune_keep_top_k` config dirs per grid).
- `--skip_grid` — skip a grid if `grid_search_ranked_results.json` already exists under that grid output directory (resume-friendly).
- `--no_final_eval` — only run grids and write the manifest (skip the two all-fold eval steps).

Outputs under `--output_root`:

| Path | Meaning |
|------|---------|
| `grid_search_active/` | Active grid; read `grid_search_ranked_results.json` |
| `grid_search_passive/` | Passive grid |
| `registration_pipeline_manifest.json` | Best configs + commands |
| `eval_all_folds_active_best_from_grid/` | All-fold active eval with best hyperparameters |
| `eval_all_folds_passive_best_from_grid/` | All-fold passive eval with best hyperparameters |

**After a pipeline run**, archive or delete old exploratory grids if disk is tight; keep at minimum the manifest and the two `eval_all_folds_*` trees you care about for reporting.

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

### `run_registration_pipeline.py`

Under `--output_root`:

- `registration_pipeline_manifest.json`
- `grid_search_active/`, `grid_search_passive/`
- optional `eval_all_folds_*` directories if final eval was not disabled

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
- run **active and passive separately**; do not assume the same hyperparameters transfer
- use `run_registration_pipeline.py` once you are happy with the search space, so best configs and final evals stay reproducible from one command

---

## Notes and Constraints

- Fractions must be in valid ranges:
  - `register_fraction_of_unknown`: `(0,1)`
  - `enrollment_fraction_per_user`: `(0,1)`
  - `unknown_recall_weight`: `[0,1]`
- `threshold_grid_points` must be `>= 11`
- `--threshold_on` is `distance` (default) or `known_user_score` (same formula as `analyze_confidence.ood_distance_to_known_user_score` applied to min distance to prototypes)
- At least 2 unknown participants are required per fold split
- Each registered participant must have at least 2 segments to split enrollment/probe

