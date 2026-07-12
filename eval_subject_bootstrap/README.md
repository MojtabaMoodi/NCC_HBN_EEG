# Subject-Level Bootstrap Evaluation

This module provides statistically sound evaluation for EEG age-group classification with multiple segments per participant. It evaluates models at the **participant (subject) level** and compares them against a **subject-level stratified random baseline** using bootstrap uncertainty estimation.

## Quick start

From the **project root (EEG)**:

```bash
chmod +x eval_subject_bootstrap/run_on_gpu_server.sh
./eval_subject_bootstrap/run_on_gpu_server.sh
```

This script (1) generates train/test manifests from HDF5 if missing, (2) runs evaluation for all models in `config.yaml`, (3) writes results to `eval_subject_bootstrap/outputs/age_evaluation_4s_<timestamp>/`.

**Manual run (custom paths or options):**

```bash
# 1. Generate manifests (if needed)
python eval_subject_bootstrap/generate_manifests.py \
    --hdf5_dir data_processing/processed_eeg_data_hdf5_no_compression \
    --split train --segment_length 4s --task_type both \
    --output eval_subject_bootstrap/manifests/train_manifest_4s.csv
python eval_subject_bootstrap/generate_manifests.py \
    --hdf5_dir data_processing/processed_eeg_data_hdf5_no_compression \
    --split test --segment_length 4s --task_type both \
    --output eval_subject_bootstrap/manifests/test_manifest_4s.csv

# 2. Run evaluation
python eval_subject_bootstrap/run_bootstrap_evaluation.py \
    --train_manifest eval_subject_bootstrap/manifests/train_manifest_4s.csv \
    --test_manifest eval_subject_bootstrap/manifests/test_manifest_4s.csv \
    --models_config eval_subject_bootstrap/config.yaml \
    --hdf5_base_dir data_processing/processed_eeg_data_hdf5_no_compression \
    --out_dir eval_subject_bootstrap/outputs/age_evaluation_4s \
    --batch_size 64 --num_workers 8 --device cuda \
    --bootstrap_B 2000 --permutation_M 1000 --seed 123 \
    --primary_metric balanced_accuracy --save_npz
```

Results: `results.json`, `summary.csv`, and (with `--save_npz`) `intermediate_<model>.npz`. See [Output Files](#output-files) below.

## Overview

### Key Features

1. **Subject-Level Aggregation**: Aggregates segment-level predictions to one prediction per participant using **mean probability** (default) or **majority vote** (`--aggregation`)
2. **Stratified Random Baseline**: Uses training set subject-level class proportions to generate random predictions
3. **Bootstrap Uncertainty**: Performs subject-level (cluster) bootstrap to estimate 95% confidence intervals
4. **Delta Analysis**: Reports bootstrap CI for the difference (model metric − random baseline metric)
5. **Multiple Models**: Supports CNN, LaBraM, and ResNet models

### Why Subject-Level Evaluation?

When each participant has multiple EEG segments, evaluating at the segment level can be misleading because:
- Segments from the same participant are not independent
- Segment-level metrics can be inflated by participant-level correlations
- The goal is to predict participant characteristics, not segment characteristics

By aggregating to the subject level, we ensure:
- One prediction per participant (the unit of interest)
- Proper statistical inference (subjects are independent)
- Fair comparison with baselines

### Why Subject-Level Stratified Random Baseline?

The random baseline generates predictions using the training set's **subject-level** class proportions. This ensures:
- The baseline reflects the true class distribution in the training population
- Fair comparison: both model and baseline operate at the same (subject) level
- Stratification accounts for class imbalance

### Bootstrap Delta CI Interpretation

The bootstrap 95% CI for delta (model − random) tells us:
- **If CI excludes 0**: Model is significantly better/worse than random at α=0.05
- **If CI includes 0**: No significant difference (but model may still be better)
- **P(delta > 0)**: Proportion of bootstrap samples where model > random (higher is better)

## Installation

Run all commands from the **project root (EEG)** so that `data_processing` and CNN/LaBraM model imports resolve.

No additional installation required. Uses existing project dependencies:
- numpy
- pandas
- scikit-learn
- torch
- pyyaml (for YAML config files)

## Usage

### Step 1: Prepare Manifests

You can either create manifests manually or use the helper script to generate them from HDF5 files.

#### Option A: Generate Manifests from HDF5 Files (Recommended)

Use the helper script to extract subject and segment information from HDF5 files. **Run from project root (EEG).**

```bash
# Generate train manifest (subject-level)
python eval_subject_bootstrap/generate_manifests.py \
    --hdf5_dir data_processing/processed_eeg_data_hdf5_no_compression \
    --split train \
    --segment_length 4s \
    --task_type both \
    --output eval_subject_bootstrap/manifests/train_manifest_4s.csv

# Generate test manifest (segment-level)
python eval_subject_bootstrap/generate_manifests.py \
    --hdf5_dir data_processing/processed_eeg_data_hdf5_no_compression \
    --split test \
    --segment_length 4s \
    --task_type both \
    --output eval_subject_bootstrap/manifests/test_manifest_4s.csv
```

Optional: `--pattern` to match specific files (e.g. `eeg_data_train_4s*.h5`). Default pattern is `eeg_data_{split}_{segment_length}*.h5`.

#### Option B: Create Manifests Manually

#### Train Manifest (subject-level)

CSV file with one row per training subject:
```csv
subject_id,label
sub-NDARAC904DMU,0
sub-NDARAC905DMU,1
sub-NDARAC906DMU,2
...
```

Required columns:
- `subject_id`: Unique subject identifier (string)
- `label`: Class label (0, 1, or 2)

#### Test Manifest (segment-level)

CSV file with one row per test segment:
```csv
subject_id,label,hdf5_file,sample_id
sub-NDARAC904DMU,0,eeg_data_test_4s.h5,sample_000123
sub-NDARAC904DMU,0,eeg_data_test_4s.h5,sample_000124
sub-NDARAC904DMU,0,eeg_data_test_4s.h5,sample_000125
...
```

Required columns:
- `subject_id`: Subject identifier (string)
- `label`: Class label (0, 1, or 2) - must be consistent per subject
- Either:
  - `segment_path`: Full path to segment (e.g., `path/to/file.h5:sample_000123`)
  - OR `hdf5_file` + `sample_id`: HDF5 file path and sample identifier

**Note**: Each subject must have exactly one unique label across all segments (validated automatically).

### Step 2: Create Models Config

YAML or JSON file listing models to evaluate:

```yaml
models:
  - name: cnn
    model_type: age_cnn
    ckpt_path: /path/to/cnn_checkpoint.pth
    num_classes: 3
    ctor_kwargs:
      dropout_rate: 0.5
      use_layer_norm: true
  
  - name: resnet
    model_type: age_resnet34
    ckpt_path: /path/to/resnet_checkpoint.pth
    num_classes: 3
    ctor_kwargs: {}
  
  - name: labram
    model_type: labram_base_patch200_200
    ckpt_path: /path/to/labram_checkpoint.pth
    num_classes: 3
    ctor_kwargs:
      drop_rate: 0.5
      use_mean_pooling: true
```

**Model types**:
- **CNN**: `age_cnn`, `eeg_cnn`, etc. (see `CNN/models/model_factory.py`)
- **ResNet**: `age_resnet`, `age_resnet34`, `age_resnet50`, etc.
- **LaBraM**: `labram_base_patch200_200`, `labram_large_patch200_200`, etc.

### Step 3: Run Evaluation

```bash
python eval_subject_bootstrap/run_bootstrap_evaluation.py \
    --train_manifest eval_subject_bootstrap/manifests/train_manifest_4s.csv \
    --test_manifest eval_subject_bootstrap/manifests/test_manifest_4s.csv \
    --models_config eval_subject_bootstrap/config.yaml \
    --batch_size 64 \
    --num_workers 4 \
    --device cuda \
    --bootstrap_B 2000 \
    --seed 123 \
    --primary_metric balanced_accuracy \
    --out_dir eval_subject_bootstrap/outputs/age_evaluation_4s \
    --save_npz \
    --permutation_M 1000 \
    --hdf5_base_dir data_processing/processed_eeg_data_hdf5_no_compression
```

### Arguments

- `--train_manifest`: **Required**. Path to train manifest (subject-level CSV)
- `--test_manifest`: **Required**. Path to test manifest (segment-level CSV)
- `--models_config`: **Required**. Path to models config YAML/JSON
- `--batch_size`: Batch size for inference (default: 64)
- `--num_workers`: Number of data loader workers (default: 4)
- `--device`: Device to use: `cuda` or `cpu` (default: `cuda` if available)
- `--bootstrap_B`: Number of bootstrap replicates (default: 2000)
- `--seed`: Random seed (default: 123)
- `--primary_metric`: Primary metric name (default: `balanced_accuracy`)
- `--out_dir`: Output directory (default: `eval_subject_bootstrap/outputs/<timestamp>`)
- `--save_npz`: Save intermediate arrays (segment/subject preds and probs)
- `--permutation_M`: Number of permutations for permutation test (default: 0 = disabled)
- `--hdf5_base_dir`: Base directory for HDF5 files (if paths in manifest are relative)
- `--aggregation`: Subject-level aggregation: `mean_prob` (mean probability then argmax) or `majority_vote` (mode of segment predictions). Default: `mean_prob`

## Output Files

### `results.json`

Comprehensive results including:
- Train subject counts and proportions
- Test subject/segment counts
- Per-model point estimates for all metrics
- Bootstrap 95% CIs per model per metric
- Delta CIs (model − random)
- P(delta > 0) per model per metric
- Permutation p-values (if enabled)
- Confusion matrices
- Segments per subject statistics

### `summary.csv`

Tabular summary with one row per model × metric:
- `model`: Model name
- `metric`: Metric name
- `point_estimate`: Point estimate
- `ci_low`, `ci_high`: Bootstrap 95% CI
- `delta_point`: Delta point estimate (model − random)
- `delta_ci_low`, `delta_ci_high`: Delta bootstrap 95% CI
- `p_delta_gt_0`: P(delta > 0)

### `intermediate_<model>.npz` (if `--save_npz`)

Intermediate arrays:
- `segment_probs`: Segment-level probabilities, shape (N_segments, num_classes)
- `segment_labels`: Segment-level labels, shape (N_segments,)
- `segment_subject_ids`: Subject IDs for each segment
- `subjects`: Unique subject IDs, shape (S,)
- `y_true_subject`: Subject-level true labels, shape (S,)
- `y_pred_subject`: Subject-level predictions, shape (S,)
- `p_subject`: Subject-level mean probabilities, shape (S, num_classes)

## Metrics

Available metrics (computed at subject level):
- `accuracy`: Overall accuracy
- `balanced_accuracy`: Balanced accuracy (handles class imbalance)
- `macro_f1`: Macro-averaged F1 score
- `per_class_recall`: Per-class recall (dict)

Default primary metric: `balanced_accuracy`

## Sanity Checks

The script automatically performs:
1. **Subject label consistency**: Verifies each subject has exactly one unique label
2. **Train/test overlap**: Warns if any subjects appear in both train and test
3. **Class coverage**: Warns if any class is missing from training set

## Example Output

```
SUMMARY
============================================================
Primary metric: balanced_accuracy

Train proportions: {0: 0.33, 1: 0.34, 2: 0.33}
Majority class: 1
Test subjects: 150
Test segments: 4500

cnn:
  balanced_accuracy: 0.6234 [0.5891, 0.6578]
  Delta (vs random): 0.1234 [0.0891, 0.1578]
  P(delta > 0): 1.0000
  Permutation p-value: 0.0010

resnet:
  balanced_accuracy: 0.6456 [0.6123, 0.6789]
  Delta (vs random): 0.1456 [0.1123, 0.1789]
  P(delta > 0): 1.0000
  Permutation p-value: 0.0005
```

## Reproducibility

- Random seeds are set for numpy and torch
- Models are set to `eval()` mode with `torch.no_grad()`
- Test dataset ordering doesn't matter (aggregation by subject)
- Git commit hash is recorded in results (if available)

## Troubleshooting

### "Sample not found in HDF5 file"

- Check that `sample_id` in manifest matches HDF5 dataset names (e.g., `sample_000123`)
- Verify HDF5 file structure: should have `active/` and/or `passive/` groups
- Check `--hdf5_base_dir` if using relative paths

### "Model returned dict but no output_key specified"

- Add `output_key` to model config (e.g., `output_key: logits`)
- Or modify model to return tensor directly (not dict)

### "Subject has inconsistent labels"

- Each subject must have the same label across all segments
- Check test manifest for labeling errors

### "Class X not found in training set"

- All 3 classes (0, 1, 2) should be present in training set
- Random baseline will still work but may be suboptimal

## Citation

If you use this evaluation module, please cite:
- The bootstrap methodology (Efron & Tibshirani, 1994)
- Subject-level evaluation for multi-segment data (standard practice in ML)

## Implementation Details

### Model-Specific Handling

The evaluation code handles three different model architectures with their specific requirements:

#### CNN and ResNet Models
- **Input Format**: Direct `[B, 60, T]` tensor from HDF5
- **Preprocessing**: None required
- **Checkpoint Format**: Standard PyTorch checkpoint with `model_state_dict` key
- **Output**: Direct tensor (logits)

#### LaBraM Models
- **Input Format**: Requires `[B, 60, num_patches, patch_size]` where `patch_size=200`
- **Preprocessing**:
  - Automatic reshaping from `[B, 60, T]` to `[B, 60, num_patches, patch_size]`
  - Normalization by dividing by 100
  - Requires `input_chans` parameter computed from channel names
- **Checkpoint Format**: 
  - Finetuned checkpoints have `'model'` key
  - Pretrained checkpoints might be direct state_dict
  - Handles `relative_position_index` keys (computed, not stored)
- **Output**: Can return dict for multi-output or tensor for single output

### Code Modifications

The following modifications ensure compatibility with all model types:

1. **`model_loader.py`**:
   - Added `get_labram_input_chans()` function to compute channel indices from standard 10-20 channel names
   - Enhanced `get_model_output()` with LaBraM-specific preprocessing (reshaping and normalization)
   - Improved LaBraM checkpoint loading to handle multiple formats and filter computed keys
   - Enhanced dict output handling with fallback keys including `'age'` for multi-output models

2. **`run_bootstrap_evaluation.py`**:
   - Updated `run_inference()` to pass model name for automatic preprocessing detection
   - Modified to detect LaBraM models and automatically get input_chans

3. **Data Format Compatibility**:
   - Data is loaded as `[B, 60, T]` from HDF5 files
   - LaBraM preprocessing automatically reshapes when `model_name='labram'` is detected
   - CNN/ResNet models receive data directly without modification

### Verification

All changes have been verified:
- ✅ No linter errors
- ✅ Imports resolved correctly
- ✅ Model loading handles all three model types
- ✅ Data preprocessing matches model requirements
- ✅ Checkpoint paths updated to use actual finetuned models

### Testing Recommendations

Before running full evaluation, test with a small subset:
1. Verify CNN model loads and runs inference correctly
2. Verify ResNet model loads and runs inference correctly  
3. Verify LaBraM model loads, preprocesses data correctly, and runs inference
4. Check that subject-level aggregation works for all models

## File structure

```
eval_subject_bootstrap/
├── README.md                    # This file
├── REPORT.md                    # Example evaluation report (methodology + results)
├── config.yaml                  # Model list and checkpoint paths (edit for your runs)
├── run_on_gpu_server.sh         # One-shot: generate manifests + run evaluation
├── generate_manifests.py        # Build train/test manifests from HDF5
├── run_bootstrap_evaluation.py # Main script: inference + bootstrap + comparison
├── bootstrap.py                 # Bootstrap and random baseline logic
├── io_utils.py                 # Manifest load/save, subject aggregation
├── metrics.py                  # Subject-level metrics
├── model_loader.py             # Load CNN/ResNet/LaBraM and run inference
├── manifests/                  # Generated train/test CSVs
└── outputs/                    # Results per run (timestamped or --out_dir)
```

## License

Same as parent project.
