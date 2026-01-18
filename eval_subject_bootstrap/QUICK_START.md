# Quick Start Guide - Running Evaluation on GPU Server

## Prerequisites
- GPU server with CUDA available
- Python environment with: torch, numpy, pandas, scikit-learn, h5py, pyyaml
- Access to the project directory and data files

## Step-by-Step Instructions

**Quick Start (Recommended):**
```bash
chmod +x eval_subject_bootstrap/run_on_gpu_server.sh
./eval_subject_bootstrap/run_on_gpu_server.sh
```

This script automatically:
1. Generates manifests (if they don't exist)
2. Runs the full evaluation with all three models
3. Saves results with timestamp

**Manual Steps (if you need customization):**

### Step 1: Generate Manifests (if not already done)

Generate train manifest (subject-level):
```bash
python eval_subject_bootstrap/generate_manifests.py \
    --hdf5_dir data_processing/processed_eeg_data_hdf5_no_compression \
    --split train \
    --segment_length 4s \
    --task_type both \
    --output eval_subject_bootstrap/manifests/train_manifest_4s.csv
```

Generate test manifest (segment-level):
```bash
python eval_subject_bootstrap/generate_manifests.py \
    --hdf5_dir data_processing/processed_eeg_data_hdf5_no_compression \
    --split test \
    --segment_length 4s \
    --task_type both \
    --output eval_subject_bootstrap/manifests/test_manifest_4s.csv
```

### Step 2: Run Evaluation

**Option A: Using the automated script (recommended)**
```bash
chmod +x eval_subject_bootstrap/run_on_gpu_server.sh

# This script automatically generates manifests (if needed) and runs evaluation
./eval_subject_bootstrap/run_on_gpu_server.sh
```

**Option B: Direct Python command (for custom parameters)**
```bash
python eval_subject_bootstrap/evaluate_models_vs_random.py \
    --train_manifest eval_subject_bootstrap/manifests/train_manifest_4s.csv \
    --test_manifest eval_subject_bootstrap/manifests/test_manifest_4s.csv \
    --models_config eval_subject_bootstrap/config.yaml \
    --hdf5_base_dir data_processing/processed_eeg_data_hdf5_no_compression \
    --out_dir eval_subject_bootstrap/outputs/age_evaluation_4s \
    --batch_size 64 \
    --num_workers 8 \
    --device cuda \
    --bootstrap_B 2000 \
    --permutation_M 1000 \
    --seed 123 \
    --primary_metric balanced_accuracy \
    --save_npz
```

### Step 3: Check Results

Results will be saved in `eval_subject_bootstrap/outputs/age_evaluation_4s/`:
- `results.json` - Complete results with bootstrap CIs
- `summary.csv` - Summary table with point estimates and CIs
- `bootstrap_deltas_*.csv` - Bootstrap distributions (if --save_npz used)
- `delta_histogram_*.png` - Visualizations (if generated)

## Parameters Explained

- `--batch_size 64`: Increase if GPU has more memory (e.g., 128)
- `--num_workers 8`: Data loading workers (adjust based on CPU cores)
- `--device cuda`: Use GPU (automatically falls back to CPU if CUDA unavailable)
- `--bootstrap_B 2000`: Number of bootstrap replicates (more = more accurate CIs, but slower)
- `--permutation_M 1000`: Permutation test replicates (set to 0 to disable)
- `--save_npz`: Save intermediate arrays for debugging/analysis

## Expected Runtime

- **Manifest generation**: ~1-2 minutes
- **Model inference**: ~10-30 minutes (depends on GPU and number of test segments)
- **Bootstrap**: ~1-2 minutes (CPU-only, fast)
- **Total**: ~15-35 minutes

## Troubleshooting

1. **CUDA out of memory**: Reduce `--batch_size` (e.g., 32 or 16)
2. **Import errors**: Make sure you're in the project root directory
3. **File not found**: Check that paths are correct relative to project root
4. **Slow data loading**: Increase `--num_workers` if you have more CPU cores

## Notes

- The evaluation runs inference once on all test segments, then performs bootstrap resampling
- Bootstrap is fast (CPU-only) and doesn't require GPU
- All three models (CNN, ResNet, LaBraM) will be evaluated automatically
- Results include 95% confidence intervals for all metrics and their differences from random baseline
