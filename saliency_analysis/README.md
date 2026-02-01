# EEG Channel Importance Analysis via Saliency Maps

This module provides tools for analyzing which EEG channels are most important for age and gender predictions using gradient-based saliency methods.

## Table of Contents

1. [Overview](#overview)
2. [Methods Implemented](#methods-implemented)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Run Scripts](#run-scripts)
6. [Understanding Results](#understanding-results)
7. [Topography and Additional Scripts](#topography-and-additional-scripts)
8. [Best Practices](#best-practices)
9. [Technical Details](#technical-details)
10. [Progress and GPU Support](#progress-and-gpu-support)
11. [Troubleshooting](#troubleshooting)
12. [References](#references)

## Overview

Saliency maps are visualization techniques that highlight which parts of the input (in this case, which EEG channels) contribute most to the model's predictions. This analysis helps:

1. **Interpretability**: Understand which brain regions/channels the model relies on
2. **Feature Selection**: Identify redundant or less important channels
3. **Model Validation**: Verify that the model uses physiologically meaningful channels
4. **Research Insights**: Discover patterns in channel importance across different tasks

### Participant Aggregation

**By default, the saliency analysis generates one aggregated saliency map across all participants.** This approach:

- **Provides population-level insights**: Shows which channels are generally important across the entire dataset
- **Reduces participant-specific noise**: Averages out individual variations to reveal common patterns
- **Is more generalizable**: The aggregated map represents patterns that apply broadly, not just to specific participants

The implementation:
- Tracks participant IDs for each sample during saliency computation
- Aggregates channel importance by computing the mean across all samples (all participants)
- Reports the number of unique participants and total samples used in the aggregation

This is the recommended approach for understanding general channel importance patterns. For participant-specific analysis, the participant IDs are preserved in the results and can be used for further analysis.

## Methods Implemented

### 1. Vanilla Gradients (Default for Fast Testing)

Computes the gradient of the model output with respect to the input features. Simple and fast (~50x faster than Integrated Gradients), but can be noisy.

**Formula:**
```
Saliency(x) = |∇F(x)|
```

**Characteristics:**
- Fast computation: Only requires one forward and one backward pass
- Can be noisy and sensitive to small input changes
- Values are always non-negative (if `abs_gradients=True`)

### 2. Integrated Gradients (Recommended for Production)

More robust attribution method that integrates gradients along the path from a baseline (zero) to the actual input. This method satisfies important axioms for attribution and provides more reliable results.

**Formula:**
```
IG(x) = (x - x') × (1/m) × Σ[i=1 to m] ∇F(x' + (i/m)(x - x'))
```

**Characteristics:**
- More robust: Less sensitive to noise and local variations
- Theoretically grounded: Satisfies important axioms (sensitivity, implementation invariance)
- More computationally expensive: Requires `num_steps` forward/backward passes (default: 50)
- Can have negative values: Negative values indicate features that reduce the prediction

**Reference**: Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017

**Note**: The default method is set to `vanilla_gradients` for faster testing. Change to `integrated_gradients` for more accurate results.

## Installation

The module uses existing dependencies from the CNN and data_processing modules. No additional installation required.

**Optional**: Install `tqdm` for better progress bars:
```bash
pip install tqdm
```

## Usage

### Command Line Interface

The easiest way to run saliency analysis is using the main script:

```bash
# Single method (vanilla_gradients - fast default)
python saliency_analysis/main.py \
    --checkpoint CNN/report_2025-12-16/results_1s/checkpoints/gender_baseline_1s_best.pth \
    --target_type gender \
    --segment_length 1s \
    --hdf5_dir data_processing/processed_eeg_data_hdf5_no_compression \
    --split test \
    --method vanilla_gradients \
    --max_samples 1000 \
    --output_dir saliency_results/gender_analysis

# Compare both methods
python saliency_analysis/main.py \
    --checkpoint CNN/report_2025-12-16/results_1s/checkpoints/gender_baseline_1s_best.pth \
    --target_type gender \
    --segment_length 1s \
    --hdf5_dir data_processing/processed_eeg_data_hdf5_no_compression \
    --split test \
    --method both \
    --max_samples 1000 \
    --output_dir saliency_results/gender_analysis_comparison
```

### Arguments

**Required:**
- `--checkpoint`: Path to trained model checkpoint
- `--target_type`: Type of prediction (`gender`, `age`, or `combined`)
- `--segment_length`: EEG segment length (`1s`, `2s`, or `4s`)

**Data Options:**
- `--hdf5_dir`: Directory containing HDF5 files (default: `data_processing/processed_eeg_data_hdf5_no_compression`)
- `--split`: Data split to analyze (`train`, `val`, or `test`)
- `--task_type`: Task type (`active`, `passive`, or `both`)

**Model Options:**
- `--model_type`: Model type (auto-detected if not specified)
- `--num_channels`: Number of EEG channels (default: 60)
- `--prediction_type`: For age models (`classification` or `regression`)

**Saliency Options:**
- `--method`: Saliency method (`vanilla_gradients`, `integrated_gradients`, or `both` to compare both methods)
- `--num_steps`: Number of steps for integrated gradients (default: 50)
- `--max_samples`: Maximum samples to process (None = all)

**Output Options:**
- `--output_dir`: Output directory for results
- `--top_k`: Number of top channels to highlight (default: 10)
- `--no_plots`: Skip generating visualization plots

### Python API

Run from the project root (EEG/) so that `CNN`, `data_processing`, and `saliency_analysis` are on the path. You can use the analyzer programmatically:

```python
import sys
from pathlib import Path
# Add project root
root = Path(__file__).resolve().parent.parent  # or Path(".") if run from EEG/
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
if str(root / "CNN") not in sys.path:
    sys.path.insert(0, str(root / "CNN"))

from saliency_analysis.analyzer import SaliencyAnalyzer
from CNN.models import ModelFactory
from CNN.config import ModelConfig
from data_processing.eeg_dataset import EEGDataLoader

# Create model (match checkpoint: e.g. gender_cnn, age_cnn, age_regression_cnn)
model_config = ModelConfig(num_channels=60, num_classes=2)
model = ModelFactory.create_model('gender_cnn', model_config)

# Create analyzer
analyzer = SaliencyAnalyzer(model, target_type='gender', num_channels=60)

# Load checkpoint
analyzer.load_checkpoint('path/to/checkpoint.pth')

# Load data
train_loader, val_loader, test_loader = EEGDataLoader.create_train_val_test_loaders(
    hdf5_dir='data_processing/processed_eeg_data_hdf5_no_compression',
    segment_length='1s',
    batch_size=32
)

# Compute saliency
results = analyzer.compute_saliency(
    data_loader=test_loader,
    method='vanilla_gradients',
    max_samples=1000
)

# Analyze channel importance
analysis = analyzer.analyze_channel_importance(method='vanilla_gradients')

# Get top channels
top_channels = analyzer.get_top_channels(method='vanilla_gradients', top_k=10)

# Generate visualizations (includes topography if MNE is available)
analyzer.visualize_results(
    output_dir='saliency_results',
    method='vanilla_gradients',
    top_k=10
)
```

## Run Scripts

### 1. `run_saliency_batch.sh` - Batch Mode (All Models)

Runs saliency analysis for all configured best models. For a subset of models, call `main.py` directly with `--batch --models <name1> <name2>`.

**Usage:**
```bash
bash saliency_analysis/run_saliency_batch.sh
```

**Configuration via Environment Variables:**
```bash
export HDF5_DIR="data_processing/processed_eeg_data_hdf5_no_compression"
export OUTPUT_DIR="saliency_results"
export DEVICE="cuda"
export NUM_GPUS="1"
export METHOD="vanilla_gradients"  # Fast default, or "integrated_gradients" or "both"
export BATCH_SIZE="128"
export MAX_SAMPLES=""  # Empty for all samples (recommended for proper aggregation)
export SPLIT="test"
export TASK_TYPE="both"
export NUM_STEPS="50"
export TOP_K="10"

bash saliency_analysis/run_saliency_batch.sh
```

### 2. `run_saliency_single.sh` - Single Model

Runs saliency analysis for a single model.

**Usage:**
```bash
bash saliency_analysis/run_saliency_single.sh \
    --checkpoint CNN/report_2025-12-16/results_1s/checkpoints/gender_baseline_1s_best.pth \
    --target_type gender \
    --segment_length 1s \
    --method vanilla_gradients \
    --max_samples 500
```

### 3. `run_saliency_slurm.sh` - SLURM Batch Job

SLURM script for submitting batch jobs to a cluster.

**Usage:**
```bash
sbatch saliency_analysis/run_saliency_slurm.sh
```

## Understanding Results

### Output Files

The analysis generates several output files:

#### 1. Saliency Results (`saliency_results_<method>.npz`)
NumPy compressed file containing:
- `saliency_maps`: Full saliency maps (num_samples, num_channels, timepoints)
- `channel_importance`: Aggregated channel importance (num_samples, num_channels)
- `labels`: True labels
- `predictions`: Model predictions
- `participant_ids`: Participant IDs for each sample

#### 2. Visualizations

- **`average_importance_<method>.png`**: Bar chart showing average importance per channel
- **`importance_distribution_<method>.png`**: Box plots showing distribution of importance values
- **`importance_by_class_<method>.png`**: Comparison of channel importance across classes
- **`saliency_aggregated_all_participants_<method>.png`**: **Aggregated saliency map across all participants** (mean across all samples) - This is the main saliency map representing the population
- **`saliency_consistency_all_participants_<method>.png`**: Consistency map (standard deviation) showing variance across participants
- **`saliency_sample_<N>_<method>.png`**: Full saliency maps for individual samples (for reference, shows 3 samples)
- **`channel_ranking_<method>.csv`**: CSV file with ranked channels and statistics
- **`topography_<method>.png`**: Scalp topography (channel importance on a head map), generated when MNE is available. See [Topography and Additional Scripts](#topography-and-additional-scripts).

**When using `--method both`:**
- **`method_comparison.png`**: Comprehensive comparison visualization with 4 subplots
- **`method_comparison_report.csv`**: Detailed CSV report comparing both methods with statistics

### Channel Importance Values

Higher importance values indicate that changes to that channel have a larger impact on the model's prediction. The exact interpretation depends on the saliency method:

- **Vanilla Gradients**: Magnitude of gradient (how much output changes per unit change in input)
- **Integrated Gradients**: Integrated gradient value (attribution score)

### Channel Ranking

Channels are ranked by their average importance across all samples. The ranking helps identify:
- **Critical channels**: Top-ranked channels that are consistently important
- **Redundant channels**: Low-ranked channels that may be removed without significant performance loss

### Class-Specific Analysis

For classification tasks, the analysis compares channel importance across different classes. This reveals:
- **Class-specific channels**: Channels important for one class but not others
- **Universal channels**: Channels important across all classes

## Topography and Additional Scripts

### Topography (scalp maps)

Saliency results can be visualized as topography plots (EEG channel importance on a scalp map) using MNE's 10–20 montage. Topography is **automatically generated** when you run saliency analysis if MNE is installed (e.g. in the `eeg_env` conda environment).

For details, saved-results workflows, and custom data, see **[TOPOGRAPHY_GUIDE.md](TOPOGRAPHY_GUIDE.md)**.

### Additional scripts

- **`create_topography.py`** – Interactive script to create topography plots from saved saliency `.npz` results or custom channel importance data.
- **`example_topography_heatmap.py`** – Example script demonstrating topography plotting with MNE.
- **`example_usage.py`** – Example usage of the saliency analyzer and visualization pipeline.

## Best Practices

1. **Use Integrated Gradients for production**: More robust than vanilla gradients
2. **Use Vanilla Gradients for testing**: ~50x faster for quick exploration
3. **Process all test samples**: For proper aggregation, set `MAX_SAMPLES=""` to process all samples
4. **Analyze Multiple Samples**: Channel importance can vary, so analyze a representative sample
5. **Compare Across Classes**: For classification, compare importance across different classes
6. **Validate with Domain Knowledge**: Check if important channels align with neurophysiological expectations
7. **Consider Task Type**: Analyze separately for active vs. passive tasks if relevant

## Technical Details

### Batches vs Samples

**Terminology:**
- **Sample**: One individual EEG segment (one participant's data at one time point)
- **Batch**: A group of samples loaded together (e.g., 32 or 128 samples per batch)

**Relationship:**
```
Total Samples = Number of Batches × Batch Size
```

**Progress Bar:**
- When `max_samples` is specified: Tracks by samples (`samples=X/Y, batches=Z`)
- When processing all samples: Tracks by batches (`samples=X, batches=Y`)

The samples count increases by `batch_size` each batch, which is expected behavior.

### Participant Aggregation

**Default Behavior:**
- Tracks participant IDs for each sample during saliency computation
- Generates one aggregated saliency map across all participants (by default)
- Reports participant statistics (number of unique participants, samples per participant)

**Why Aggregate:**
- **Population-level insights**: Shows which channels are generally important
- **Generalizability**: Represents patterns that apply broadly
- **Reduced noise**: Averages out individual variations
- **Statistical robustness**: More samples = more reliable estimates

**Accessing Participant Information:**
```python
results = analyzer.compute_saliency(data_loader, method='vanilla_gradients')
participant_ids = results['participant_ids']  # Array of participant IDs for each sample
```

### Saliency Aggregation Method

**Current Implementation (Best Practice):**

1. **Normalize per sample**: L2 normalization of each sample's saliency map before aggregation
   - Prevents samples with high-magnitude gradients from dominating
   - Ensures each sample contributes equally
   - Formula: `normalized_map = map / ||map||_2` (if norm > 0)

2. **Aggregate across samples**: Mean aggregation of normalized saliency maps
   - Formula: `aggregated_map = mean(normalized_maps, axis=0)`

3. **Compute consistency**: Standard deviation map shows variance across participants
   - Low variance = consistent pattern, high variance = variable pattern

**Why This Approach:**
- ✅ Prevents magnitude bias
- ✅ Reveals population-level patterns
- ✅ Shows consistency across participants
- ✅ Follows literature best practices

### Absolute vs Signed Values

**Current Implementation:**
- **Computation**: `abs_gradients=True` by default (computes absolute values)
- **Visualization**: Shows absolute values (standard practice for aggregated maps)
- **Colormap**: `viridis` (sequential) for absolute values

**Why Absolute Values:**
- ✅ **Standard practice** in literature for aggregated maps
- ✅ **Avoids cancellation** when aggregating across samples
- ✅ **Clearer visualization** of importance magnitude
- ✅ **Better for aggregation** (averaging makes sense)

**Signed Values (Alternative):**
- Useful for individual samples to understand directionality
- Shows which features increase vs decrease prediction
- Not recommended for aggregated maps (cancellation issues)

### Method Comparison

**Key Differences:**

| Aspect | Vanilla Gradients | Integrated Gradients |
|--------|------------------|---------------------|
| **Speed** | ~50x faster | Slower (50 steps per sample) |
| **Robustness** | Can be noisy | More stable |
| **Theoretical Grounding** | Simple gradient | Axiomatically grounded |
| **Values** | Always ≥ 0 (if abs) | Can be negative or positive |
| **Use Case** | Quick exploration | Production/publication |

**When to Use:**
- **Vanilla Gradients**: Fast testing, quick exploration
- **Integrated Gradients**: Production analysis, publications, when accuracy matters

## Progress and GPU Support

### Progress Indicators

The saliency analysis includes comprehensive progress tracking:

1. **Sample-Level Progress**: Progress bar showing samples processed, batches, and processing rate
2. **Batch-Level Progress**: Nested progress bars for large batches
3. **Model-Level Progress**: When processing multiple models in batch mode
4. **Fallback Progress**: Simple print statements if `tqdm` is not installed

**Example Output:**
```
Computing saliency: 100%|████████████| 1000/1000 [05:23<00:00, 3.10samples/s, samples=1000, batches=32]
```

### Multi-GPU Support

**✅ Yes, Multi-GPU is Supported!**

The saliency analysis fully supports multiple GPUs using PyTorch's `DataParallel`.

**Usage:**
```bash
# Single GPU
export NUM_GPUS=1
bash saliency_analysis/run_saliency_batch.sh

# Multiple GPUs
export NUM_GPUS=4
bash saliency_analysis/run_saliency_batch.sh
```

**Performance Benefits:**
- 2 GPUs: ~1.7-1.9x faster
- 4 GPUs: ~3.0-3.5x faster
- 8 GPUs: ~5.0-6.0x faster

**GPU Information Display:**
The code automatically displays GPU information and confirms multi-GPU setup.

### SLURM Usage

**Interactive Session:**
```bash
salloc --gres=gpu:4 --time=24:00:00 --mem=64G
export NUM_GPUS=4
bash saliency_analysis/run_saliency_batch.sh
```

**Batch Job:**
```bash
sbatch saliency_analysis/run_saliency_slurm.sh
```

## Troubleshooting

### Process Hanging or Not Showing Progress

**Symptoms:**
- Log shows "Computing saliency..." and then nothing for hours
- Process may have crashed silently or is stuck

**Solutions:**

1. **Progress Bar Not Flushing**: ✅ Fixed - Progress bar now writes to stdout
2. **Data Loading Hanging**: ✅ Fixed - `num_workers=0` by default to avoid multiprocessing issues
3. **First Sample Taking Very Long**: Normal for Integrated Gradients (50 steps per sample)
4. **Silent Crash**: Check error logs in output directory

**What to Check:**
```bash
# Check if process is running
ps aux | grep -i "saliency\|python.*main.py" | grep -v grep

# Check GPU usage
nvidia-smi
# If GPU is at 0% utilization, process is likely stuck

# Check memory usage
free -h

# Test with limited samples
export MAX_SAMPLES=10
bash saliency_analysis/run_saliency_batch.sh
```

**Expected Behavior:**
- Immediate output (within seconds)
- First batch loaded within 1-2 minutes
- Progress updates every few seconds

### Out of Memory Errors

- Reduce `--max_samples` to process fewer samples
- Use smaller batch size in data loader
- Process samples in batches
- Use fewer GPUs

### Model Type Detection Fails

- Explicitly specify `--model_type` argument
- Check checkpoint file contains model metadata

### No Results Generated

- Verify checkpoint path is correct
- Check HDF5 files exist in specified directory
- Ensure data split matches available files

### Performance Issues

**If taking much longer than expected:**
- Check GPU utilization: `watch -n 1 nvidia-smi` (should be >80%)
- Verify progress bar is updating regularly
- Check for memory swapping (very slow)

**Performance Expectations:**
- **Vanilla Gradients**: ~0.1-0.2 seconds per sample
- **Integrated Gradients**: ~2-5 seconds per sample
- **1000 samples**: ~2-5 minutes (VG) or ~30-60 minutes (IG)
- **All test samples**: ~30-60 minutes (VG) or ~5-10 hours (IG)

## References

1. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. ICML.
2. Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034.

## License

Same as parent project.
