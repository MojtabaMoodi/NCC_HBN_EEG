# EEG Channel Importance Analysis via Saliency Maps

This module provides tools for analyzing which EEG channels are most important for age and gender predictions using gradient-based saliency methods.

## Overview

Saliency maps are visualization techniques that highlight which parts of the input (in this case, which EEG channels) contribute most to the model's predictions. This analysis helps:

1. **Interpretability**: Understand which brain regions/channels the model relies on
2. **Feature Selection**: Identify redundant or less important channels
3. **Model Validation**: Verify that the model uses physiologically meaningful channels
4. **Research Insights**: Discover patterns in channel importance across different tasks

## Methods Implemented

### 1. Vanilla Gradients
Computes the gradient of the model output with respect to the input features. Simple and fast, but can be noisy.

### 2. Integrated Gradients (Recommended)
More robust attribution method that integrates gradients along the path from a baseline (zero) to the actual input. This method satisfies important axioms for attribution and provides more reliable results.

**Reference**: Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017

## Installation

The module uses existing dependencies from the CNN and data_processing modules. No additional installation required.

## Usage

### Command Line Interface

The easiest way to run saliency analysis is using the main script:

```bash
# Single method (integrated_gradients)
python saliency_analysis/main.py \
    --checkpoint CNN/experiment_results/gender_baseline_1s/checkpoints/gender_baseline_1s_best.pth \
    --target_type gender \
    --segment_length 1s \
    --hdf5_dir data_processing/processed_eeg_data_hdf5 \
    --split test \
    --method integrated_gradients \
    --max_samples 1000 \
    --output_dir saliency_results/gender_analysis

# Compare both methods
python saliency_analysis/main.py \
    --checkpoint CNN/experiment_results/gender_baseline_1s/checkpoints/gender_baseline_1s_best.pth \
    --target_type gender \
    --segment_length 1s \
    --hdf5_dir data_processing/processed_eeg_data_hdf5 \
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
- `--hdf5_dir`: Directory containing HDF5 files (default: `data_processing/processed_eeg_data_hdf5`)
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

You can also use the analyzer programmatically:

```python
from saliency_analysis.analyzer import SaliencyAnalyzer
from models import ModelFactory
from config import ModelConfig
from eeg_dataset import EEGDataLoader

# Create model
model_config = ModelConfig(num_channels=60, num_classes=2)
model = ModelFactory.create_model('gender_cnn', model_config)

# Create analyzer
analyzer = SaliencyAnalyzer(model, target_type='gender', num_channels=60)

# Load checkpoint
analyzer.load_checkpoint('path/to/checkpoint.pth')

# Load data
train_loader, val_loader, test_loader = EEGDataLoader.create_train_val_test_loaders(
    hdf5_dir='data_processing/processed_eeg_data_hdf5',
    segment_length='1s',
    batch_size=32
)

# Compute saliency
results = analyzer.compute_saliency(
    data_loader=test_loader,
    method='integrated_gradients',
    max_samples=1000
)

# Analyze channel importance
analysis = analyzer.analyze_channel_importance(method='integrated_gradients')

# Get top channels
top_channels = analyzer.get_top_channels(method='integrated_gradients', top_k=10)

# Generate visualizations
analyzer.visualize_results(
    output_dir='saliency_results',
    method='integrated_gradients',
    top_k=10
)
```

## Output Files

The analysis generates several output files:

### 1. Saliency Results (`saliency_results_<method>.npz`)
NumPy compressed file containing:
- `saliency_maps`: Full saliency maps (num_samples, num_channels, timepoints)
- `channel_importance`: Aggregated channel importance (num_samples, num_channels)
- `labels`: True labels
- `predictions`: Model predictions

### 2. Visualizations

- **`average_importance_<method>.png`**: Bar chart showing average importance per channel
- **`importance_distribution_<method>.png`**: Box plots showing distribution of importance values
- **`importance_by_class_<method>.png`**: Comparison of channel importance across classes
- **`saliency_sample_<N>_<method>.png`**: Full saliency maps for individual samples
- **`channel_ranking_<method>.csv`**: CSV file with ranked channels and statistics

**When using `--method both`:**
- **`method_comparison.png`**: Comprehensive comparison visualization with 4 subplots:
  - Side-by-side bar comparison of top channels
  - Scatter plot showing correlation between methods
  - Difference plot (IG - VG) for top channels
  - Ranking comparison plot
- **`method_comparison_report.csv`**: Detailed CSV report comparing both methods with statistics

### 3. Console Output

The script prints:
- Top 10 most important channels with statistics
- Summary of analysis results
- When using `--method both`: Correlation coefficient and comparison statistics

## Understanding Results

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

## Best Practices

1. **Use Integrated Gradients**: More robust than vanilla gradients
2. **Analyze Multiple Samples**: Channel importance can vary, so analyze a representative sample
3. **Compare Across Classes**: For classification, compare importance across different classes
4. **Validate with Domain Knowledge**: Check if important channels align with neurophysiological expectations
5. **Consider Task Type**: Analyze separately for active vs. passive tasks if relevant

## Example Workflow

1. **Train your model** using the CNN module
2. **Run saliency analysis** on test set:
   ```bash
   python saliency_analysis/main.py \
       --checkpoint <checkpoint_path> \
       --target_type gender \
       --segment_length 1s \
       --split test
   ```
3. **Review visualizations** in the output directory
4. **Check channel ranking** CSV for quantitative results
5. **Compare with literature** to validate findings

## Troubleshooting

### Out of Memory Errors
- Reduce `--max_samples` to process fewer samples
- Use smaller batch size in data loader
- Process samples in batches

### Model Type Detection Fails
- Explicitly specify `--model_type` argument
- Check checkpoint file contains model metadata

### No Results Generated
- Verify checkpoint path is correct
- Check HDF5 files exist in specified directory
- Ensure data split matches available files

## References

1. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. ICML.
2. Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034.

## License

Same as parent project.

