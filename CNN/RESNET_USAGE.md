# Running ResNet Age Classification Experiment

## Prerequisites

**Note**: The ResNet module is required. If the ResNet module is not available, the ModelFactory will raise an `ImportError` when attempting to import ResNet models.

## Quick Start

Run ResNet18 for age classification with default settings (1s segments, 50 epochs):

```bash
cd /home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/CNN
python run_resnet_age.py
```

## Command Line Options

```bash
python run_resnet_age.py [OPTIONS]
```

### Options:

- `--mode`: EEG segment length (`1s`, `2s`, or `4s`) - Default: `1s`
- `--resnet_type`: ResNet architecture type (`18`, `34`, or `50`) - Default: `18`
- `--epochs`: Number of training epochs - Default: `50`
- `--learning_rate`: Learning rate - Default: `0.0001`
- `--batch_size`: Batch size (auto-selected if not specified)
- `--num_gpus`: Number of GPUs to use - Default: `2`
- `--random_seed`: Random seed - Default: `42`
- `--results_dir`: Directory to save results - Default: `experiment_results`
- `--reports_dir`: Directory to save reports - Default: `reports`

## Examples

### Run with 1s segments (default)
```bash
python run_resnet_age.py
```

### Run with 2s segments, ResNet34, 100 epochs
```bash
python run_resnet_age.py --mode 2s --resnet_type 34 --epochs 100
```

### Run with ResNet50
```bash
python run_resnet_age.py --resnet_type 50
```

### Run with custom learning rate and batch size
```bash
python run_resnet_age.py --learning_rate 0.001 --batch_size 256
```

### Run with single GPU
```bash
python run_resnet_age.py --num_gpus 1
```

## What the Experiment Does

1. **Training**: 
   - Trains ResNet model (18, 34, or 50) on age classification (3 classes: <8.5, 8.5-12.5, >12.5 years)
   - Uses both active and passive task data for training
   - Saves best model checkpoint based on validation loss

2. **Evaluation**:
   - Evaluates on overall test set
   - **Separately evaluates on active tasks** and reports accuracy
   - **Separately evaluates on passive tasks** and reports accuracy
   - Generates confusion matrices and detailed metrics

3. **Outputs**:
   - Model checkpoints saved in `experiment_results/age_resnet{type}_{mode}/checkpoints/`
   - Evaluation results saved in `experiment_results/age_resnet{type}_{mode}/`
   - Reports generated in `reports/`
   - Task-specific metrics (active/passive) included in results

## Expected Output

The script will:
- Print training progress for each epoch
- Show validation loss and accuracy during training
- Display overall test set metrics
- **Show separate accuracy for active and passive tasks**
- Save all results to JSON files
- Generate visualization plots

## Model Architecture

- **Model**: ResNet18, ResNet34, or ResNet50 (configurable via `--resnet_type`)
- **ResNet18**: 8 BasicBlocks arranged as [2,2,2,2], 512 final channels
- **ResNet34**: 16 BasicBlocks arranged as [3,4,6,3], 512 final channels
- **ResNet50**: 16 BottleneckBlocks arranged as [3,4,6,3], 2048 final channels
- **Input**: EEG data (60 channels × timepoints)
- **Output**: 3 classes for age classification
- **Features**: Skip connections, batch normalization, adaptive pooling

## Notes

- The experiment automatically handles different segment lengths (1s, 2s, 4s)
- Training uses early stopping with patience=10
- Best model is saved based on validation loss
- Evaluation separates active and passive tasks automatically

