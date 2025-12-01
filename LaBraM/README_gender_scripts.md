# LaBraM Gender Classification Scripts

This directory contains shell scripts for fine-tuning LaBraM models on gender classification tasks.

## Scripts Overview

### 1. `test_gender_setup.sh` - Quick Test
**Purpose**: Verify the setup works before running full experiments
**Usage**: 
```bash
./test_gender_setup.sh
```
**Features**:
- Runs only 2 epochs with small batch size
- Tests basic functionality
- Quick verification (~5-10 minutes)

### 2. `run_gender_baseline.sh` - Single Baseline Experiment
**Purpose**: Run gender baseline classification experiment
**Usage**:
```bash
./run_gender_baseline.sh
```
**Features**:
- Uses `gender_baseline` dataset
- 30 epochs training
- Full LaBraM-Base hyperparameters
- Single experiment (~2-3 hours)

### 3. `run_gender_experiments.sh` - Multiple Experiments
**Purpose**: Run all gender classification experiments
**Usage**:
```bash
./run_gender_experiments.sh
```
**Features**:
- Runs 4 different experiment types:
  - `gender_baseline` - Standard train/val/test split
  - `gender_cv_gender_stratified` - 5-fold cross-validation
  - `gender_cross_task_active_to_passive` - Train on active, test on passive
  - `gender_cross_task_passive_to_active` - Train on passive, test on active
- Sequential execution (~8-12 hours total)

## Configuration Details

### Model Architecture
- **Model**: `labram_base_patch200_200` (LaBraM-Base)
- **Input Size**: 200 time points
- **Classes**: 2 (male/female)
- **Patch Size**: 200 (from model name)

### Training Hyperparameters
Based on LaBraM paper recommendations:
- **Batch Size**: 64 (reduced from paper's 512 for single GPU)
- **Epochs**: 30
- **Learning Rate**: 5e-4
- **Min Learning Rate**: 1e-6
- **Warmup Epochs**: 5
- **Weight Decay**: 0.05
- **Drop Path**: 0.1
- **Layer Decay**: 0.65 (for Base model)

### Data Configuration
- **Segment Length**: 1s (200 time points at 200Hz)
- **Data Path**: `/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/data_processing/processed_eeg_data_1s_segments`
- **Task Type**: Both active and passive tasks included

## Output Structure

Each experiment creates:
```
outputs/
├── gender_baseline_YYYYMMDD_HHMMSS/
│   ├── checkpoint.pth              # Latest checkpoint
│   ├── checkpoint-best.pth         # Best validation checkpoint
│   ├── checkpoint-5.pth            # Epoch 5 checkpoint
│   ├── checkpoint-10.pth           # Epoch 10 checkpoint
│   ├── ...                         # More checkpoints
│   ├── log.txt                     # Training logs
│   └── logs/                       # TensorBoard logs
└── ...
```

## Usage Recommendations

### 1. First Time Setup
```bash
# Test the setup
./test_gender_setup.sh

# If successful, run baseline experiment
./run_gender_baseline.sh
```

### 2. Full Experiment Suite
```bash
# Run all gender experiments
./run_gender_experiments.sh
```

### 3. Custom Experiments
You can modify the scripts to:
- Change hyperparameters
- Use different datasets
- Adjust batch size for your GPU memory
- Modify number of epochs

## Monitoring Training

### TensorBoard
```bash
# Start TensorBoard (in another terminal)
tensorboard --logdir=outputs/gender_baseline_YYYYMMDD_HHMMSS/logs
```

### Log Files
- **Training logs**: `outputs/*/log.txt`
- **Console output**: Shows real-time progress
- **Checkpoints**: Saved every 5 epochs

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce `BATCH_SIZE` in the script
2. **Data path not found**: Check if the data directory exists
3. **Import errors**: Ensure all dependencies are installed
4. **Model not found**: Check if LaBraM models are properly registered

### Performance Tips
1. **GPU Memory**: Adjust batch size based on your GPU
2. **CPU**: Increase `NUM_WORKERS` if you have more CPU cores
3. **Storage**: Ensure enough disk space for checkpoints and logs

## Expected Results

### Baseline Performance
- **Training Time**: ~2-3 hours for 30 epochs
- **Memory Usage**: ~8-12GB GPU memory
- **Expected Accuracy**: 70-85% (typical for EEG gender classification)

### Cross-Validation
- **Training Time**: ~2-3 hours per fold (5 folds total)
- **Expected Accuracy**: Similar to baseline with confidence intervals

### Cross-Task
- **Training Time**: ~2-3 hours per direction
- **Expected Accuracy**: Lower than baseline (domain shift effect)

## Next Steps

After running these experiments, you can:
1. **Analyze results** using the saved checkpoints
2. **Run age classification** experiments
3. **Run combined** age+gender experiments
4. **Run multi-output** experiments
5. **Compare with CNN** baseline results
