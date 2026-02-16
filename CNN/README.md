# EEG Classification - Systematic CNN Framework

A comprehensive, DRY (Don't Repeat Yourself) framework for EEG classification experiments using CNN models. This framework supports both gender and age classification with systematic experiment management, comprehensive reporting, and advanced data loading strategies including cross-validation and cross-task evaluation.

## 🏗️ Architecture Overview

The framework follows a systematic, modular design with clear separation of concerns:

```
CNN/
├── main.py              # Main entry point for comprehensive experiments
├── config.py            # Centralized configuration management
├── experiment.py        # Experiment lifecycle management and execution
├── trainer.py           # Model-agnostic training with progress logging
├── evaluator.py         # Comprehensive evaluation and metrics
├── report_generator.py  # HTML, text, and visualization reports
├── models/              # Model definitions and factory
│   ├── __init__.py
│   ├── base_model.py    # Base class for all EEG CNN models
│   ├── model.py         # Specific model implementations (Gender, Age)
│   └── model_factory.py # Factory pattern for model creation
├── resnet/              # ResNet variants for EEG (see docs/ResNet.md)
│   ├── resnet_model.py  # ResNet18/34/50 and task-specific wrappers
│   ├── run_resnet_age.py
│   └── run_resnet_gender.py
├── docs/                # Supplementary documentation
│   ├── ResNet.md        # ResNet architecture, usage, implementation
│   ├── Troubleshooting.md # Gender/class collapse and imbalanced data
│   └── GPU_SETUP_CHANGES.md # GPU detection, multi-GPU setup, saliency device handling
├── checkpoints/         # Model checkpoints (organized by model type)
├── experiment_results/  # Experiment results and metrics
└── README.md            # This documentation
```

## 🚀 Key Features

### 1. **DRY Principle Implementation**
- **BaseEEGCNN**: Common functionality centralized in base class
- **Model Factory**: Systematic model creation and configuration
- **Reusable Components**: Training, evaluation, and reporting modules

### 2. **Experiment Framework**
- **Default CLI**: Runs 3 baseline experiments per segment length (gender classification, age classification, age regression). Segment length via `--mode 1s` | `2s` | `4s`.
- **Data Loaders**: Standard train/val/test from HDF5; optional stratified or oversample for gender/age (`--balance_method`).
- **Additional experiment types** (cross-validation, cross-task) are defined in code and can be enabled in `main.py`.
- **Logging**: Training progress, metrics, early stopping (configurable patience), checkpointing (best and last).

### 3. **Comprehensive Evaluation**
- **Multiple Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Confusion Matrices**: Visual and numerical confusion matrices
- **Per-Class Analysis**: Detailed performance breakdown by class
- **Model Comparison**: Side-by-side comparison of different models

### 4. **Rich Reporting**
- **HTML Reports**: Interactive web-based reports
- **Visualizations**: Accuracy comparisons, training time analysis
- **CSV Export**: Easy data analysis and further processing
- **Text Summaries**: Quick overview of experiment results

## 📊 Supported Models

### Gender Classification (2 classes)
- **EEGGenderCNN**: Binary classification (Female/Male)
- **Target**: `gender` field from dataset
- **Classes**: 0 (Female), 1 (Male)

### Age Classification (3 classes)
- **EEGAgeCNN**: Multi-class classification
- **Target**: `age` field from dataset
- **Classes**: 0 (<8.5 years), 1 (8.5-12.5 years), 2 (>12.5 years)

## 🛠️ Usage

### Quick Start

Run from the **project root (EEG)** or from **CNN/**:

```bash
# Run 3 baseline experiments for 4s segments (gender, age classification, age regression)
python CNN/main.py --mode 4s
# Or from CNN/:  python main.py --mode 4s

# Other segment lengths
python CNN/main.py --mode 1s
python CNN/main.py --mode 2s

# Optional: balance method for gender/age (stratified batches or oversampling)
python CNN/main.py --mode 4s --balance_method stratified --results_dir CNN_4s_stratified
python CNN/main.py --mode 4s --balance_method oversample --results_dir CNN_4s_oversample

# Optional: custom epochs, learning rate, batch size
python CNN/main.py --mode 4s --epochs 100 --learning_rate 0.00005 --batch_size 64
```

**CLI arguments:** `--mode` (1s | 2s | 4s; default: 1s), `--epochs`, `--learning_rate`, `--batch_size`, `--random_seed`, `--num_gpus`, `--results_dir`, `--reports_dir`, `--eval_only`, `--aggregate_by_participant`, `--balance_method` (optional: `stratified` | `oversample`; default: None).

### Default experiments (per segment length)

The default run executes **3 experiments**:

1. **gender_baseline_{1s|2s|4s}** — Gender classification (2 classes)
2. **age_classification_{1s|2s|4s}** — Age classification (3 classes)
3. **age_regression_{1s|2s|4s}** — Age regression (continuous)

Additional experiment types (e.g. cross-validation, cross-task) are implemented in `main.py` and can be enabled by uncommenting the corresponding block.

### Re-evaluate with participant-level aggregation (no retraining)

You can re-run **evaluation only** on an already-trained model and get **participant-level** metrics by aggregating segment predictions per participant. No retraining; the saved checkpoint is loaded.

```bash
# From the project root (EEG) or from CNN/
python CNN/main.py --mode 4s --eval_only --aggregate_by_participant majority_vote --results_dir experiment_results

# Or from inside CNN/:
python main.py --mode 4s --eval_only --aggregate_by_participant majority_vote --results_dir experiment_results
```

- `--eval_only`: skip training, load checkpoint from `results_dir/checkpoints/<experiment_name>_best.pth`.
- `--aggregate_by_participant majority_vote`: aggregate segment predictions per participant. **Classification:** confidence-weighted majority vote (each segment’s vote weighted by the probability it assigned to its predicted class; when probabilities are not available, plain mode with tie-break). **Regression:** median of predicted values per participant.

Evaluation reports three metrics for comparison: **segment-level** (one prediction per window), **participant (mean probability)** (mean of softmax probs over segments then argmax), and **participant (majority vote)** (confidence-weighted majority as above). On some datasets mean probability can be slightly higher than majority vote; both are reported for transparency.

Use the same `--mode` (e.g. `4s`) and `--results_dir` as for the original run. Keep the whole command on one line (no line break inside `--results_dir ...`).

### ResNet experiments (gender / age)

ResNet18, ResNet34, and ResNet50 for EEG are run via dedicated scripts (not `main.py`). Run from **project root (EEG)** or from **CNN/**:

```bash
# From project root (EEG):
python -m CNN.resnet.run_resnet_age --mode 4s --resnet_type 34
python -m CNN.resnet.run_resnet_gender --mode 4s --resnet_type 18

# From CNN/:
python -m resnet.run_resnet_age --mode 4s --resnet_type 34
python -m resnet.run_resnet_gender --mode 4s --resnet_type 18
```

Options: `--mode` (1s/2s/4s), `--resnet_type` (18/34/50), `--epochs`, `--learning_rate`, `--batch_size`, `--num_gpus`, `--results_dir`, `--reports_dir`. Full usage, architecture, and API: **[docs/ResNet.md](docs/ResNet.md)**.

### Advanced Usage

```python
from CNN.experiment import Experiment
from CNN.config import ExperimentConfig, DataConfig, ModelConfig, TrainingConfig, SystemConfig

# Create custom experiment
data_config = DataConfig(
    hdf5_dir="/path/to/processed_eeg_data_hdf5",
    segment_length=800,
    batch_size=128,
    task_type="both",
)

model_config = ModelConfig(num_channels=60, dropout_rate=0.3)
training_config = TrainingConfig(
    epochs=100,
    learning_rate=0.01,
    target_key="gender",
    balance_method="stratified",  # or "oversample" or None
)

config = ExperimentConfig(
    name="my_custom_experiment",
    model_type="gender_cnn",
    target_type="gender",
    data_config=data_config,
    model_config=model_config,
    training_config=training_config,
)

# Run experiment (setup() creates loaders internally from data_config)
system_config = SystemConfig(results_dir="my_results")
experiment = Experiment(config, system_config)
experiment.setup()
result = experiment.run()
```

## 📈 Data Loading Strategies

The **default CLI** uses the standard train/val/test split (strategy 1). Other strategies are used when the corresponding experiment types are enabled in `main.py`.

### 1. Standard Train/Val/Test Split
- **Usage**: Default baseline experiments (gender, age classification, age regression)
- **Method**: `EEGDataLoader.create_train_val_test_loaders()` (from `data_processing`)
- **Purpose**: Train on both task types (active + passive), evaluate on standard splits. Optional `--balance_method stratified` or `oversample` for gender/age.

### 2. Cross-Validation
- **Method**: `create_cross_validation_loaders()` (when CV experiments are enabled)
- **Purpose**: 5-fold CV with stratification

### 3. Cross-Task Evaluation
- **Method**: `create_cross_task_loaders()`
- **Purpose**: Train on one task type, evaluate on another (active ↔ passive)

### 4. Cross-Task Cross-Validation
- **Method**: `create_cross_task_cross_validation_loaders()`
- **Purpose**: Cross-task and cross-validation combined

### 5. Imbalanced data (gender/age classification)

For imbalanced classes, the framework supports:

- **Class weights (default when not using oversample)**: Inverse-frequency weights are computed from the training HDF5 data and applied in the loss. Use `class_weight_power` in config (e.g. 1.5) to upweight minorities more.
- **Stratified batching** (`--balance_method stratified`): Each training batch has (roughly) equal counts per class; participant round-robin within class for diversity. Class weights can still be used.
- **Oversampling** (`--balance_method oversample`): Sampling with replacement so each class is seen in proportion to inverse frequency; class weights are not set (balance via data).

```bash
# Stratified batches (recommended for strong per-batch balance)
python main.py --mode 4s --balance_method stratified --results_dir CNN_4s_stratified

# Oversampling (balance via sampling; no class weights in loss)
python main.py --mode 4s --balance_method oversample --results_dir CNN_4s_oversample
```

When using stratified or oversample, the trainer does not abort on temporary model collapse (all predictions to one class); validation collapse is also allowed to recover. See [docs/Troubleshooting.md](docs/Troubleshooting.md) for background.

## 📋 Configuration

### Data Configuration
```python
data_config = DataConfig(
    hdf5_dir="/path/to/processed_eeg_data_hdf5",  # Default CLI uses paths from CNN/utils.py DATA_PATHS
    segment_length=800,  # 200=1s, 400=2s, 800=4s
    batch_size=128,
    num_workers=4,
    random_seed=42,
    task_type="both",  # "active", "passive", or "both"
    use_cross_validation=True,
    n_folds=5,
    cv_strategy="stratified",  # "stratified", "kfold", "group"
    train_task_type="active",  # "active" or "passive" (for cross-task)
    val_test_task_type="passive"  # "active" or "passive" (for cross-task)
)
```

### Model Configuration
```python
model_config = ModelConfig(
    num_channels=60,
    dropout_rate=0.5,
    use_layer_norm=True
)
```

### Training Configuration
```python
training_config = TrainingConfig(
    epochs=50,
    learning_rate=0.001,
    patience=10,
    min_delta=1e-4,
    save_every=5,
    target_key="gender",  # "gender", "age", "combined", "multi_output"
    prediction_type="classification",  # "classification" or "regression"
    balance_method="stratified",  # None | "stratified" | "oversample" (gender/age only)
    # class_weight: computed from train data when balance_method != "oversample"; set explicitly to override
    # class_weight_power: exponent for inverse-frequency weights (default 1.0; >1 upweights minority more)
)
```

### System Configuration
```python
system_config = SystemConfig(
    results_dir='experiment_results',
    reports_dir='reports',
    device='auto',  # 'auto', 'cuda', or 'cpu'
    num_gpus=1,     # or None for auto-detect
    verbose=True
)
```

## 📊 Output Structure

Outputs are under `--results_dir` (default: `experiment_results`) and `--reports_dir` (default: `reports`):

```
<results_dir>/
├── checkpoints/                    # Best and last checkpoints per experiment
│   ├── gender_baseline_4s_best.pth
│   ├── gender_baseline_4s_last.pth
│   ├── age_classification_4s_best.pth
│   ├── age_regression_4s_best.pth
│   └── ...
├── gender_baseline_4s/             # Per-experiment results
│   ├── gender_baseline_4s_result.json
│   ├── gender_baseline_4s_training_metrics.json
│   └── (evaluation outputs, confusion matrices, etc.)
├── age_classification_4s/
├── age_regression_4s/
├── experiment_summary.json
└── model_comparison.json

<reports_dir>/
├── experiment_summary.txt
├── experiment_report.html
├── experiment_results.csv
└── (visualizations)
```

For `--eval_only`, the checkpoint path is `<results_dir>/checkpoints/<experiment_name>_best.pth`.

## 💾 Checkpoint Management

Checkpoints are stored under `<results_dir>/checkpoints/` with one file per experiment: `<experiment_name>_best.pth` and optionally `<experiment_name>_last.pth`. Example: `experiment_results/checkpoints/gender_baseline_4s_best.pth`.

### Loading Checkpoints
```python
# Checkpoints are saved under results_dir/checkpoints/<experiment_name>_best.pth
# Example: experiment_results/checkpoints/gender_baseline_4s_best.pth
checkpoint_path = "experiment_results/checkpoints/gender_baseline_4s_best.pth"

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
# Optional: checkpoint also contains 'optimizer_state_dict', 'scheduler_state_dict', 'config', 'model_info'
```

### Checkpoint contents
Each checkpoint contains: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `best_val_loss`, `config`, `model_info` (and possibly other keys depending on the trainer).

## 🔧 Extending the Framework

### Adding New Models

1. **Create Model Class**:
```python
class MyCustomCNN(BaseEEGCNN):
    def _build_conv_layers(self):
        return nn.ModuleList([
            # Your custom layers
        ])
```

2. **Register with Factory**:
```python
ModelFactory.register_model('my_custom_cnn', MyCustomCNN)
```

### Adding New Experiment Types

1. **Define Configuration**:
```python
config = ExperimentConfig(
    name="my_experiment",
    model_type="my_custom_cnn",
    target_type="gender",
    data_config=DataConfig(...),
    model_config=ModelConfig(...),
    training_config=TrainingConfig(...)
)
```

2. **Add to Comprehensive Experiments**:
```python
# In main.py, add to create_comprehensive_experiments()
experiments.append(config)
```

### Adding New Data Loading Strategies

1. **Create Data Loader Function**:
```python
def create_my_custom_loaders(data_config):
    # Your custom data loading logic
    return train_loader, val_loader, test_loader
```

2. **Update create_specialized_data_loaders**:
```python
# In main.py, add new condition
elif data_config.my_custom_condition:
    return create_my_custom_loaders(data_config)
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all modules are in the Python path
2. **CUDA Issues**: Check GPU availability and memory
3. **Data Loading**: Verify data paths and preprocessing pipeline
4. **Memory Issues**: Reduce batch size or use gradient accumulation

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with minimal configuration
config = TrainingConfig(epochs=1, save_every=1)
```

## 📚 Dependencies

- PyTorch >= 1.9.0
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pandas

## 🤝 Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive docstrings and type hints
3. Include unit tests for new functionality
4. Update documentation for new features

## 📊 Experiment results and optional experiment types

The **default** run executes **3 experiments** per segment length (gender, age classification, age regression). Results are written under `--results_dir`: `experiment_summary.json`, `model_comparison.json`, per-experiment folders with metrics, checkpoints, and (when reports are generated) training plots and confusion matrices.

**Additional experiment types** (e.g. cross-validation, cross-task, combined, multi-output) are implemented in `main.py` but are **commented out** by default. To enable them, uncomment the corresponding blocks in `create_all_experiments_for_segment_length()` in `CNN/main.py`. Those runs produce the same result structure with more experiment entries.

## 📚 Documentation

| Document | Description |
|---------|-------------|
| [README.md](README.md) | This file — framework overview, usage, config |
| [docs/ResNet.md](docs/ResNet.md) | ResNet models: architecture, layer counts, usage, scripts |
| [docs/Troubleshooting.md](docs/Troubleshooting.md) | Gender/class collapse, imbalanced data, class weights, `--balance_method` |
| [docs/GPU_SETUP_CHANGES.md](docs/GPU_SETUP_CHANGES.md) | GPU detection, multi-GPU setup, gpu_utils, saliency device handling |

## 📄 License

This project follows the same license as the parent EEG classification project.

---

**Note**: This framework is designed to be extensible and maintainable. The modular architecture allows for easy addition of new models, experiments, and evaluation metrics while maintaining the DRY principle and systematic approach to machine learning experiments. The comprehensive experiment framework provides robust evaluation across multiple data loading strategies and experimental conditions.
