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
├── checkpoints/         # Model checkpoints (organized by model type)
├── experiment_results/  # Experiment results and metrics
└── README.md           # This documentation
```

## 🚀 Key Features

### 1. **DRY Principle Implementation**
- **BaseEEGCNN**: Common functionality centralized in base class
- **Model Factory**: Systematic model creation and configuration
- **Reusable Components**: Training, evaluation, and reporting modules

### 2. **Comprehensive Experiment Framework**
- **12 Experiment Types**: Basic, cross-validation, cross-task, and cross-task cross-validation
- **Specialized Data Loaders**: Each experiment type uses appropriate data loading strategy
- **Cross-Validation Support**: 5-fold cross-validation with proper result aggregation
- **Cross-Task Evaluation**: Train on one task type, evaluate on another
- **Comprehensive Logging**: Track all metrics, training progress, and results
- **Early Stopping**: Prevent overfitting with configurable patience
- **Checkpointing**: Save best models and regular checkpoints

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

```bash
# Run all 12 comprehensive experiments
python main.py --mode comprehensive

# Run single experiment
python main.py --mode single --experiment test_gender --target gender --epochs 10

# Run default experiments (basic gender and age)
python main.py --mode default

# Run custom batch experiments
python main.py --mode batch --epochs 100 --learning_rate 0.01
```

### Comprehensive Experiment Types

The framework supports 12 different experiment types:

#### **1-2. Basic Experiments**
- `gender_baseline_train_val_test`: Standard gender classification
- `age_baseline_train_val_test`: Standard age classification

#### **3-4. Cross-Validation Experiments**
- `gender_cv_gender_stratified`: 5-fold CV with gender stratification
- `age_cv_age_stratified`: 5-fold CV with age stratification

#### **5-8. Cross-Task Experiments**
- `gender_cross_task_active_to_passive`: Train on active, test on passive
- `gender_cross_task_passive_to_active`: Train on passive, test on active
- `age_cross_task_active_to_passive`: Train on active, test on passive
- `age_cross_task_passive_to_active`: Train on passive, test on active

#### **9-12. Cross-Task Cross-Validation Experiments**
- `gender_cross_task_cv_active_to_passive_gender_stratified`: 5-fold CV with cross-task
- `gender_cross_task_cv_passive_to_active_gender_stratified`: 5-fold CV with cross-task
- `age_cross_task_cv_active_to_passive_age_stratified`: 5-fold CV with cross-task
- `age_cross_task_cv_passive_to_active_age_stratified`: 5-fold CV with cross-task

### Advanced Usage

```python
from experiment import Experiment, ExperimentConfig
from config import DataConfig, ModelConfig, TrainingConfig, SystemConfig
from models import ModelFactory

# Create custom experiment
data_config = DataConfig(
    pickle_dir="/path/to/data",
    batch_size=128,
    use_cross_validation=True,
    n_folds=5,
    cv_strategy="gender"
)

model_config = ModelConfig(num_channels=60, dropout_rate=0.3)
training_config = TrainingConfig(epochs=100, learning_rate=0.01)

config = ExperimentConfig(
    name="my_custom_experiment",
    model_type="gender_cnn",
    target_type="gender",
    data_config=data_config,
    model_config=model_config,
    training_config=training_config
)

# Run experiment
system_config = SystemConfig(results_dir="my_results")
experiment = Experiment(config, system_config)
experiment.setup(train_loader, val_loader, test_loader)
result = experiment.run()
```

## 📈 Data Loading Strategies

### 1. Standard Train/Val/Test Split
- **Usage**: Basic experiments (1-2)
- **Method**: `create_train_val_test_loaders()`
- **Purpose**: Standard machine learning evaluation

### 2. Cross-Validation
- **Usage**: Cross-validation experiments (3-4)
- **Method**: `create_cross_validation_loaders()`
- **Purpose**: Robust evaluation with 5-fold CV and stratification

### 3. Cross-Task Evaluation
- **Usage**: Cross-task experiments (5-8)
- **Method**: `create_cross_task_loaders()`
- **Purpose**: Test generalization across different task types (active ↔ passive)

### 4. Cross-Task Cross-Validation
- **Usage**: Cross-task cross-validation experiments (9-12)
- **Method**: `create_cross_task_cross_validation_loaders()`
- **Purpose**: Most robust evaluation combining cross-task and cross-validation

## 📋 Configuration

### Data Configuration
```python
data_config = DataConfig(
    pickle_dir="/path/to/data",
    batch_size=128,
    num_workers=4,
    random_seed=42,
    use_cross_validation=True,
    n_folds=5,
    cv_strategy="gender",  # "gender" or "age"
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
    target_key="gender"  # "gender" or "age"
)
```

### System Configuration
```python
system_config = SystemConfig(
    results_dir='experiment_results',
    reports_dir='reports',
    device='cuda',  # 'cuda' or 'cpu'
    verbose=True
)
```

## 📊 Output Structure

```
checkpoints/                         # Model checkpoints organized by model type
├── EEGGenderCNN/                   # Gender classification model checkpoints
│   ├── gender_cnn_baseline_best.pth
│   ├── gender_cnn_baseline_epoch_10.pth
│   └── gender_cnn_high_dropout_best.pth
├── EEGAgeCNN/                      # Age classification model checkpoints
│   ├── age_cnn_baseline_best.pth
│   ├── age_cnn_baseline_epoch_10.pth
│   └── age_cnn_high_dropout_best.pth
└── ...

experiment_results/
├── experiment_summary.json          # Overall experiment summary
├── model_comparison.json           # Model comparison results
├── gender_baseline_train_val_test/  # Basic experiment results
│   ├── gender_baseline_train_val_test_result.json
│   ├── gender_baseline_train_val_test_training_metrics.json
│   └── (other experiment-specific files)
├── gender_cv_gender_stratified/     # Cross-validation experiment (organized)
│   ├── fold_1/                     # Individual fold results
│   │   ├── gender_cv_gender_stratified_result.json
│   │   ├── gender_cv_gender_stratified_training_metrics.json
│   │   └── (other fold-specific files)
│   ├── fold_2/
│   ├── fold_3/
│   ├── fold_4/
│   ├── fold_5/
│   └── gender_cv_gender_stratified_result.json  # Aggregated results
├── gender_cross_task_cv_active_to_passive_gender_stratified/  # Cross-task CV
│   ├── fold_1/
│   ├── fold_2/
│   ├── fold_3/
│   ├── fold_4/
│   ├── fold_5/
│   └── gender_cross_task_cv_active_to_passive_gender_stratified_result.json
└── ...

reports/
├── experiment_summary.txt          # Text summary
├── experiment_report.html          # HTML report
├── experiment_results.csv          # CSV data
├── accuracy_comparison.png         # Visualizations
├── training_time_comparison.png
└── performance_by_target.png
```

## 💾 Checkpoint Management

The framework organizes checkpoints by model type for better management:

### Checkpoint Structure
```
checkpoints/
├── EEGGenderCNN/          # All gender classification model checkpoints
│   ├── experiment1_best.pth
│   ├── experiment1_epoch_10.pth
│   └── experiment2_best.pth
├── EEGAgeCNN/             # All age classification model checkpoints
│   ├── experiment1_best.pth
│   └── experiment2_best.pth
└── ...
```

### Loading Checkpoints
```python
# Load a specific model checkpoint
model_name = "EEGGenderCNN"
experiment_name = "gender_cnn_baseline"
checkpoint_path = f"checkpoints/{model_name}/{experiment_name}_best.pth"

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Checkpoint Contents
Each checkpoint contains:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `scheduler_state_dict`: Learning rate scheduler state
- `epoch`: Training epoch
- `best_val_loss`: Best validation loss
- `config`: Training configuration
- `model_info`: Model architecture information

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

## 📊 Comprehensive Experiment Results

When running `--mode comprehensive`, the framework executes all 12 experiment types and provides:

### **Aggregated Results**
- **Cross-validation experiments**: Results averaged across 5 folds
- **Cross-task experiments**: Performance on different task types
- **Comprehensive comparison**: All experiment types in one report

### **Result Files**
- `experiment_summary.json`: Complete results summary
- `model_comparison.json`: Side-by-side model comparison
- Individual experiment folders with detailed metrics
- Training progress plots and confusion matrices

### **Performance Insights**
- **Baseline Performance**: Standard train/val/test split results
- **Robustness**: Cross-validation stability analysis
- **Generalization**: Cross-task performance evaluation
- **Comprehensive**: Combined cross-task and cross-validation analysis

## 📄 License

This project follows the same license as the parent EEG classification project.

---

**Note**: This framework is designed to be extensible and maintainable. The modular architecture allows for easy addition of new models, experiments, and evaluation metrics while maintaining the DRY principle and systematic approach to machine learning experiments. The comprehensive experiment framework provides robust evaluation across multiple data loading strategies and experimental conditions.
