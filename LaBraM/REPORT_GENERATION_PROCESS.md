# LaBraM Report Generation Process

## Overview

LaBraM generates comprehensive reports from JSON-formatted log files using the `report_generator.py` script. The process extracts metrics, aggregates results, and creates multiple report formats (text, HTML, CSV, visualizations).

## Log File Format

### Structure
- **Format**: JSON Lines (one JSON object per line)
- **Location**: `outputs/{experiment_name}/log.txt`
- **Content**: One JSON object per epoch containing training and evaluation metrics

### Example Log Entry
```json
{
  "train_lr": 0.00014994946942900462,
  "train_min_lr": 5.543940617280463e-07,
  "train_loss": 0.9689723954658316,
  "train_class_acc": 0.5000049321338383,
  "train_loss_scale": 853.3333333333334,
  "train_weight_decay": 0.05000000000000037,
  "train_grad_norm": Infinity,
  "val_accuracy": 0.4966634672517026,
  "val_balanced_accuracy": 0.4841386241198486,
  "val_f1_weighted": 0.4964280950466308,
  "val_loss": 0.9630864994567737,
  "test_accuracy": 0.5309557109557109,
  "test_balanced_accuracy": 0.526822638698165,
  "test_f1_weighted": 0.5327838008093472,
  "test_loss": 0.9024075855101857,
  "epoch": 1,
  "n_parameters": 5825339
}
```

## Report Generation Process

### 1. Collection Phase (`_collect_all_results()`)

**Location**: `outputs/` directory

**Process**:
1. Scans for experiment directories matching pattern: `*_{segment_length}` (e.g., `age_baseline_1s`, `gender_baseline_2s`, `age_baseline_4s`)
2. For each experiment directory:
   - Checks if it's a **Cross-Validation (CV) experiment** (has `fold_*` subdirectories)
   - If CV: Aggregates results from all folds
   - If regular: Parses single `log.txt` file

**CV Detection**:
- Looks for subdirectories matching `fold_*` pattern
- Each fold has its own `log.txt` file

### 2. Log File Parsing (`_parse_log_file()`)

**Input**: Path to `log.txt` file

**Process**:
1. **Read all lines** from log file
2. **Parse JSON lines**: Each line is a JSON object (one per epoch)
3. **Find best epoch**: Epoch with highest `val_accuracy`
4. **Extract metrics**:
   - **Best epoch metrics**: Used for validation/test reporting
   - **Final epoch metrics**: Used for training metrics
   - **Task-type metrics**: Active/passive task breakdown (if available)

**Output Structure**:
```python
{
    'experiment_name': str,
    'target_type': str,  # 'age', 'gender', 'combined', 'multi_output'
    'experiment_type': str,  # 'baseline', 'cv_stratified', etc.
    'segment_length': str,  # '1s', '2s', '4s'
    'best_epoch': int,
    'n_parameters': int,
    'train_metrics': {
        'final_loss': float,
        'final_acc': float,
        'final_lr': float,
    },
    'val_metrics': {
        'loss': float,
        'accuracy': float,
        'balanced_accuracy': float,
        'pr_auc': float,
        'roc_auc': float,
    },
    'test_metrics': {
        'loss': float,
        'accuracy': float,
        'balanced_accuracy': float,
        'pr_auc': float,
        'roc_auc': float,
    },
    'task_type_metrics': {  # Optional
        'active': {...},
        'passive': {...}
    },
    'all_epochs': List[Dict],  # All epoch metrics
    'success': bool
}
```

### 3. CV Aggregation (`_aggregate_cv_results()`)

**For Cross-Validation Experiments**:

**Process**:
1. Parse log file from each fold (`fold_0/log.txt`, `fold_1/log.txt`, etc.)
2. Extract metrics from each fold
3. **Average metrics** across folds:
   - Mean for all metrics
   - Standard deviation for accuracy metrics
4. Store individual fold results for detailed analysis

**Output Structure**:
```python
{
    # Same structure as regular experiment, plus:
    'n_folds': int,
    'is_cv': True,
    'val_metrics': {
        # ... plus:
        'accuracy_std': float,
        'balanced_accuracy_std': float,
    },
    'test_metrics': {
        # ... plus:
        'accuracy_std': float,
        'balanced_accuracy_std': float,
    },
    'fold_metrics': List[Dict]  # Individual fold results
}
```

### 4. Experiment Name Parsing (`_parse_experiment_name()`)

**Pattern Recognition**:
- Extracts segment length: `_1s`, `_2s`, or `_4s`
- Extracts target type: `age_`, `gender_`, `combined_`, `multi_output_`
- Extracts experiment type: Remaining part after target type

**Examples**:
- `age_baseline_1s` → target: `age`, type: `baseline`, segment: `1s`
- `gender_baseline_2s` → target: `gender`, type: `baseline`, segment: `2s`
- `age_baseline_4s` → target: `age`, type: `baseline`, segment: `4s`
- `gender_cv_gender_stratified_1s` → target: `gender`, type: `cv_stratified`, segment: `1s`

## Report Types Generated

### 1. Text Summary (`_generate_text_summary()`)

**Output**: `labram_experiment_summary_{segment_length}.txt`

**Content**:
- Overall statistics (total experiments, successful experiments)
- Grouped by target type (age, gender, combined, multi_output)
- For each experiment:
  - Experiment name and type
  - Best epoch
  - Validation and test metrics
  - CV statistics (if applicable)
  - Task-type-specific metrics (active/passive breakdown)
  - Dataset sample counts

**Example Output**:
```
AGE CLASSIFICATION:
----------------------------------------

Experiment: age_baseline_1s
Type: baseline
Segment Length: 1s
Best Epoch: 3
Validation Accuracy: 0.5476
Test Accuracy: 0.5553
Test Balanced Accuracy: 0.5462
```

### 2. HTML Report (`_generate_html_report()`)

**Output**: `labram_experiment_report_{segment_length}.html`

**Features**:
- Styled HTML with CSS
- Grouped by target type
- Visual metric cards
- Tables for detailed metrics
- Task-type-specific metrics displayed prominently
- Dataset sample counts

**Structure**:
- Header with summary statistics
- Sections for each target type
- Individual experiment cards with metrics
- Visualizations embedded (if generated)

### 3. CSV Summary (`_generate_csv_summary()`)

**Output**: `labram_experiment_results_{segment_length}.csv`

**Columns**:
- Experiment name
- Target type
- Experiment type
- Segment length
- Best epoch
- Model parameters
- Validation metrics (accuracy, balanced accuracy, loss)
- Test metrics (accuracy, balanced accuracy, loss, ROC-AUC, PR-AUC)
- CV statistics (if applicable)
- Task-type metrics (if available)

**Use Case**: Easy import into spreadsheet software or further analysis

### 4. Visualizations (`_generate_visualizations()`)

**Outputs**:
- `accuracy_comparison_{segment_length}.png`: Bar chart comparing accuracies
- `metrics_comparison_{segment_length}.png`: Multi-metric comparison
- `performance_by_target_{segment_length}.png`: Performance grouped by target type

**Libraries Used**:
- `matplotlib` for plotting
- `seaborn` for styling
- `pandas` for data manipulation

## Key Features

### Best Epoch Selection
- **Criterion**: Highest `val_accuracy`
- **Usage**: Validation and test metrics from best epoch are reported
- **Rationale**: Prevents overfitting by selecting model with best validation performance

### Cross-Validation Support
- Automatically detects CV experiments
- Aggregates metrics across folds
- Reports mean ± standard deviation
- Stores individual fold results for detailed analysis

### Task-Type Metrics
- Extracts active/passive task breakdown from `task_type_metrics` field in log
- Reports accuracy and F1-score for each task type (active/passive)
- Shows sample counts (n=) for each task type in reports
- Available when task type evaluation is performed during training (every 5 epochs and final epoch)
- Includes dataset sample counts for train/val/test splits by task type

### Error Handling
- Gracefully handles missing log files
- Skips invalid JSON lines
- Reports parsing errors without crashing
- Marks failed experiments in reports

## Usage

### Command Line
```bash
# Generate reports for specific segment length
python report_generator.py 1s
python report_generator.py 2s
python report_generator.py 4s

# Generate reports for all segment lengths (1s, 2s, 4s)
python report_generator.py
```

### Programmatic Usage
```python
from report_generator import LaBraMReportGenerator

# Generate reports for specific segment length
generator = LaBraMReportGenerator(segment_length='1s', outputs_dir='outputs')
generator.generate_all_reports()

# Generate reports for 2s segments
generator_2s = LaBraMReportGenerator(segment_length='2s', outputs_dir='outputs')
generator_2s.generate_all_reports()

# Generate reports for 4s segments
generator_4s = LaBraMReportGenerator(segment_length='4s', outputs_dir='outputs')
generator_4s.generate_all_reports()
```

## File Structure

```
LaBraM/
├── outputs/
│   ├── age_baseline_1s/
│   │   └── log.txt          # JSON lines, one per epoch
│   ├── gender_baseline_2s/
│   │   └── log.txt
│   ├── age_baseline_4s/
│   │   └── log.txt
│   ├── gender_cv_gender_stratified_1s/
│   │   ├── fold_0/
│   │   │   └── log.txt      # CV fold log
│   │   ├── fold_1/
│   │   │   └── log.txt
│   │   └── ...
│   └── ...
├── results/
│   ├── reports_1s/
│   │   ├── labram_experiment_summary_1s.txt
│   │   ├── labram_experiment_report_1s.html
│   │   ├── labram_experiment_results_1s.csv
│   │   ├── accuracy_comparison_1s.png
│   │   ├── metrics_comparison_1s.png
│   │   └── performance_by_target_1s.png
│   ├── reports_2s/
│   │   └── ... (same structure)
│   └── reports_4s/
│       └── ... (same structure)
└── report_generator.py
```

## Differences from CNN Report Generator

### LaBraM Report Generator:
- **Input**: JSON Lines format (`log.txt`)
- **Best Epoch**: Selected by highest `val_accuracy`
- **CV Support**: Built-in CV aggregation
- **Task-Type Metrics**: Extracts active/passive breakdown
- **Metrics**: Includes ROC-AUC, PR-AUC, balanced accuracy

### CNN Report Generator:
- **Input**: Structured JSON files (`experiment_results.json`)
- **Best Epoch**: Selected during training (checkpoint-based)
- **CV Support**: Not implemented
- **Task-Type Metrics**: Evaluated separately during evaluation
- **Metrics**: Focuses on accuracy, F1-score, confusion matrices

## Summary

The LaBraM report generation process:
1. **Scans** experiment directories in `outputs/` matching segment length pattern (`*_1s`, `*_2s`, `*_4s`)
2. **Parses** JSON Lines log files (one epoch per line)
3. **Selects** best epoch based on validation accuracy
4. **Aggregates** CV results if applicable
5. **Generates** multiple report formats (text, HTML, CSV, visualizations)
6. **Groups** results by target type and experiment type
7. **Extracts** task-type-specific metrics when available (from `task_type_metrics` field)
8. **Displays** dataset sample counts by task type (train/val/test splits)

This provides a comprehensive view of all LaBraM experiments with easy-to-read summaries and detailed metrics for analysis. Reports are generated separately for each segment length (1s, 2s, 4s) to allow comparison across different temporal resolutions.
