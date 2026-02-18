"""
Report Generator for LaBraM EEG Classification Experiments
Creates comprehensive HTML and text reports from LaBraM experiment results.
"""

import os
import json
import glob
from typing import Dict, Any, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class LaBraMReportGenerator:
    """
    Generates comprehensive reports from LaBraM experiment results.
    Creates both HTML and text reports with visualizations.
    """
    
    def __init__(self, segment_length: str = '1s', outputs_dir: str = 'outputs', output_dir: str = None):
        """
        Initialize report generator.
        
        Args:
            segment_length: Segment length ('1s', '2s', or '4s') to generate reports for
            outputs_dir: Directory containing LaBraM experiment outputs
            output_dir: Directory to save generated reports (defaults to reports_{segment_length})
        """
        self.segment_length = segment_length
        self.outputs_dir = outputs_dir
        self.output_dir = output_dir or f'reports_{segment_length}'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_all_reports(self):
        """Generate all types of reports."""
        print("Generating comprehensive LaBraM reports...")
        
        # Collect all experiment results
        experiment_results = self._collect_all_results()
        
        # Generate reports
        self._generate_text_summary(experiment_results)
        self._generate_html_report(experiment_results)
        self._generate_visualizations(experiment_results)
        self._generate_csv_summary(experiment_results)
        
        print(f"Reports generated in: {self.output_dir}")
    
    def _collect_all_results(self) -> List[Dict[str, Any]]:
        """Collect results from all experiments for the specified segment length."""
        results = []
        
        # Find all experiment directories for the specified segment length
        experiment_dirs = glob.glob(os.path.join(self.outputs_dir, f'*_{self.segment_length}'))
        
        for exp_dir in experiment_dirs:
            experiment_name = os.path.basename(exp_dir)
            log_file = os.path.join(exp_dir, 'log.txt')
            
            # Check if this is a CV experiment (has fold subdirectories)
            fold_dirs = glob.glob(os.path.join(exp_dir, 'fold_*'))
            
            if fold_dirs:
                # CV experiment - aggregate results from all folds
                result = self._aggregate_cv_results(exp_dir, experiment_name, fold_dirs)
                if result:
                    results.append(result)
            elif os.path.exists(log_file):
                # Regular experiment - single log file
                result = self._parse_log_file(log_file, experiment_name)
                if result:
                    results.append(result)
        
        return results
    
    def _parse_log_file(self, log_file: str, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Parse LaBraM log file and extract metrics."""
        try:
            # Read all lines
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                return None
            
            # Parse JSON lines
            metrics_list = []
            for line in lines:
                try:
                    metric = json.loads(line.strip())
                    metrics_list.append(metric)
                except json.JSONDecodeError:
                    continue
            
            if not metrics_list:
                return None
            
            # Get best epoch (highest validation accuracy)
            best_epoch_idx = max(range(len(metrics_list)), 
                               key=lambda i: metrics_list[i].get('val_accuracy', 0))
            best_metrics = metrics_list[best_epoch_idx]
            
            # Parse experiment name
            target_type, experiment_type = self._parse_experiment_name(experiment_name)
            
            # Extract task_type_metrics from the last epoch (final evaluation)
            task_type_metrics = None
            if metrics_list:
                last_metrics = metrics_list[-1]
                if 'task_type_metrics' in last_metrics:
                    task_type_metrics = last_metrics['task_type_metrics']
            
            # Extract metrics
            result = {
                'experiment_name': experiment_name,
                'target_type': target_type,
                'experiment_type': experiment_type,
                'segment_length': getattr(self, 'segment_length', 'unknown'),
                'best_epoch': best_metrics.get('epoch', 0),
                'n_parameters': best_metrics.get('n_parameters', 0),
                'train_metrics': {
                    'final_loss': metrics_list[-1].get('train_loss', 0),
                    'final_acc': metrics_list[-1].get('train_class_acc', 0),
                    'final_lr': metrics_list[-1].get('train_lr', 0),
                },
                'val_metrics': {
                    'loss': best_metrics.get('val_loss', 0),
                    'accuracy': best_metrics.get('val_accuracy', 0),
                    'balanced_accuracy': best_metrics.get('val_balanced_accuracy', 0),
                    'pr_auc': best_metrics.get('val_pr_auc', 0),
                    'roc_auc': best_metrics.get('val_roc_auc', 0),
                },
                'test_metrics': {
                    'loss': best_metrics.get('test_loss', 0),
                    'accuracy': best_metrics.get('test_accuracy', 0),
                    'balanced_accuracy': best_metrics.get('test_balanced_accuracy', 0),
                    'pr_auc': best_metrics.get('test_pr_auc', 0),
                    'roc_auc': best_metrics.get('test_roc_auc', 0),
                },
                'task_type_metrics': task_type_metrics,  # Active/passive task metrics
                'test_segment_metrics': best_metrics.get('test_segment_metrics'),  # Segment-level (per-window) when --aggregate_by_participant
                'test_participant_mean_prob_metrics': best_metrics.get('test_participant_mean_prob_metrics'),
                'test_participant_metrics': best_metrics.get('test_participant_metrics'),  # Participant-level majority vote
                'all_epochs': metrics_list,
                'success': True
            }
            
            return result
            
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")
            return None
    
    def _aggregate_cv_results(self, exp_dir: str, experiment_name: str, fold_dirs: List[str]) -> Optional[Dict[str, Any]]:
        """Aggregate results from all CV folds."""
        try:
            fold_metrics = []
            
            for fold_dir in sorted(fold_dirs):
                log_file = os.path.join(fold_dir, 'log.txt')
                if os.path.exists(log_file):
                    fold_result = self._parse_log_file(log_file, experiment_name)
                    if fold_result:
                        fold_metrics.append(fold_result)
            
            if not fold_metrics:
                return None
            
            # Average metrics across folds
            avg_metrics = {
                'experiment_name': experiment_name,
                'target_type': fold_metrics[0]['target_type'],
                'experiment_type': fold_metrics[0]['experiment_type'],
                'segment_length': fold_metrics[0]['segment_length'],
                'best_epoch': np.mean([m['best_epoch'] for m in fold_metrics]),
                'n_parameters': fold_metrics[0]['n_parameters'],
                'n_folds': len(fold_metrics),
                'train_metrics': {
                    'final_loss': np.mean([m['train_metrics']['final_loss'] for m in fold_metrics]),
                    'final_acc': np.mean([m['train_metrics']['final_acc'] for m in fold_metrics]),
                },
                'val_metrics': {
                    'loss': np.mean([m['val_metrics']['loss'] for m in fold_metrics]),
                    'accuracy': np.mean([m['val_metrics']['accuracy'] for m in fold_metrics]),
                    'balanced_accuracy': np.mean([m['val_metrics']['balanced_accuracy'] for m in fold_metrics]),
                    'pr_auc': np.mean([m['val_metrics']['pr_auc'] for m in fold_metrics]),
                    'roc_auc': np.mean([m['val_metrics']['roc_auc'] for m in fold_metrics]),
                    'accuracy_std': np.std([m['val_metrics']['accuracy'] for m in fold_metrics]),
                    'balanced_accuracy_std': np.std([m['val_metrics']['balanced_accuracy'] for m in fold_metrics]),
                },
                'test_metrics': {
                    'loss': np.mean([m['test_metrics']['loss'] for m in fold_metrics]),
                    'accuracy': np.mean([m['test_metrics']['accuracy'] for m in fold_metrics]),
                    'balanced_accuracy': np.mean([m['test_metrics']['balanced_accuracy'] for m in fold_metrics]),
                    'pr_auc': np.mean([m['test_metrics']['pr_auc'] for m in fold_metrics]),
                    'roc_auc': np.mean([m['test_metrics']['roc_auc'] for m in fold_metrics]),
                    'accuracy_std': np.std([m['test_metrics']['accuracy'] for m in fold_metrics]),
                    'balanced_accuracy_std': np.std([m['test_metrics']['balanced_accuracy'] for m in fold_metrics]),
                },
                'success': True,
                'is_cv': True,
                'fold_metrics': fold_metrics  # Store individual fold results
            }
            
            return avg_metrics
            
        except Exception as e:
            print(f"Error aggregating CV results for {experiment_name}: {e}")
            return None
    
    def _parse_experiment_name(self, experiment_name: str) -> tuple:
        """Parse experiment name to extract target type and experiment type."""
        # Extract segment length (1s, 2s, or 4s)
        if '_1s' in experiment_name:
            segment_length = '1s'
            base_name = experiment_name.replace('_1s', '')
        elif '_2s' in experiment_name:
            segment_length = '2s'
            base_name = experiment_name.replace('_2s', '')
        elif '_4s' in experiment_name:
            segment_length = '4s'
            base_name = experiment_name.replace('_4s', '')
        else:
            # Don't overwrite self.segment_length if we can't parse it from the name
            # Use the existing self.segment_length set in __init__
            segment_length = getattr(self, 'segment_length', 'unknown')
            base_name = experiment_name
        
        # Store segment length for later use (only if we successfully parsed it)
        if segment_length != 'unknown':
            self.segment_length = segment_length
        
        # Determine target type
        if base_name.startswith('age_'):
            target_type = 'age'
            experiment_type = base_name.replace('age_', '')
        elif base_name.startswith('gender_'):
            target_type = 'gender'
            experiment_type = base_name.replace('gender_', '')
        elif base_name.startswith('combined_'):
            target_type = 'combined'
            experiment_type = base_name.replace('combined_', '')
        elif base_name.startswith('multi_output_'):
            target_type = 'multi_output'
            experiment_type = base_name.replace('multi_output_', '')
        else:
            target_type = 'unknown'
            experiment_type = base_name
        
        return target_type, experiment_type
    
    def _generate_text_summary(self, experiment_results: List[Dict[str, Any]]):
        """Generate text summary report."""
        report_file = os.path.join(self.output_dir, f'labram_experiment_summary_{self.segment_length}.txt')
        
        with open(report_file, 'w') as f:
            f.write(f"LaBraM EEG CLASSIFICATION EXPERIMENT SUMMARY ({self.segment_length} segments)\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Segment Length: {self.segment_length}\n\n")
            
            # Overall statistics
            successful = [r for r in experiment_results if r.get('success', False)]
            
            f.write(f"Total Experiments: {len(experiment_results)}\n")
            f.write(f"Successful: {len(successful)}\n\n")
            
            # Group by target type
            by_target = {}
            for result in successful:
                target = result['target_type']
                if target not in by_target:
                    by_target[target] = []
                by_target[target].append(result)
            
            # Summary by target type
            for target_type, results in by_target.items():
                f.write(f"\n{target_type.upper()} CLASSIFICATION:\n")
                f.write("-" * 40 + "\n")
                
                for result in results:
                    f.write(f"\nExperiment: {result['experiment_name']}\n")
                    f.write(f"Type: {result['experiment_type']}\n")
                    f.write(f"Segment Length: {result.get('segment_length', 'unknown')}\n")
                    
                    # Check if this is a CV experiment
                    if result.get('is_cv', False):
                        f.write(f"Cross-Validation: {result['n_folds']} folds\n")
                        f.write(f"Validation Accuracy: {result['val_metrics']['accuracy']:.4f} ± {result['val_metrics'].get('accuracy_std', 0):.4f}\n")
                        f.write(f"Test Accuracy: {result['test_metrics']['accuracy']:.4f} ± {result['test_metrics'].get('accuracy_std', 0):.4f}\n")
                        f.write(f"Test Balanced Accuracy: {result['test_metrics']['balanced_accuracy']:.4f} ± {result['test_metrics'].get('balanced_accuracy_std', 0):.4f}\n")
                    else:
                        f.write(f"Best Epoch: {result['best_epoch']:.0f}\n")
                        f.write(f"Validation Accuracy: {result['val_metrics']['accuracy']:.4f}\n")
                        f.write(f"Test Accuracy: {result['test_metrics']['accuracy']:.4f}\n")
                        f.write(f"Test Balanced Accuracy: {result['test_metrics']['balanced_accuracy']:.4f}\n")
                    
                    f.write(f"Test ROC-AUC: {result['test_metrics']['roc_auc']:.4f}\n")
                    f.write(f"Test PR-AUC: {result['test_metrics']['pr_auc']:.4f}\n")
                    if result.get('test_segment_metrics') and result.get('test_participant_metrics'):
                        seg = result['test_segment_metrics']
                        part_mp = result.get('test_participant_mean_prob_metrics') or {}
                        part_mv = result['test_participant_metrics']
                        seg_acc = seg.get('accuracy')
                        if seg_acc is not None:
                            f.write(f"\nSegment vs participant-level (with --aggregate_by_participant):\n")
                            f.write(f"  Segment-level (per-window) Accuracy: {seg_acc:.4f}\n")
                            if part_mp.get('accuracy') is not None:
                                f.write(f"  Participant (mean prob) Accuracy: {part_mp['accuracy']:.4f}\n")
                            if part_mv.get('accuracy') is not None:
                                f.write(f"  Participant (majority vote) Accuracy: {part_mv['accuracy']:.4f}\n")
                    # Add dataset sample counts if available
                    if result.get('train_task_type_counts'):
                        counts = result['train_task_type_counts']
                        f.write(f"\nTraining Dataset Sample Counts:\n")
                        f.write(f"  Active tasks: {counts.get('active', 0):,} samples\n")
                        f.write(f"  Passive tasks: {counts.get('passive', 0):,} samples\n")
                    if result.get('val_task_type_counts'):
                        counts = result['val_task_type_counts']
                        f.write(f"\nValidation Dataset Sample Counts:\n")
                        f.write(f"  Active tasks: {counts.get('active', 0):,} samples\n")
                        f.write(f"  Passive tasks: {counts.get('passive', 0):,} samples\n")
                    if result.get('test_task_type_counts'):
                        counts = result['test_task_type_counts']
                        f.write(f"\nTest Dataset Sample Counts:\n")
                        f.write(f"  Active tasks: {counts.get('active', 0):,} samples\n")
                        f.write(f"  Passive tasks: {counts.get('passive', 0):,} samples\n")
                    
                    # Add task-type-specific metrics if available
                    if result.get('task_type_metrics'):
                        f.write(f"\nTask-Type-Specific Metrics (Test Set):\n")
                        task_metrics = result['task_type_metrics']
                        if 'active' in task_metrics:
                            active_acc = task_metrics['active'].get('accuracy', 0.0)
                            active_f1 = task_metrics['active'].get('f1_weighted', 0.0)
                            active_n = task_metrics['active'].get('num_samples', 0)
                            f.write(f"  Active Task Accuracy: {active_acc:.4f} (n={active_n:,})\n")
                            f.write(f"  Active Task F1-Score: {active_f1:.4f}\n")
                        if 'passive' in task_metrics:
                            passive_acc = task_metrics['passive'].get('accuracy', 0.0)
                            passive_f1 = task_metrics['passive'].get('f1_weighted', 0.0)
                            passive_n = task_metrics['passive'].get('num_samples', 0)
                            f.write(f"  Passive Task Accuracy: {passive_acc:.4f} (n={passive_n:,})\n")
                            f.write(f"  Passive Task F1-Score: {passive_f1:.4f}\n")
                        f.write(f"  Note: Combined accuracy is weighted by sample counts, not (Active + Passive)/2\n")
    
    def _generate_html_report(self, experiment_results: List[Dict[str, Any]]):
        """Generate HTML report."""
        html_file = os.path.join(self.output_dir, f'labram_experiment_report_{self.segment_length}.html')
        
        # Group by target type
        by_target = {}
        for result in experiment_results:
            if result.get('success', False):
                target = result['target_type']
                if target not in by_target:
                    by_target[target] = []
                by_target[target].append(result)
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LaBraM EEG Classification Experiment Report ({self.segment_length})</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .header {{ background-color: #4CAF50; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 30px 0; background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .experiment {{ margin: 15px 0; padding: 15px; border-left: 4px solid #4CAF50; background-color: #f9f9f9; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 10px 0; }}
        .metric {{ background-color: white; padding: 10px; border-radius: 5px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        .metric-label {{ font-size: 12px; color: #666; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        h2 {{ color: #333; }}
        h3 {{ color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LaBraM EEG Classification Experiment Report ({self.segment_length} segments)</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Segment Length: {self.segment_length}</p>
        <p>Total Experiments: {len(experiment_results)} | Successful: {len([r for r in experiment_results if r.get('success', False)])}</p>
    </div>
"""
        
        # Generate sections for each target type
        for target_type in ['gender', 'age', 'combined', 'multi_output']:
            if target_type in by_target:
                results = by_target[target_type]
                
                html_content += f"""
    <div class="section">
        <h2>{target_type.upper()} Classification Experiments</h2>
"""
                
                for result in results:
                    # Check if this is a CV experiment
                    is_cv = result.get('is_cv', False)
                    cv_info = f" ({result['n_folds']} folds)" if is_cv else ""
                    
                    html_content += f"""
        <div class="experiment">
            <h3>{result['experiment_name']}{cv_info}</h3>
            <p><strong>Type:</strong> {result['experiment_type']} | <strong>Best Epoch:</strong> {result['best_epoch']:.0f}</p>
            <p><strong>Model Parameters:</strong> {result['n_parameters']:,}</p>
            
            <div class="metrics">
"""
                    # Display metrics with std dev if CV
                    if is_cv and 'accuracy_std' in result['test_metrics']:
                        html_content += f"""
                <div class="metric">
                    <div class="metric-value">{result['val_metrics']['accuracy']:.1%}</div>
                    <div class="metric-label">Val Accuracy ±{result['val_metrics'].get('accuracy_std', 0):.2%}</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{result['test_metrics']['accuracy']:.1%}</div>
                    <div class="metric-label">Test Accuracy ±{result['test_metrics'].get('accuracy_std', 0):.2%}</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{result['test_metrics']['balanced_accuracy']:.1%}</div>
                    <div class="metric-label">Test Balanced Acc ±{result['test_metrics'].get('balanced_accuracy_std', 0):.2%}</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{result['test_metrics']['roc_auc']:.1%}</div>
                    <div class="metric-label">Test ROC-AUC</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{result['test_metrics']['pr_auc']:.1%}</div>
                    <div class="metric-label">Test PR-AUC</div>
                </div>
"""
                    else:
                        html_content += f"""
                <div class="metric">
                    <div class="metric-value">{result['val_metrics']['accuracy']:.1%}</div>
                    <div class="metric-label">Validation Accuracy</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{result['test_metrics']['accuracy']:.1%}</div>
                    <div class="metric-label">Test Accuracy</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{result['test_metrics']['balanced_accuracy']:.1%}</div>
                    <div class="metric-label">Test Balanced Accuracy</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{result['test_metrics']['roc_auc']:.1%}</div>
                    <div class="metric-label">Test ROC-AUC</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{result['test_metrics']['pr_auc']:.1%}</div>
                    <div class="metric-label">Test PR-AUC</div>
                </div>
"""
                    
                    # Add dataset sample counts if available
                    if result.get('train_task_type_counts') or result.get('val_task_type_counts') or result.get('test_task_type_counts'):
                        html_content += """
            <div class="metrics" style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
                <h4 style="margin: 0 0 10px 0; color: #666;">Dataset Sample Counts:</h4>
"""
                        if result.get('train_task_type_counts'):
                            counts = result['train_task_type_counts']
                            html_content += f"""
                <div style="grid-column: 1 / -1; margin-bottom: 5px;">
                    <strong>Training:</strong> {counts.get('active', 0):,} active, {counts.get('passive', 0):,} passive
                </div>
"""
                        if result.get('val_task_type_counts'):
                            counts = result['val_task_type_counts']
                            html_content += f"""
                <div style="grid-column: 1 / -1; margin-bottom: 5px;">
                    <strong>Validation:</strong> {counts.get('active', 0):,} active, {counts.get('passive', 0):,} passive
                </div>
"""
                        if result.get('test_task_type_counts'):
                            counts = result['test_task_type_counts']
                            html_content += f"""
                <div style="grid-column: 1 / -1; margin-bottom: 5px;">
                    <strong>Test:</strong> {counts.get('active', 0):,} active, {counts.get('passive', 0):,} passive
                </div>
"""
                        html_content += """
            </div>
"""
                    
                    # Add task-type-specific metrics if available
                    if result.get('task_type_metrics'):
                        task_metrics = result['task_type_metrics']
                        html_content += """
            <div class="metrics" style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
                <h4 style="margin: 0 0 10px 0; color: #666;">Task-Type-Specific Metrics (Test Set):</h4>
"""
                        if 'active' in task_metrics:
                            active_acc = task_metrics['active'].get('accuracy', 0.0)
                            active_f1 = task_metrics['active'].get('f1_weighted', 0.0)
                            active_n = task_metrics['active'].get('num_samples', 0)
                            html_content += f"""
                <div class="metric">
                    <div class="metric-value">{active_acc:.1%}</div>
                    <div class="metric-label">Active Task Accuracy (n={active_n:,})</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{active_f1:.3f}</div>
                    <div class="metric-label">Active Task F1-Score</div>
                </div>
"""
                        if 'passive' in task_metrics:
                            passive_acc = task_metrics['passive'].get('accuracy', 0.0)
                            passive_f1 = task_metrics['passive'].get('f1_weighted', 0.0)
                            passive_n = task_metrics['passive'].get('num_samples', 0)
                            html_content += f"""
                <div class="metric">
                    <div class="metric-value">{passive_acc:.1%}</div>
                    <div class="metric-label">Passive Task Accuracy (n={passive_n:,})</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{passive_f1:.3f}</div>
                    <div class="metric-label">Passive Task F1-Score</div>
                </div>
"""
                        html_content += """
                <div style="grid-column: 1 / -1; font-size: 0.9em; color: #666; font-style: italic; margin-top: 10px;">
                    Note: Combined accuracy is weighted by sample counts, not (Active + Passive)/2
                </div>
            </div>
"""
                    
                    html_content += """
            </div>
        </div>
"""
                
                html_content += "    </div>\n"
        
        # Add summary table
        html_content += """
    <div class="section">
        <h2>Summary Table</h2>
        <table>
            <tr>
                <th>Experiment</th>
                <th>Target</th>
                <th>Type</th>
                <th>Best Epoch</th>
                <th>Val Accuracy</th>
                <th>Test Accuracy</th>
                <th>Test Balanced Acc</th>
                <th>Test ROC-AUC</th>
                <th>Test PR-AUC</th>
            </tr>
"""
        
        for result in experiment_results:
            if result.get('success', False):
                html_content += f"""
            <tr>
                <td>{result['experiment_name']}</td>
                <td>{result['target_type']}</td>
                <td>{result['experiment_type']}</td>
                <td>{result['best_epoch']}</td>
                <td>{result['val_metrics']['accuracy']:.4f}</td>
                <td>{result['test_metrics']['accuracy']:.4f}</td>
                <td>{result['test_metrics']['balanced_accuracy']:.4f}</td>
                <td>{result['test_metrics']['roc_auc']:.4f}</td>
                <td>{result['test_metrics']['pr_auc']:.4f}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
</body>
</html>
"""
        
        with open(html_file, 'w') as f:
            f.write(html_content)
    
    def _generate_visualizations(self, experiment_results: List[Dict[str, Any]]):
        """Generate visualization plots."""
        successful = [r for r in experiment_results if r.get('success', False)]
        
        if not successful:
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Accuracy comparison by experiment
        self._plot_accuracy_comparison(successful)
        
        # 2. Performance by target type
        self._plot_performance_by_target(successful)
        
        # 3. Metrics comparison
        self._plot_metrics_comparison(successful)
    
    def _plot_accuracy_comparison(self, successful_results: List[Dict[str, Any]]):
        """Plot accuracy comparison across experiments."""
        experiments = [r['experiment_name'] for r in successful_results]
        test_accuracies = [r['test_metrics']['accuracy'] for r in successful_results]
        
        plt.figure(figsize=(14, 8))
        bars = plt.barh(range(len(experiments)), test_accuracies, color='skyblue', alpha=0.7)
        plt.yticks(range(len(experiments)), experiments)
        plt.xlabel('Test Accuracy', fontsize=12, fontweight='bold')
        plt.title(f'LaBraM Test Accuracy by Experiment ({self.segment_length} segments)', fontsize=16, fontweight='bold')
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, test_accuracies)):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{acc:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'accuracy_comparison_{self.segment_length}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_by_target(self, successful_results: List[Dict[str, Any]]):
        """Plot performance metrics by target type."""
        # Group by target type
        target_groups = {}
        for result in successful_results:
            target = result['target_type']
            if target not in target_groups:
                target_groups[target] = []
            target_groups[target].append(result)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        targets = list(target_groups.keys())
        
        # Plot 1: Test Accuracy by target type
        accuracies = [max([r['test_metrics']['accuracy'] 
                          for r in target_groups[target]]) for target in targets]
        
        axes[0, 0].bar(targets, accuracies, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        axes[0, 0].set_title('Best Test Accuracy by Target Type', fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        # Plot 2: Test Balanced Accuracy
        balanced_accs = [max([r['test_metrics']['balanced_accuracy'] 
                             for r in target_groups[target]]) for target in targets]
        
        axes[0, 1].bar(targets, balanced_accs, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        axes[0, 1].set_title('Best Test Balanced Accuracy', fontweight='bold')
        axes[0, 1].set_ylabel('Balanced Accuracy')
        axes[0, 1].set_ylim(0, 1)
        
        # Plot 3: ROC-AUC by target type
        roc_aucs = [max([r['test_metrics']['roc_auc'] 
                        for r in target_groups[target]]) for target in targets]
        
        axes[1, 0].bar(targets, roc_aucs, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        axes[1, 0].set_title('Best Test ROC-AUC', fontweight='bold')
        axes[1, 0].set_ylabel('ROC-AUC')
        axes[1, 0].set_ylim(0, 1)
        
        # Plot 4: PR-AUC by target type
        pr_aucs = [max([r['test_metrics']['pr_auc'] 
                       for r in target_groups[target]]) for target in targets]
        
        axes[1, 1].bar(targets, pr_aucs, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        axes[1, 1].set_title('Best Test PR-AUC', fontweight='bold')
        axes[1, 1].set_ylabel('PR-AUC')
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'performance_by_target_{self.segment_length}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_comparison(self, successful_results: List[Dict[str, Any]]):
        """Plot all metrics comparison."""
        experiments = [r['experiment_name'] for r in successful_results]
        
        accuracies = [r['test_metrics']['accuracy'] for r in successful_results]
        balanced_accs = [r['test_metrics']['balanced_accuracy'] for r in successful_results]
        roc_aucs = [r['test_metrics']['roc_auc'] for r in successful_results]
        pr_aucs = [r['test_metrics']['pr_auc'] for r in successful_results]
        
        x = np.arange(len(experiments))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        ax.bar(x - 1.5*width, accuracies, width, label='Test Accuracy', alpha=0.8)
        ax.bar(x - 0.5*width, balanced_accs, width, label='Test Balanced Acc', alpha=0.8)
        ax.bar(x + 0.5*width, roc_aucs, width, label='Test ROC-AUC', alpha=0.8)
        ax.bar(x + 1.5*width, pr_aucs, width, label='Test PR-AUC', alpha=0.8)
        
        ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'LaBraM Test Metrics Comparison ({self.segment_length} segments)', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'metrics_comparison_{self.segment_length}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_csv_summary(self, experiment_results: List[Dict[str, Any]]):
        """Generate CSV summary for easy analysis."""
        successful = [r for r in experiment_results if r.get('success', False)]
        
        if not successful:
            return
        
        # Prepare data for CSV
        data = []
        for result in successful:
            data.append({
                'experiment_name': result['experiment_name'],
                'target_type': result['target_type'],
                'experiment_type': result['experiment_type'],
                'segment_length': result.get('segment_length', 'unknown'),
                'best_epoch': result['best_epoch'],
                'n_parameters': result['n_parameters'],
                'train_loss': result['train_metrics']['final_loss'],
                'train_acc': result['train_metrics']['final_acc'],
                'val_loss': result['val_metrics']['loss'],
                'val_accuracy': result['val_metrics']['accuracy'],
                'val_balanced_accuracy': result['val_metrics']['balanced_accuracy'],
                'val_pr_auc': result['val_metrics']['pr_auc'],
                'val_roc_auc': result['val_metrics']['roc_auc'],
                'test_loss': result['test_metrics']['loss'],
                'test_accuracy': result['test_metrics']['accuracy'],
                'test_balanced_accuracy': result['test_metrics']['balanced_accuracy'],
                'test_pr_auc': result['test_metrics']['pr_auc'],
                'test_roc_auc': result['test_metrics']['roc_auc'],
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        csv_file = os.path.join(self.output_dir, f'labram_experiment_results_{self.segment_length}.csv')
        df.to_csv(csv_file, index=False)
        
        print(f"CSV summary saved to: {csv_file}")


def main():
    """Main function to generate reports for 1s, 2s, or 4s segment lengths."""
    import sys
    
    # Check if specific segment length is requested
    segment_length = sys.argv[1] if len(sys.argv) > 1 else None
    
    if segment_length and segment_length in ['1s', '2s', '4s']:
        # Generate report for specific segment length
        generator = LaBraMReportGenerator(segment_length=segment_length)
        generator.generate_all_reports()
    else:
        # Generate reports for all segment lengths
        print("Generating reports for all segment lengths (1s, 2s, 4s)...")
        
        # Generate 1s reports
        print("\nGenerating reports for 1s segment length...")
        generator_1s = LaBraMReportGenerator(segment_length='1s')
        generator_1s.generate_all_reports()
        
        # Generate 2s reports
        print("\nGenerating reports for 2s segment length...")
        generator_2s = LaBraMReportGenerator(segment_length='2s')
        generator_2s.generate_all_reports()
        
        # Generate 4s reports
        print("\nGenerating reports for 4s segment length...")
        generator_4s = LaBraMReportGenerator(segment_length='4s')
        generator_4s.generate_all_reports()
        
        print("\nAll reports generated successfully!")


if __name__ == '__main__':
    main()
