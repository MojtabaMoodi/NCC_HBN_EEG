"""
Report Generator for EEG Classification Experiments
Creates comprehensive HTML and text reports from experiment results.
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def _format_float_metric(value: Any) -> str:
    """Format a scalar metric for HTML/CSV; non-numeric values become 'N/A'."""
    return f"{value:.4f}" if isinstance(value, (int, float)) else "N/A"


class ReportGenerator:
    """
    Generates comprehensive reports from experiment results.
    Creates both HTML and text reports with visualizations.
    """
    
    def __init__(self, results_dir: str, output_dir: str = 'reports'):
        """
        Initialize report generator.
        
        Args:
            results_dir: Directory containing experiment results
            output_dir: Directory to save generated reports
        """
        self.results_dir = results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_all_reports(self, experiment_results: List[Dict[str, Any]]):
        """Generate all types of reports."""
        print("Generating comprehensive reports...")
        
        # Generate text summary
        self._generate_text_summary(experiment_results)
        
        # Generate HTML report
        self._generate_html_report(experiment_results)
        
        # Generate visualizations
        self._generate_visualizations(experiment_results)
        
        # Generate CSV summary
        self._generate_csv_summary(experiment_results)
        
        print(f"Reports generated in: {self.output_dir}")
    
    def _generate_text_summary(self, experiment_results: List[Dict[str, Any]]):
        """Generate text summary report."""
        report_file = os.path.join(self.output_dir, 'experiment_summary.txt')
        
        with open(report_file, 'w') as f:
            f.write("EEG CLASSIFICATION EXPERIMENT SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall statistics
            successful = [r for r in experiment_results if r.success]
            failed = [r for r in experiment_results if not r.success]
            
            f.write(f"Total Experiments: {len(experiment_results)}\n")
            f.write(f"Successful: {len(successful)}\n")
            f.write(f"Failed: {len(failed)}\n\n")
            
            # Successful experiments summary
            if successful:
                f.write("SUCCESSFUL EXPERIMENTS:\n")
                f.write("-" * 30 + "\n")
                
                for result in successful:
                    f.write(f"\nExperiment: {result.experiment_name}\n")
                    f.write(f"Model: {result.model_type}\n")
                    f.write(f"Target: {result.target_type}\n")
                    
                    # Check for multi-output metrics
                    if result.target_type == 'multi_output' and 'gender_head' in result.metrics and 'age_head' in result.metrics:
                        gender_metrics = result.metrics['gender_head']
                        age_metrics = result.metrics['age_head']
                        f.write(f"Gender Head - Accuracy: {gender_metrics['accuracy']:.4f}\n")
                        f.write(f"Gender Head - F1-Score: {gender_metrics['f1_weighted']:.4f}\n")
                        f.write(f"Age Head - Accuracy: {age_metrics['accuracy']:.4f}\n")
                        f.write(f"Age Head - F1-Score: {age_metrics['f1_weighted']:.4f}\n")
                    elif 'mae' in result.metrics:
                        # Regression task
                        f.write(f"MAE: {result.metrics['mae']:.4f} years\n")
                        f.write(f"RMSE: {result.metrics['rmse']:.4f} years\n")
                        f.write(f"R²: {result.metrics['r2']:.4f}\n")
                    else:
                        # Classification task
                        acc = result.metrics.get('accuracy')
                        f1 = result.metrics.get('f1_weighted')
                        acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else "N/A"
                        f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else "N/A"
                        f.write(f"Combined Accuracy: {acc_str}\n")
                        f.write(f"F1-Score: {f1_str}\n")
                        
                        # Add task-specific accuracies if available
                        if 'task_type_metrics' in result.metrics:
                            task_metrics = result.metrics['task_type_metrics']
                            if 'active' in task_metrics and 'passive' in task_metrics:
                                active_acc = task_metrics['active'].get('accuracy')
                                passive_acc = task_metrics['passive'].get('accuracy')
                                active_num_samples = task_metrics['active'].get('num_samples', 0)
                                passive_num_samples = task_metrics['passive'].get('num_samples', 0)
                                active_acc_str = f"{active_acc:.4f}" if isinstance(active_acc, (int, float)) else "N/A"
                                passive_acc_str = f"{passive_acc:.4f}" if isinstance(passive_acc, (int, float)) else "N/A"
                                f.write(f"Active Task Accuracy: {active_acc_str} (n={active_num_samples})\n")
                                f.write(f"Passive Task Accuracy: {passive_acc_str} (n={passive_num_samples})\n")
                                # Note: Combined accuracy is a weighted average based on sample counts, not a simple average
                                f.write(f"  (Note: Combined accuracy is weighted by sample counts, not (Active + Passive)/2)\n")
                            elif 'active' in task_metrics:
                                active_acc = task_metrics['active'].get('accuracy')
                                active_num_samples = task_metrics['active'].get('num_samples', 0)
                                active_acc_str = f"{active_acc:.4f}" if isinstance(active_acc, (int, float)) else "N/A"
                                f.write(f"Active Task Accuracy: {active_acc_str} (n={active_num_samples})\n")
                            elif 'passive' in task_metrics:
                                passive_acc = task_metrics['passive'].get('accuracy')
                                passive_num_samples = task_metrics['passive'].get('num_samples', 0)
                                passive_acc_str = f"{passive_acc:.4f}" if isinstance(passive_acc, (int, float)) else "N/A"
                                f.write(f"Passive Task Accuracy: {passive_acc_str} (n={passive_num_samples})\n")
                    
                    f.write(f"Training Time: {result.total_time:.2f}s\n")
            
            # Failed experiments
            if failed:
                f.write("\n\nFAILED EXPERIMENTS:\n")
                f.write("-" * 20 + "\n")
                
                for result in failed:
                    f.write(f"\nExperiment: {result.experiment_name}\n")
                    f.write(f"Error: {result.error}\n")
    
    def _generate_html_report(self, experiment_results: List[Dict[str, Any]]):
        """Generate HTML report."""
        html_file = os.path.join(self.output_dir, 'experiment_report.html')
        
        successful = [r for r in experiment_results if r.success]
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>EEG Classification Experiment Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .experiment {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .success {{ border-left: 5px solid #4CAF50; }}
        .failure {{ border-left: 5px solid #f44336; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }}
        .metric {{ background-color: #f9f9f9; padding: 10px; border-radius: 3px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>EEG Classification Experiment Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total Experiments: {len(experiment_results)} | Successful: {len(successful)} | Failed: {len([r for r in experiment_results if not r.success])}</p>
    </div>
"""
        
        # Add experiment details
        for result in experiment_results:
            status_class = "success" if result.success else "failure"
            status_text = "SUCCESS" if result.success else "FAILED"
            
            html_content += f"""
    <div class="experiment {status_class}">
        <h2>{result.experiment_name} - {status_text}</h2>
        <p><strong>Model:</strong> {result.model_type} | <strong>Target:</strong> {result.target_type}</p>
"""
            
            if result.success:
                metrics = result.metrics
                
                # Check for multi-output metrics
                if result.target_type == 'multi_output' and 'gender_head' in metrics and 'age_head' in metrics:
                    gender_metrics = metrics['gender_head']
                    age_metrics = metrics['age_head']
                    html_content += f"""
        <div class="metrics">
            <div class="metric" style="grid-column: span 2; border: 2px solid #4CAF50; padding: 15px;">
                <h3 style="margin-top: 0;">Gender Head</h3>
                <div><strong>Accuracy:</strong> {gender_metrics['accuracy']:.4f}</div>
                <div><strong>Precision:</strong> {gender_metrics['precision_weighted']:.4f}</div>
                <div><strong>Recall:</strong> {gender_metrics['recall_weighted']:.4f}</div>
                <div><strong>F1-Score:</strong> {gender_metrics['f1_weighted']:.4f}</div>
                {f"<div><strong>ROC-AUC:</strong> {gender_metrics['roc_auc']:.4f}</div>" if gender_metrics.get('roc_auc') else ""}
            </div>
            <div class="metric" style="grid-column: span 2; border: 2px solid #2196F3; padding: 15px;">
                <h3 style="margin-top: 0;">Age Head</h3>
                <div><strong>Accuracy:</strong> {age_metrics['accuracy']:.4f}</div>
                <div><strong>Precision:</strong> {age_metrics['precision_weighted']:.4f}</div>
                <div><strong>Recall:</strong> {age_metrics['recall_weighted']:.4f}</div>
                <div><strong>F1-Score:</strong> {age_metrics['f1_weighted']:.4f}</div>
            </div>
            <div class="metric"><strong>Training Time:</strong> {result.total_time:.2f}s</div>
        </div>
"""
                elif 'mae' in metrics:
                    # Regression task (overall + optional active/passive splits)
                    reg_parts = [
                        f"<div class=\"metric\"><strong>MAE:</strong> {metrics['mae']:.4f} years</div>",
                        f"<div class=\"metric\"><strong>RMSE:</strong> {metrics['rmse']:.4f} years</div>",
                        f"<div class=\"metric\"><strong>R²:</strong> {metrics['r2']:.4f}</div>",
                    ]
                    if 'task_type_metrics' in metrics:
                        tm = metrics['task_type_metrics']
                        active = tm.get('active') or {}
                        passive = tm.get('passive') or {}
                        if active or passive:
                            reg_parts.append(
                                f"<div class=\"metric\"><strong>Active MAE:</strong> "
                                f"{_format_float_metric(active.get('mae'))} years "
                                f"(n={active.get('num_samples', 'N/A')})</div>"
                            )
                            reg_parts.append(
                                f"<div class=\"metric\"><strong>Passive MAE:</strong> "
                                f"{_format_float_metric(passive.get('mae'))} years "
                                f"(n={passive.get('num_samples', 'N/A')})</div>"
                            )
                            reg_parts.append(
                                f"<div class=\"metric\"><strong>Active R²:</strong> "
                                f"{_format_float_metric(active.get('r2'))}</div>"
                            )
                            reg_parts.append(
                                f"<div class=\"metric\"><strong>Passive R²:</strong> "
                                f"{_format_float_metric(passive.get('r2'))}</div>"
                            )
                    reg_parts.append(
                        f"<div class=\"metric\"><strong>Training Time:</strong> {result.total_time:.2f}s</div>"
                    )
                    html_content += "        <div class=\"metrics\">\n"
                    for part in reg_parts:
                        html_content += f"            {part}\n"
                    html_content += "        </div>\n"
                else:
                    # Classification task
                    acc = metrics.get('accuracy')
                    prec = metrics.get('precision_weighted')
                    rec = metrics.get('recall_weighted')
                    f1 = metrics.get('f1_weighted')
                    acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else "N/A"
                    prec_str = f"{prec:.4f}" if isinstance(prec, (int, float)) else "N/A"
                    rec_str = f"{rec:.4f}" if isinstance(rec, (int, float)) else "N/A"
                    f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else "N/A"
                    
                    # Build accuracy section with task-specific metrics
                    accuracy_html = f"<div class=\"metric\"><strong>Combined Accuracy:</strong> {acc_str}</div>"
                    
                    # Add task-specific accuracies if available
                    if 'task_type_metrics' in metrics:
                        task_metrics = metrics['task_type_metrics']
                        if 'active' in task_metrics and 'passive' in task_metrics:
                            active_acc = task_metrics['active'].get('accuracy')
                            passive_acc = task_metrics['passive'].get('accuracy')
                            active_num_samples = task_metrics['active'].get('num_samples', 0)
                            passive_num_samples = task_metrics['passive'].get('num_samples', 0)
                            active_acc_str = f"{active_acc:.4f}" if isinstance(active_acc, (int, float)) else "N/A"
                            passive_acc_str = f"{passive_acc:.4f}" if isinstance(passive_acc, (int, float)) else "N/A"
                            accuracy_html += f"<div class=\"metric\"><strong>Active Task Accuracy:</strong> {active_acc_str} (n={active_num_samples})</div>"
                            accuracy_html += f"<div class=\"metric\"><strong>Passive Task Accuracy:</strong> {passive_acc_str} (n={passive_num_samples})</div>"
                            accuracy_html += f"<div class=\"metric\" style=\"font-size: 0.9em; color: #666;\"><em>Note: Combined accuracy is weighted by sample counts, not (Active + Passive)/2</em></div>"
                        elif 'active' in task_metrics:
                            active_acc = task_metrics['active'].get('accuracy')
                            active_num_samples = task_metrics['active'].get('num_samples', 0)
                            active_acc_str = f"{active_acc:.4f}" if isinstance(active_acc, (int, float)) else "N/A"
                            accuracy_html += f"<div class=\"metric\"><strong>Active Task Accuracy:</strong> {active_acc_str} (n={active_num_samples})</div>"
                        elif 'passive' in task_metrics:
                            passive_acc = task_metrics['passive'].get('accuracy')
                            passive_num_samples = task_metrics['passive'].get('num_samples', 0)
                            passive_acc_str = f"{passive_acc:.4f}" if isinstance(passive_acc, (int, float)) else "N/A"
                            accuracy_html += f"<div class=\"metric\"><strong>Passive Task Accuracy:</strong> {passive_acc_str} (n={passive_num_samples})</div>"
                    
                    html_content += f"""
        <div class="metrics">
            {accuracy_html}
            <div class="metric"><strong>Precision:</strong> {prec_str}</div>
            <div class="metric"><strong>Recall:</strong> {rec_str}</div>
            <div class="metric"><strong>F1-Score:</strong> {f1_str}</div>
            <div class="metric"><strong>Training Time:</strong> {result.total_time:.2f}s</div>
        </div>
"""
            else:
                html_content += f"""
        <p><strong>Error:</strong> {result.error}</p>
"""
            
            html_content += "    </div>\n"
        
        # Add comparison table
        if successful:
            # Check if any experiments are multi-output to determine table headers
            has_multi_output = any(r.target_type == 'multi_output' and 'gender_head' in r.metrics and 'age_head' in r.metrics 
                                   for r in successful)
            
            if has_multi_output:
                html_content += """
    <h2>Model Comparison</h2>
    <table>
        <tr>
            <th>Experiment</th>
            <th>Model</th>
            <th>Target</th>
            <th>Gender Head Acc</th>
            <th>Age Head Acc</th>
            <th>Gender F1</th>
            <th>Age F1</th>
            <th>Training Time (s)</th>
        </tr>
"""
                
                for result in successful:
                    metrics = result.metrics
                    if result.target_type == 'multi_output' and 'gender_head' in metrics and 'age_head' in metrics:
                        gender_metrics = metrics['gender_head']
                        age_metrics = metrics['age_head']
                        html_content += f"""
        <tr>
            <td>{result.experiment_name}</td>
            <td>{result.model_type}</td>
            <td>{result.target_type}</td>
            <td>{gender_metrics['accuracy']:.4f}</td>
            <td>{age_metrics['accuracy']:.4f}</td>
            <td>{gender_metrics['f1_weighted']:.4f}</td>
            <td>{age_metrics['f1_weighted']:.4f}</td>
            <td>{result.total_time:.2f}</td>
        </tr>
"""
                    elif 'mae' in metrics:
                        # Regression task
                        html_content += f"""
        <tr>
            <td>{result.experiment_name}</td>
            <td>{result.model_type}</td>
            <td>{result.target_type}</td>
            <td>{metrics['mae']:.4f} years</td>
            <td>{metrics['rmse']:.4f} years</td>
            <td>{metrics['r2']:.4f}</td>
            <td>N/A</td>
            <td>{result.total_time:.2f}</td>
        </tr>
"""
                    else:
                        # Single-output classification: show both columns but use same values
                        html_content += f"""
        <tr>
            <td>{result.experiment_name}</td>
            <td>{result.model_type}</td>
            <td>{result.target_type}</td>
            <td>{f"{metrics.get('accuracy'):.4f}" if isinstance(metrics.get('accuracy'), (int, float)) else "N/A"}</td>
            <td>{f"{metrics.get('accuracy'):.4f}" if isinstance(metrics.get('accuracy'), (int, float)) else "N/A"}</td>
            <td>{f"{metrics.get('f1_weighted'):.4f}" if isinstance(metrics.get('f1_weighted'), (int, float)) else "N/A"}</td>
            <td>{f"{metrics.get('f1_weighted'):.4f}" if isinstance(metrics.get('f1_weighted'), (int, float)) else "N/A"}</td>
            <td>{result.total_time:.2f}</td>
        </tr>
"""
            else:
                has_task_metrics = any('task_type_metrics' in r.metrics for r in successful)
                all_regression = bool(successful) and all('mae' in r.metrics for r in successful)
                regression_with_task_split = all_regression and has_task_metrics

                if regression_with_task_split:
                    html_content += """
    <h2>Model Comparison</h2>
    <table>
        <tr>
            <th>Experiment</th>
            <th>Model</th>
            <th>Target</th>
            <th>MAE (combined)</th>
            <th>Active MAE</th>
            <th>Passive MAE</th>
            <th>R² (combined)</th>
            <th>Training Time (s)</th>
        </tr>
"""
                    for result in successful:
                        m = result.metrics
                        tm = m.get('task_type_metrics', {})
                        active = tm.get('active') or {}
                        passive = tm.get('passive') or {}
                        html_content += f"""
        <tr>
            <td>{result.experiment_name}</td>
            <td>{result.model_type}</td>
            <td>{result.target_type}</td>
            <td>{m['mae']:.4f} years</td>
            <td>{_format_float_metric(active.get('mae'))} years</td>
            <td>{_format_float_metric(passive.get('mae'))} years</td>
            <td>{m['r2']:.4f}</td>
            <td>{result.total_time:.2f}</td>
        </tr>
"""
                elif has_task_metrics and not regression_with_task_split:
                    html_content += """
    <h2>Model Comparison</h2>
    <table>
        <tr>
            <th>Experiment</th>
            <th>Model</th>
            <th>Target</th>
            <th>Combined Accuracy</th>
            <th>Active Accuracy</th>
            <th>Passive Accuracy</th>
            <th>F1-Score</th>
            <th>Training Time (s)</th>
        </tr>
"""
                    for result in successful:
                        metrics = result.metrics
                        if 'mae' in metrics:
                            tm = metrics.get('task_type_metrics', {})
                            active = tm.get('active') or {}
                            passive = tm.get('passive') or {}
                            html_content += f"""
        <tr>
            <td>{result.experiment_name}</td>
            <td>{result.model_type}</td>
            <td>{result.target_type}</td>
            <td>{metrics['mae']:.4f} years</td>
            <td>{_format_float_metric(active.get('mae'))} years</td>
            <td>{_format_float_metric(passive.get('mae'))} years</td>
            <td>{metrics['r2']:.4f}</td>
            <td>{result.total_time:.2f}</td>
        </tr>
"""
                        else:
                            acc = metrics.get('accuracy')
                            f1 = metrics.get('f1_weighted')
                            acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else "N/A"
                            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else "N/A"
                            if 'task_type_metrics' in metrics:
                                task_metrics = metrics['task_type_metrics']
                                active_acc = task_metrics.get('active', {}).get('accuracy')
                                passive_acc = task_metrics.get('passive', {}).get('accuracy')
                                active_acc_str = f"{active_acc:.4f}" if isinstance(active_acc, (int, float)) else "N/A"
                                passive_acc_str = f"{passive_acc:.4f}" if isinstance(passive_acc, (int, float)) else "N/A"
                                html_content += f"""
        <tr>
            <td>{result.experiment_name}</td>
            <td>{result.model_type}</td>
            <td>{result.target_type}</td>
            <td>{acc_str}</td>
            <td>{active_acc_str}</td>
            <td>{passive_acc_str}</td>
            <td>{f1_str}</td>
            <td>{result.total_time:.2f}</td>
        </tr>
"""
                            else:
                                html_content += f"""
        <tr>
            <td>{result.experiment_name}</td>
            <td>{result.model_type}</td>
            <td>{result.target_type}</td>
            <td>{acc_str}</td>
            <td>N/A</td>
            <td>N/A</td>
            <td>{f1_str}</td>
            <td>{result.total_time:.2f}</td>
        </tr>
"""
                else:
                    html_content += """
    <h2>Model Comparison</h2>
    <table>
        <tr>
            <th>Experiment</th>
            <th>Model</th>
            <th>Target</th>
            <th>Accuracy</th>
            <th>F1-Score</th>
            <th>Training Time (s)</th>
        </tr>
"""
                    for result in successful:
                        metrics = result.metrics
                        if 'mae' in metrics:
                            html_content += f"""
        <tr>
            <td>{result.experiment_name}</td>
            <td>{result.model_type}</td>
            <td>{result.target_type}</td>
            <td>{metrics['mae']:.4f} years</td>
            <td>{metrics['r2']:.4f}</td>
            <td>{result.total_time:.2f}</td>
        </tr>
"""
                        else:
                            acc = metrics.get('accuracy')
                            f1 = metrics.get('f1_weighted')
                            acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else "N/A"
                            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else "N/A"
                            html_content += f"""
        <tr>
            <td>{result.experiment_name}</td>
            <td>{result.model_type}</td>
            <td>{result.target_type}</td>
            <td>{acc_str}</td>
            <td>{f1_str}</td>
            <td>{result.total_time:.2f}</td>
        </tr>
"""
            
            html_content += "    </table>\n"
        
        html_content += """
</body>
</html>
"""
        
        with open(html_file, 'w') as f:
            f.write(html_content)
    
    def _generate_visualizations(self, experiment_results: List[Dict[str, Any]]):
        """Generate visualization plots."""
        successful = [r for r in experiment_results if r.success]
        
        if not successful:
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Accuracy comparison
        self._plot_accuracy_comparison(successful)
        
        # 2. Training time comparison
        self._plot_training_time_comparison(successful)
        
        # 3. Model performance by target type
        self._plot_performance_by_target(successful)
    
    def _plot_accuracy_comparison(self, successful_results: List[Dict[str, Any]]):
        """Plot accuracy comparison across experiments."""
        experiments = [r.experiment_name for r in successful_results]
        
        # Check if we have multi-output experiments
        has_multi_output = any(r.target_type == 'multi_output' and 'gender_head' in r.metrics and 'age_head' in r.metrics 
                              for r in successful_results)
        
        if has_multi_output:
            # Create grouped bar chart for multi-output
            gender_accs = []
            age_accs = []
            single_accs = []
            single_exps = []
            
            for result in successful_results:
                if result.target_type == 'multi_output' and 'gender_head' in result.metrics:
                    gender_accs.append(result.metrics['gender_head']['accuracy'])
                    age_accs.append(result.metrics['age_head']['accuracy'])
                elif 'mae' in result.metrics:
                    # Regression task - use MAE (inverted for plotting, lower is better)
                    # For plotting purposes, we'll use 1/MAE normalized to [0,1] range
                    # Or just skip regression tasks in accuracy plots
                    pass  # Skip regression tasks in accuracy plots
                else:
                    single_accs.append(result.metrics.get('accuracy', 0))
                    single_exps.append(result.experiment_name)
            
            # Plot multi-output experiments
            if gender_accs:
                x = range(len(experiments))
                width = 0.35
                fig, ax = plt.subplots(figsize=(14, 6))
                
                multi_exps = [exp for exp, r in zip(experiments, successful_results) 
                            if r.target_type == 'multi_output' and 'gender_head' in r.metrics]
                multi_x = range(len(multi_exps))
                
                if multi_exps:
                    ax.bar([xi - width/2 for xi in multi_x], gender_accs, width, 
                          label='Gender Head', color='#4CAF50', alpha=0.7)
                    ax.bar([xi + width/2 for xi in multi_x], age_accs, width, 
                          label='Age Head', color='#2196F3', alpha=0.7)
                    ax.set_xticks(multi_x)
                    ax.set_xticklabels(multi_exps, rotation=45, ha='right')
                
                # Add single-output experiments if any
                if single_exps:
                    single_x_start = len(multi_exps) + 0.5
                    single_x = [single_x_start + i for i in range(len(single_exps))]
                    ax.bar(single_x, single_accs, width=0.7, label='Single Output', 
                          color='skyblue', alpha=0.7)
                    ax.set_xticks(list(ax.get_xticks()) + single_x)
                    ax.set_xticklabels(list(ax.get_xticklabels()) + 
                                     [exp[:30] for exp in single_exps], rotation=45, ha='right')
                
                ax.set_xlabel('Experiment', fontsize=12)
                ax.set_ylabel('Accuracy', fontsize=12)
                ax.set_title('Model Accuracy Comparison (Multi-Output Shows Separate Heads)', 
                           fontsize=16, fontweight='bold')
                ax.legend()
                ax.set_ylim(0, 1)
                
                # Add value labels
                for i, (g_acc, a_acc) in enumerate(zip(gender_accs, age_accs)):
                    ax.text(i - width/2, g_acc + 0.01, f'{g_acc:.3f}', 
                           ha='center', va='bottom', fontsize=8)
                    ax.text(i + width/2, a_acc + 0.01, f'{a_acc:.3f}', 
                           ha='center', va='bottom', fontsize=8)
                
                if single_accs:
                    for i, acc in enumerate(single_accs):
                        ax.text(single_x_start + i, acc + 0.01, f'{acc:.3f}', 
                               ha='center', va='bottom', fontsize=8)
            else:
                # Fallback to single output plotting (shouldn't happen, but just in case)
                # Filter to only classification experiments (skip regression)
                classification_results = [r for r in successful_results if 'accuracy' in r.metrics]
                if not classification_results:
                    # No classification experiments to plot
                    plt.close()
                    return
                
                experiments_filtered = [r.experiment_name for r in classification_results]
                accuracies = [r.metrics.get('accuracy', 0) for r in classification_results]
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(range(len(experiments_filtered)), accuracies, color='skyblue', alpha=0.7)
                ax.set_xlabel('Experiment', fontsize=12)
                ax.set_ylabel('Accuracy', fontsize=12)
                ax.set_title('Model Accuracy Comparison (Classification Only)', fontsize=16, fontweight='bold')
                ax.set_xticks(range(len(experiments_filtered)))
                ax.set_xticklabels(experiments_filtered, rotation=45, ha='right')
                ax.set_ylim(0, 1)
                for bar, acc in zip(bars, accuracies):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom')
        else:
            # Original single-output plotting
            # Filter to only classification experiments (skip regression which has 'mae' instead of 'accuracy')
            classification_results = [r for r in successful_results if 'accuracy' in r.metrics]
            if not classification_results:
                # No classification experiments to plot
                plt.close()
                return
            
            experiments_filtered = [r.experiment_name for r in classification_results]
            accuracies = [r.metrics.get('accuracy', 0) for r in classification_results]
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(range(len(experiments_filtered)), accuracies, color='skyblue', alpha=0.7)
            ax.set_xlabel('Experiment', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title('Model Accuracy Comparison (Classification Only)', fontsize=16, fontweight='bold')
            ax.set_xticks(range(len(experiments_filtered)))
            ax.set_xticklabels(experiments_filtered, rotation=45, ha='right')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'accuracy_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_training_time_comparison(self, successful_results: List[Dict[str, Any]]):
        """Plot training time comparison."""
        experiments = [r.experiment_name for r in successful_results]
        times = [r.total_time for r in successful_results]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(experiments, times, color='lightcoral', alpha=0.7)
        plt.title('Training Time Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Experiment', fontsize=12)
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                    f'{time:.1f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_time_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_by_target(self, successful_results: List[Dict[str, Any]]):
        """Plot performance metrics by target type."""
        # Group by target type
        target_groups = {}
        for result in successful_results:
            target = result.target_type
            if target not in target_groups:
                target_groups[target] = []
            target_groups[target].append(result)
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy by target type (skip regression tasks)
        targets = list(target_groups.keys())
        accuracies = []
        valid_targets = []
        for target in targets:
            # Only include targets that have accuracy metrics (classification tasks)
            target_accs = [r.metrics.get('accuracy', 0) 
                          for r in target_groups[target] 
                          if 'accuracy' in r.metrics]
            if target_accs:
                accuracies.append(max(target_accs))
                valid_targets.append(target)
        
        if valid_targets:
            axes[0].bar(valid_targets, accuracies, color=['skyblue', 'lightcoral'], alpha=0.7)
            axes[0].set_title('Best Accuracy by Target Type', fontweight='bold')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_ylim(0, 1)
        else:
            axes[0].text(0.5, 0.5, 'No classification tasks to display', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Best Accuracy by Target Type', fontweight='bold')
        
        # F1-score by target type (skip regression tasks)
        f1_scores = []
        valid_targets_f1 = []
        for target in targets:
            # Only include targets that have f1_weighted metrics (classification tasks)
            target_f1s = [r.metrics.get('f1_weighted', 0) 
                         for r in target_groups[target] 
                         if 'f1_weighted' in r.metrics]
            if target_f1s:
                f1_scores.append(max(target_f1s))
                valid_targets_f1.append(target)
        
        if valid_targets_f1:
            axes[1].bar(valid_targets_f1, f1_scores, color=['skyblue', 'lightcoral'], alpha=0.7)
            axes[1].set_title('Best F1-Score by Target Type', fontweight='bold')
            axes[1].set_ylabel('F1-Score')
            axes[1].set_ylim(0, 1)
        else:
            axes[1].text(0.5, 0.5, 'No classification tasks to display', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Best F1-Score by Target Type', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_by_target.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_csv_summary(self, experiment_results: List[Dict[str, Any]]):
        """Generate CSV summary for easy analysis."""
        successful = [r for r in experiment_results if r.success]
        
        if not successful:
            return
        
        # Prepare data for CSV
        data = []
        for result in successful:
            metrics = result.metrics
            
            # Check for multi-output metrics
            if result.target_type == 'multi_output' and 'gender_head' in metrics and 'age_head' in metrics:
                gender_metrics = metrics['gender_head']
                age_metrics = metrics['age_head']
                data.append({
                    'experiment_name': result.experiment_name,
                    'model_type': result.model_type,
                    'target_type': result.target_type,
                    'gender_head_accuracy': gender_metrics['accuracy'],
                    'gender_head_precision': gender_metrics['precision_weighted'],
                    'gender_head_recall': gender_metrics['recall_weighted'],
                    'gender_head_f1': gender_metrics['f1_weighted'],
                    'gender_head_roc_auc': gender_metrics.get('roc_auc', None),
                    'age_head_accuracy': age_metrics['accuracy'],
                    'age_head_precision': age_metrics['precision_weighted'],
                    'age_head_recall': age_metrics['recall_weighted'],
                    'age_head_f1': age_metrics['f1_weighted'],
                    # Keep primary metrics for backward compatibility
                    # Use averaged metrics if available, otherwise calculate average from heads
                    'accuracy': metrics.get('accuracy', (gender_metrics['accuracy'] + age_metrics['accuracy']) / 2.0),
                    'precision_weighted': metrics.get('precision_weighted', (gender_metrics['precision_weighted'] + age_metrics['precision_weighted']) / 2.0),
                    'recall_weighted': metrics.get('recall_weighted', (gender_metrics['recall_weighted'] + age_metrics['recall_weighted']) / 2.0),
                    'f1_weighted': metrics.get('f1_weighted', (gender_metrics['f1_weighted'] + age_metrics['f1_weighted']) / 2.0),
                    'training_time': result.total_time,
                    'evaluation_time': result.evaluation_time
                })
            elif 'mae' in metrics:
                # Regression task
                row = {
                    'experiment_name': result.experiment_name,
                    'model_type': result.model_type,
                    'target_type': result.target_type,
                    'mae': metrics.get('mae', None),
                    'rmse': metrics.get('rmse', None),
                    'r2': metrics.get('r2', None),
                    'mae_normalized': metrics.get('mae_normalized', None),
                    'training_time': result.total_time,
                    'evaluation_time': result.evaluation_time,
                }
                if 'task_type_metrics' in metrics:
                    tm = metrics['task_type_metrics']
                    for split in ('active', 'passive'):
                        sub = tm.get(split) or {}
                        row[f'{split}_mae'] = sub.get('mae', None)
                        row[f'{split}_rmse'] = sub.get('rmse', None)
                        row[f'{split}_r2'] = sub.get('r2', None)
                        row[f'{split}_num_samples'] = sub.get('num_samples', None)
                data.append(row)
            else:
                # Classification task
                row = {
                    'experiment_name': result.experiment_name,
                    'model_type': result.model_type,
                    'target_type': result.target_type,
                    'combined_accuracy': metrics.get('accuracy', None),
                    'precision_weighted': metrics.get('precision_weighted', None),
                    'recall_weighted': metrics.get('recall_weighted', None),
                    'f1_weighted': metrics.get('f1_weighted', None),
                    'training_time': result.total_time,
                    'evaluation_time': result.evaluation_time
                }
                
                # Add task-specific accuracies if available
                if 'task_type_metrics' in metrics:
                    task_metrics = metrics['task_type_metrics']
                    row['active_accuracy'] = task_metrics.get('active', {}).get('accuracy', None)
                    row['passive_accuracy'] = task_metrics.get('passive', {}).get('accuracy', None)
                else:
                    row['active_accuracy'] = None
                    row['passive_accuracy'] = None
                
                data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        csv_file = os.path.join(self.output_dir, 'experiment_results.csv')
        df.to_csv(csv_file, index=False)
        
        print(f"CSV summary saved to: {csv_file}")
