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
            successful = [r for r in experiment_results if r['success']]
            failed = [r for r in experiment_results if not r['success']]
            
            f.write(f"Total Experiments: {len(experiment_results)}\n")
            f.write(f"Successful: {len(successful)}\n")
            f.write(f"Failed: {len(failed)}\n\n")
            
            # Successful experiments summary
            if successful:
                f.write("SUCCESSFUL EXPERIMENTS:\n")
                f.write("-" * 30 + "\n")
                
                for result in successful:
                    f.write(f"\nExperiment: {result['experiment_name']}\n")
                    f.write(f"Model: {result['model_type']}\n")
                    f.write(f"Target: {result['target_type']}\n")
                    f.write(f"Accuracy: {result['evaluation_results']['metrics']['accuracy']:.4f}\n")
                    f.write(f"F1-Score: {result['evaluation_results']['metrics']['f1_weighted']:.4f}\n")
                    f.write(f"Training Time: {result['total_time']:.2f}s\n")
            
            # Failed experiments
            if failed:
                f.write("\n\nFAILED EXPERIMENTS:\n")
                f.write("-" * 20 + "\n")
                
                for result in failed:
                    f.write(f"\nExperiment: {result['experiment_name']}\n")
                    f.write(f"Error: {result['error']}\n")
    
    def _generate_html_report(self, experiment_results: List[Dict[str, Any]]):
        """Generate HTML report."""
        html_file = os.path.join(self.output_dir, 'experiment_report.html')
        
        successful = [r for r in experiment_results if r['success']]
        
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
        <p>Total Experiments: {len(experiment_results)} | Successful: {len(successful)} | Failed: {len([r for r in experiment_results if not r['success']])}</p>
    </div>
"""
        
        # Add experiment details
        for result in experiment_results:
            status_class = "success" if result['success'] else "failure"
            status_text = "SUCCESS" if result['success'] else "FAILED"
            
            html_content += f"""
    <div class="experiment {status_class}">
        <h2>{result['experiment_name']} - {status_text}</h2>
        <p><strong>Model:</strong> {result['model_type']} | <strong>Target:</strong> {result['target_type']}</p>
"""
            
            if result['success']:
                metrics = result['evaluation_results']['metrics']
                html_content += f"""
        <div class="metrics">
            <div class="metric"><strong>Accuracy:</strong> {metrics['accuracy']:.4f}</div>
            <div class="metric"><strong>Precision:</strong> {metrics['precision_weighted']:.4f}</div>
            <div class="metric"><strong>Recall:</strong> {metrics['recall_weighted']:.4f}</div>
            <div class="metric"><strong>F1-Score:</strong> {metrics['f1_weighted']:.4f}</div>
            <div class="metric"><strong>Training Time:</strong> {result['total_time']:.2f}s</div>
        </div>
"""
            else:
                html_content += f"""
        <p><strong>Error:</strong> {result['error']}</p>
"""
            
            html_content += "    </div>\n"
        
        # Add comparison table
        if successful:
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
                metrics = result['evaluation_results']['metrics']
                html_content += f"""
        <tr>
            <td>{result['experiment_name']}</td>
            <td>{result['model_type']}</td>
            <td>{result['target_type']}</td>
            <td>{metrics['accuracy']:.4f}</td>
            <td>{metrics['f1_weighted']:.4f}</td>
            <td>{result['total_time']:.2f}</td>
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
        successful = [r for r in experiment_results if r['success']]
        
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
        experiments = [r['experiment_name'] for r in successful_results]
        accuracies = [r['evaluation_results']['metrics']['accuracy'] for r in successful_results]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(experiments, accuracies, color='skyblue', alpha=0.7)
        plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Experiment', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'accuracy_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_training_time_comparison(self, successful_results: List[Dict[str, Any]]):
        """Plot training time comparison."""
        experiments = [r['experiment_name'] for r in successful_results]
        times = [r['total_time'] for r in successful_results]
        
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
            target = result['target_type']
            if target not in target_groups:
                target_groups[target] = []
            target_groups[target].append(result)
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy by target type
        targets = list(target_groups.keys())
        accuracies = [max([r['evaluation_results']['metrics']['accuracy'] 
                          for r in target_groups[target]]) for target in targets]
        
        axes[0].bar(targets, accuracies, color=['skyblue', 'lightcoral'], alpha=0.7)
        axes[0].set_title('Best Accuracy by Target Type', fontweight='bold')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim(0, 1)
        
        # F1-score by target type
        f1_scores = [max([r['evaluation_results']['metrics']['f1_weighted'] 
                         for r in target_groups[target]]) for target in targets]
        
        axes[1].bar(targets, f1_scores, color=['skyblue', 'lightcoral'], alpha=0.7)
        axes[1].set_title('Best F1-Score by Target Type', fontweight='bold')
        axes[1].set_ylabel('F1-Score')
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_by_target.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_csv_summary(self, experiment_results: List[Dict[str, Any]]):
        """Generate CSV summary for easy analysis."""
        successful = [r for r in experiment_results if r['success']]
        
        if not successful:
            return
        
        # Prepare data for CSV
        data = []
        for result in successful:
            metrics = result['evaluation_results']['metrics']
            data.append({
                'experiment_name': result['experiment_name'],
                'model_type': result['model_type'],
                'target_type': result['target_type'],
                'accuracy': metrics['accuracy'],
                'precision_weighted': metrics['precision_weighted'],
                'recall_weighted': metrics['recall_weighted'],
                'f1_weighted': metrics['f1_weighted'],
                'training_time': result['total_time'],
                'evaluation_time': result['evaluation_results']['evaluation_time']
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        csv_file = os.path.join(self.output_dir, 'experiment_results.csv')
        df.to_csv(csv_file, index=False)
        
        print(f"CSV summary saved to: {csv_file}")
