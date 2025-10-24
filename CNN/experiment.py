"""
Experiment Class for EEG Classification Framework
Represents a single experiment with its complete lifecycle.
"""

import time
import os
import sys
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import torch
import numpy as np

from models import ModelFactory
from trainer import EEGTrainer
from evaluator import EEGEvaluator, EvaluationResults
from config import ExperimentConfig, TrainingConfig, DataConfig, ModelConfig, SystemConfig
from utils import safe_json_dump, convert_numpy_types

# Import data processing modules
sys.path.append('/home/mojtabam/projects/def-aghodsib/mojtabam/EEG/data_processing')
from eeg_dataset import EEGDataLoader
from target_transforms import gender_classification_transform, age_classification_transform, combined_gender_age_classification_transform

@dataclass
class ExperimentResult:
    """Represents the result of an experiment."""
    experiment_name: str
    model_type: str
    target_type: str
    success: bool
    total_time: float
    training_time: float
    evaluation_time: float
    description: Optional[str] = None  # Human-readable description of the experiment
    metrics: Optional[Dict[str, Any]] = None
    training_metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    model_checkpoint_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'experiment_name': self.experiment_name,
            'model_type': self.model_type,
            'target_type': self.target_type,
            'success': self.success,
            'total_time': self.total_time,
            'training_time': self.training_time,
            'evaluation_time': self.evaluation_time,
            'description': self.description,
            'metrics': self.metrics,
            'training_metrics': self.training_metrics,
            'error': self.error,
            'model_checkpoint_path': self.model_checkpoint_path
        }

class ExperimentLogger:
    """Handles logging and reporting for experiments."""
    
    def __init__(self, results_dir: str = 'experiment_results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Training metrics storage
        self.training_metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': [],
            'epoch_times': []
        }
    
    def log_experiment_start(self, config: ExperimentConfig):
        """Log the start of an experiment."""
        print(f"\n{'='*80}")
        print(f"STARTING EXPERIMENT: {config.name}")
        print(f"Model: {config.model_type} | Target: {config.target_type}")
        print(f"Channels: {config.model_config.num_channels}")
        print(f"{'='*80}")
    
    def log_experiment_progress(self, epoch: int, total_epochs: int, train_loss: float, 
                              val_loss: float, train_acc: float, val_acc: float, 
                              lr: float = None, epoch_time: float = None):
        """Log training progress for the experiment and store metrics."""
        # Store metrics
        self.training_metrics['train_loss'].append(train_loss)
        self.training_metrics['val_loss'].append(val_loss)
        self.training_metrics['train_acc'].append(train_acc)
        self.training_metrics['val_acc'].append(val_acc)
        if lr is not None:
            self.training_metrics['learning_rates'].append(lr)
        if epoch_time is not None:
            self.training_metrics['epoch_times'].append(epoch_time)
        
        # Log progress
        if lr is not None and epoch_time is not None:
            print(f"  → Epoch {epoch+1:3d}/{total_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {lr:.6f} | Time: {epoch_time:.2f}s")
        else:
            print(f"  → Epoch {epoch+1:3d}/{total_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get the stored training metrics."""
        return self.training_metrics.copy()
    
    def save_training_metrics(self, experiment_name: str):
        """Save training metrics to JSON file in experiment-specific directory."""
        # Create experiment-specific directory
        exp_dir = os.path.join(self.results_dir, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        metrics_file = os.path.join(exp_dir, f'{experiment_name}_training_metrics.json')
        safe_json_dump(convert_numpy_types(self.training_metrics), metrics_file)
        print(f"   Training metrics saved to: {metrics_file}")
    
    def log_evaluation_start(self, model_name: str, target_type: str):
        """Log the start of evaluation."""
        print(f"Evaluating {model_name} on {target_type} classification...")
    
    def log_evaluation_success(self, results):
        """Log successful evaluation completion."""
        print(f"✅ Evaluation completed")
        print(f"   Accuracy: {results.metrics['accuracy']:.4f}")
        print(f"   F1-Score: {results.metrics['f1_weighted']:.4f}")
        print(f"   Evaluation time: {results.evaluation_time:.2f}s")
    
    def save_evaluation_results(self, results, experiment_name: str):
        """Save evaluation results to files."""
        eval_dir = os.path.join(self.results_dir, experiment_name)
        os.makedirs(eval_dir, exist_ok=True)
        
        # Save metrics as JSON
        metrics_file = os.path.join(eval_dir, f'{results.model_name}_{results.target_type}_metrics.json')
        safe_json_dump(convert_numpy_types(results.to_dict()), metrics_file)
        
        # Save predictions
        predictions_file = os.path.join(eval_dir, f'{results.model_name}_{results.target_type}_predictions.json')
        predictions_data = convert_numpy_types({
            'predictions': results.predictions,
            'true_labels': results.true_labels,
            'probabilities': results.probabilities,
            'class_names': results.class_names
        })
        safe_json_dump(predictions_data, predictions_file)
        
        # Save confusion matrix plot
        self._plot_confusion_matrix(results, eval_dir)
        
        print(f"   Evaluation results saved to: {eval_dir}")
    
    def _plot_confusion_matrix(self, results, save_dir: str):
        """Plot and save confusion matrix."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = np.array(results.metrics['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=results.class_names,
                   yticklabels=results.class_names)
        plt.title(f'{results.model_name} - {results.target_type.title()} Classification')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plot_file = os.path.join(save_dir, f'{results.model_name}_{results.target_type}_confusion_matrix.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    def log_experiment_success(self, result: ExperimentResult):
        """Log successful experiment completion."""
        print(f"\n✅ Experiment '{result.experiment_name}' completed successfully!")
        print(f"   Total time: {result.total_time:.2f}s")
        print(f"   Training time: {result.training_time:.2f}s")
        print(f"   Evaluation time: {result.evaluation_time:.2f}s")
        if result.metrics:
            print(f"   Accuracy: {result.metrics.get('accuracy', 'N/A'):.4f}")
    
    def log_experiment_failure(self, result: ExperimentResult):
        """Log failed experiment."""
        print(f"\n❌ Experiment '{result.experiment_name}' failed!")
        print(f"   Error: {result.error}")
        print(f"   Time: {result.total_time:.2f}s")
    
    def save_experiment_result(self, result: ExperimentResult):
        """Save experiment result to file in experiment-specific directory."""
        if result.success:
            # Create experiment-specific directory
            exp_dir = os.path.join(self.results_dir, result.experiment_name)
            os.makedirs(exp_dir, exist_ok=True)
            
            result_file = os.path.join(
                exp_dir, 
                f"{result.experiment_name}_result.json"
            )
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            safe_json_dump(convert_numpy_types(result.to_dict()), result_file)
            print(f"   Result saved to: {result_file}")

class Experiment:
    """
    Represents a single experiment with its complete lifecycle.
    """
    
    def __init__(self, config: ExperimentConfig, system_config: SystemConfig = None):
        """
        Initialize an experiment. Data loaders will be created during setup.
        
        Args:
            config: Experiment configuration
            system_config: System-wide configuration
        """
        self.config = config
        self.system_config = system_config or SystemConfig()
        self.logger = ExperimentLogger(self.system_config.results_dir)
        self.model = None
        self.trainer = None
        self.evaluator = None
        self.result = None
        self.data_loaders = None
        self.is_cross_validation = False
    
    def setup(self):
        """
        Setup the experiment components (data loaders, model, trainer, evaluator).
        """
        print(f"Setting up experiment: {self.config.name}")
        
        # Create data loaders using the DataConfig
        self._create_data_loaders()
        
        # Create model using ModelFactory
        self.model = ModelFactory.create_model(
            self.config.model_type,
            **self.config.model_config.to_dict()
        )
        
        # Create trainer
        self.trainer = EEGTrainer(self.model, self.config.training_config, self.config.name)
        
        # Create evaluator
        self.evaluator = EEGEvaluator(self.model, self.config.target_type)
        
        print(f"✅ Experiment setup complete")
    
    def _create_data_loaders(self):
        """
        Create data loaders using the experiment's DataConfig.
        """
        data_config = self.config.data_config
        target_type = self.config.target_type
        
        # Select appropriate transforms based on target type
        gender_transform = None
        age_transform = None
        combined_transform = None
        
        if target_type == 'gender':
            gender_transform = gender_classification_transform
        elif target_type == 'age':
            age_transform = age_classification_transform
        elif target_type == 'combined':
            combined_transform = combined_gender_age_classification_transform
        
        # Handle different experiment types based on data config
        if data_config.use_cross_validation and data_config.train_task_type and data_config.val_test_task_type:
            # Cross-task cross-validation experiments
            print(f"Creating cross-task cross-validation loaders: {data_config.train_task_type} -> {data_config.val_test_task_type}")
            self.data_loaders = EEGDataLoader.create_cross_task_cross_validation_loaders(
                pickle_dir=data_config.pickle_dir,
                train_task_type=data_config.train_task_type,
                val_test_task_type=data_config.val_test_task_type,
                batch_size=data_config.batch_size,
                num_workers=data_config.num_workers,
                random_seed=data_config.random_seed,
                target_type=target_type,
                gender_transform=gender_transform,
                age_transform=age_transform,
                transform=combined_transform
            )
            self.is_cross_validation = True
            
        elif data_config.use_cross_validation:
            # Standard cross-validation experiments
            print(f"Creating cross-validation loaders for {target_type} classification")
            self.data_loaders = EEGDataLoader.create_cross_validation_loaders(
                pickle_dir=data_config.pickle_dir,
                batch_size=data_config.batch_size,
                num_workers=data_config.num_workers,
                random_seed=data_config.random_seed,
                task_type=data_config.task_type,
                target_type=target_type,
                gender_transform=gender_transform,
                age_transform=age_transform,
                transform=combined_transform
            )
            self.is_cross_validation = True
            
        elif data_config.train_task_type and data_config.val_test_task_type:
            # Cross-task experiments
            print(f"Creating cross-task loaders: {data_config.train_task_type} -> {data_config.val_test_task_type}")
            self.data_loaders = EEGDataLoader.create_cross_task_loaders(
                pickle_dir=data_config.pickle_dir,
                train_task_type=data_config.train_task_type,
                val_test_task_type=data_config.val_test_task_type,
                batch_size=data_config.batch_size,
                num_workers=data_config.num_workers,
                random_seed=data_config.random_seed,
                target_type=target_type,
                gender_transform=gender_transform,
                age_transform=age_transform,
                transform=combined_transform
            )
            self.is_cross_validation = False
            
        else:
            # Standard train/val/test split experiments
            print(f"Creating standard loaders for {target_type} classification")
            self.data_loaders = EEGDataLoader.create_train_val_test_loaders(
                pickle_dir=data_config.pickle_dir,
                batch_size=data_config.batch_size,
                num_workers=data_config.num_workers,
                random_seed=data_config.random_seed,
                task_type=data_config.task_type,
                target_type=target_type,
                gender_transform=gender_transform,
                age_transform=age_transform,
                transform=combined_transform
            )
            self.is_cross_validation = False

    def run(self) -> ExperimentResult:
        """
        Run the complete experiment.
        Handles both single experiments and cross-validation experiments.
        
        Returns:
            ExperimentResult containing all experiment information
        """
        if self.model is None or self.trainer is None or self.evaluator is None:
            raise ValueError("Experiment not set up. Call setup() first.")
        
        self.logger.log_experiment_start(self.config)
        experiment_start = time.time()
        
        try:
            if self.is_cross_validation:
                # Handle cross-validation experiments
                self.result = self._run_cross_validation()
            else:
                # Handle single experiments
                self.result = self._run_single_experiment(experiment_start)
            
            self.logger.log_experiment_success(self.result)
            
        except Exception as e:
            total_time = time.time() - experiment_start
            
            # Create failed result
            self.result = ExperimentResult(
                experiment_name=self.config.name,
                model_type=self.config.model_type,
                target_type=self.config.target_type,
                success=False,
                total_time=total_time,
                training_time=0.0,
                evaluation_time=0.0,
                description=self.config.description,
                error=str(e)
            )
            
            self.logger.log_experiment_failure(self.result)
        
        # Save result
        self.logger.save_experiment_result(self.result)
        
        return self.result
    
    def _run_single_experiment(self, experiment_start: float) -> ExperimentResult:
        """Run a single experiment (train/val/test split)."""
        # Extract data loaders from injected data
        self.train_loader, self.val_loader, self.test_loader = self.data_loaders
        
        # Training phase
        training_start = time.time()
        self._train()
        training_time = time.time() - training_start
        
        # Evaluation phase
        evaluation_start = time.time()
        evaluation_results = self._evaluate()
        evaluation_time = time.time() - evaluation_start
        
        total_time = time.time() - experiment_start
        
        # Create successful result
        result = ExperimentResult(
            experiment_name=self.config.name,
            model_type=self.config.model_type,
            target_type=self.config.target_type,
            success=True,
            total_time=total_time,
            training_time=training_time,
            evaluation_time=evaluation_time,
            description=self.config.description,
            metrics=evaluation_results.metrics,
            model_checkpoint_path=self._get_checkpoint_path()
        )
        
        # Add training metrics to result
        result.training_metrics = self.logger.get_training_metrics()
        
        return result
    
    def _run_cross_validation(self) -> ExperimentResult:
        """Run cross-validation experiment with multiple folds."""
        print(f"Running {len(self.data_loaders)} folds for cross-validation...")
        fold_results = []
        
        for fold_idx, (train_loader, val_loader, test_loader) in enumerate(self.data_loaders):
            print(f"  Fold {fold_idx + 1}/{len(self.data_loaders)}")
            
            # Store data loaders for this fold
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader
            
            # Run single fold
            fold_start = time.time()
            
            # Training phase
            training_start = time.time()
            self._train()
            training_time = time.time() - training_start
            
            # Evaluation phase
            evaluation_start = time.time()
            evaluation_results = self._evaluate()
            evaluation_time = time.time() - evaluation_start
            
            fold_time = time.time() - fold_start
            
            # Create fold result
            fold_result = ExperimentResult(
                experiment_name=f"{self.config.name}_fold_{fold_idx + 1}",
                model_type=self.config.model_type,
                target_type=self.config.target_type,
                success=True,
                total_time=fold_time,
                training_time=training_time,
                evaluation_time=evaluation_time,
                description=f"Fold {fold_idx + 1} of {self.config.description}",
                metrics=evaluation_results.metrics,
                model_checkpoint_path=self._get_checkpoint_path()
            )
            
            fold_results.append(fold_result)
        
        # Aggregate fold results
        if fold_results and all(r.success for r in fold_results):
            # Calculate average metrics across folds
            avg_metrics = {}
            for key in fold_results[0].metrics.keys():
                if isinstance(fold_results[0].metrics[key], (int, float)):
                    # Numeric metrics - calculate average
                    avg_metrics[key] = sum(r.metrics[key] for r in fold_results) / len(fold_results)
                else:
                    # Non-numeric metrics (like lists) - use first fold's value
                    avg_metrics[key] = fold_results[0].metrics[key]
            
            # Create aggregated result
            aggregated_result = ExperimentResult(
                experiment_name=self.config.name,
                model_type=self.config.model_type,
                target_type=self.config.target_type,
                success=True,
                total_time=sum(r.total_time for r in fold_results),
                training_time=sum(r.training_time for r in fold_results),
                evaluation_time=sum(r.evaluation_time for r in fold_results),
                description=self.config.description,
                metrics=avg_metrics,
                training_metrics=fold_results[0].training_metrics,  # Use first fold's training metrics
                model_checkpoint_path=fold_results[0].model_checkpoint_path
            )
            
            print(f"✅ Cross-validation completed: {len(fold_results)} folds")
            return aggregated_result
        else:
            print(f"❌ Cross-validation failed")
            return ExperimentResult(
                experiment_name=self.config.name,
                model_type=self.config.model_type,
                target_type=self.config.target_type,
                success=False,
                total_time=0.0,
                training_time=0.0,
                evaluation_time=0.0,
                description=self.config.description,
                error="Cross-validation failed"
            )
    
    def _train(self):
        """Train the model."""
        print(f"Training {self.config.model_type} for {self.config.target_type} classification...")
        
        # Create progress callback
        def progress_callback(epoch, total_epochs, train_loss, val_loss, train_acc, val_acc, lr, epoch_time):
            self.logger.log_experiment_progress(epoch, total_epochs, train_loss, val_loss, train_acc, val_acc, lr, epoch_time)
        
        # Train the model with progress callback
        self.trainer.train(
            self.train_loader,
            self.val_loader,
            progress_callback=progress_callback,
            experiment_results_dir=self.logger.results_dir
        )
        
        print(f"✅ Training completed")
        
        # Save training metrics
        self.logger.save_training_metrics(self.config.name)
    
    def _evaluate(self) -> EvaluationResults:
        """Evaluate the model."""
        self.logger.log_evaluation_start(self.config.model_type, self.config.target_type)
        
        # Load best model
        checkpoint_path = self._get_checkpoint_path()
        if os.path.exists(checkpoint_path):
            self.evaluator.load_checkpoint(checkpoint_path)
        
        # Evaluate on test set
        results = self.evaluator.evaluate(self.test_loader)
        
        # Log and save results
        self.logger.log_evaluation_success(results)
        self.logger.save_evaluation_results(results, self.config.name)
        
        return results
    
    def _get_checkpoint_path(self) -> str:
        """Get the path to the best model checkpoint."""
        # Create experiment-specific checkpoint directory
        exp_checkpoint_dir = os.path.join(
            self.logger.results_dir,
            'checkpoints'
        )
        os.makedirs(exp_checkpoint_dir, exist_ok=True)
        
        return os.path.join(
            exp_checkpoint_dir,
            f'{self.config.name}_best.pth'
        )

def run_experiments(experiments: List[ExperimentConfig], 
                   system_config: SystemConfig = None,
                   random_seed: int = 42,
                   generate_reports: bool = True) -> List[ExperimentResult]:
    """
    Unified experiment orchestrator that coordinates multiple experiments.
    
    Each experiment is now self-contained and handles its own:
    - Data loader creation
    - Model setup
    - Training and evaluation
    - Cross-validation (if applicable)
    
    This function just orchestrates the execution of multiple experiments.
    
    Args:
        experiments: List of experiment configurations
        system_config: System-wide configuration
        random_seed: Random seed for reproducibility
        generate_reports: Whether to generate comparison reports
        
    Returns:
        List of ExperimentResult objects
    """
    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Use default system config if not provided
    system_config = system_config or SystemConfig()
    
    print(f"\n{'='*80}")
    print(f"STARTING EXPERIMENTS")
    print(f"Number of experiments: {len(experiments)}")
    print(f"Random seed: {random_seed}")
    print(f"Results directory: {system_config.results_dir}")
    print(f"Generate reports: {generate_reports}")
    print(f"{'='*80}")
    
    results = []
    
    for i, config in enumerate(experiments):
        print(f"\nProgress: {i+1}/{len(experiments)}")
        print(f"Experiment: {config.name}")
        
        try:
            # Create and run experiment (data loaders created internally)
            experiment = Experiment(config, system_config)
            experiment.setup()  # Setup data loaders, model, trainer, evaluator
            result = experiment.run()  # Run the experiment
            results.append(result)
            
            if result.success:
                print(f"✅ Experiment completed: {config.name}")
            else:
                print(f"❌ Experiment failed: {config.name} - {result.error}")
                
        except Exception as e:
            print(f"❌ Experiment {config.name} failed: {str(e)}")
            # Create failed result
            failed_result = ExperimentResult(
                experiment_name=config.name,
                model_type=config.model_type,
                target_type=config.target_type,
                success=False,
                total_time=0.0,
                training_time=0.0,
                evaluation_time=0.0,
                description=config.description,
                error=str(e)
            )
            results.append(failed_result)
    
    # Save summary of all results
    _save_experiment_summary(results, system_config.results_dir)
    
    # Generate reports if requested
    if generate_reports:
        from report_generator import ReportGenerator
        report_generator = ReportGenerator(system_config.results_dir, system_config.reports_dir)
        report_generator.generate_all_reports(results)
        print(f"Reports generated in: {system_config.reports_dir}")
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"Results saved to: {system_config.results_dir}")
    print(f"{'='*80}")
    
    return results

def _save_experiment_summary(results: List[ExperimentResult], results_dir: str):
    """Save summary of all experiment results."""
    # Save individual results in experiment-specific directories
    for result in results:
        if result.success:
            # Create experiment-specific directory
            exp_dir = os.path.join(results_dir, result.experiment_name)
            os.makedirs(exp_dir, exist_ok=True)
            
            result_file = os.path.join(
                exp_dir, 
                f"{result.experiment_name}_complete.json"
            )
            safe_json_dump(convert_numpy_types(result.to_dict()), result_file)
    
    # Save summary
    summary = {
        'total_experiments': len(results),
        'successful_experiments': sum(1 for r in results if r.success),
        'failed_experiments': sum(1 for r in results if not r.success),
        'experiments': [r.to_dict() for r in results]
    }
    
    summary_file = os.path.join(results_dir, 'experiment_summary.json')
    safe_json_dump(convert_numpy_types(summary), summary_file)
    
    # Generate comparison report for successful experiments
    successful_results = [r for r in results if r.success]
    if successful_results:
        _generate_comparison_report(successful_results, results_dir)

def _generate_comparison_report(successful_results: List[ExperimentResult], results_dir: str):
    """Generate comparison report for successful experiments."""
    # Create evaluation results for comparison
    eval_results = []
    for result in successful_results:
        eval_result = EvaluationResults(
            result.model_type, 
            result.target_type
        )
        eval_result.metrics = result.metrics
        eval_result.evaluation_time = result.evaluation_time
        eval_results.append(eval_result)
    
    # Generate comparison
    from evaluator import compare_models
    comparison = compare_models(eval_results)
    
    # Add experiment names and descriptions to comparison
    for i, result in enumerate(successful_results):
        if i < len(comparison['models']):
            comparison['models'][i]['experiment_name'] = result.experiment_name
            # Use description from experiment result
            comparison['models'][i]['description'] = result.description or 'Custom experiment configuration'
    
    # Save comparison
    comparison_file = os.path.join(results_dir, 'model_comparison.json')
    safe_json_dump(convert_numpy_types(comparison), comparison_file)
    
    # Print comparison with description column
    print("\n" + "="*120)
    print("MODEL COMPARISON")
    print("="*120)
    print(f"{'Model':<20} {'Target':<10} {'Accuracy':<10} {'F1-Score':<10} {'Time (s)':<10} {'Description':<50}")
    print("-"*120)
    
    for model_info in comparison['models']:
        description = model_info.get('description', 'Unknown experiment')
        # Truncate description if too long
        if len(description) > 47:
            description = description[:44] + "..."
        
        print(f"{model_info['name']:<20} {model_info['target_type']:<10} "
              f"{model_info['accuracy']:<10.4f} {model_info['f1_weighted']:<10.4f} "
              f"{model_info['evaluation_time']:<10.2f} {description:<50}")
    
    print(f"\nBest Model: {comparison['best_model']} (Accuracy: {comparison['best_accuracy']:.4f})")
    print("="*120)
