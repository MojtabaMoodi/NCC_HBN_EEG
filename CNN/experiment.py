"""
Experiment Class for EEG Classification Framework
Represents a single experiment with its complete lifecycle.
"""

import time
import os
import sys
import gc
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import tensor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from CNN.models import ModelFactory
from CNN.trainer import EEGTrainer
from CNN.evaluator import EEGEvaluator, EvaluationResults
from CNN.config import ExperimentConfig, TrainingConfig, DataConfig, ModelConfig, SystemConfig
from CNN.utils import safe_json_dump, convert_numpy_types

from data_processing.eeg_dataset import EEGDataset, EEGDataLoader, compute_class_weights_from_train_hdf5
from data_processing.target_transforms import (
    gender_classification_transform, 
    age_classification_transform, 
    combined_gender_age_classification_transform,
    create_age_regression_transform_from_hdf5,
    AgeRegressionTransform,
    create_user_identification_transform_from_hdf5
)
from data_processing.multi_file_dataset import MultiFileEEGDataset


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
    metrics: Optional[Dict[str, Any]] = None  # Participant-level (majority vote) when aggregation used
    segment_metrics: Optional[Dict[str, Any]] = None  # Segment-level (per-window accuracy)
    participant_mean_prob_metrics: Optional[Dict[str, Any]] = None  # Participant-level (mean prob then argmax) for fair comparison
    training_metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    model_checkpoint_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        d = {
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
        if self.segment_metrics is not None:
            d['segment_metrics'] = self.segment_metrics
        if self.participant_mean_prob_metrics is not None:
            d['participant_mean_prob_metrics'] = self.participant_mean_prob_metrics
        return d

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
                              lr: float = None, epoch_time: float = None,
                              is_regression: bool = False,
                              age_min: Optional[float] = None,
                              age_max: Optional[float] = None):
        """Log training progress. For regression with age_min/age_max, train_acc/val_acc are normalized MAE; display MAE in years."""
        # Store metrics (key remains 'train_acc' for compatibility; value is accuracy or normalized MAE)
        self.training_metrics['train_loss'].append(train_loss)
        self.training_metrics['val_loss'].append(val_loss)
        self.training_metrics['train_acc'].append(train_acc)
        self.training_metrics['val_acc'].append(val_acc)
        if lr is not None:
            self.training_metrics['learning_rates'].append(lr)
        if epoch_time is not None:
            self.training_metrics['epoch_times'].append(epoch_time)

        if is_regression and age_min is not None and age_max is not None:
            age_range = age_max - age_min
            train_mae_years = train_acc * age_range
            val_mae_years = val_acc * age_range
            metric_str = f"Train MAE: {train_mae_years:.2f}y | Val MAE: {val_mae_years:.2f}y"
        else:
            metric_str = f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"

        if lr is not None and epoch_time is not None:
            print(f"  → Epoch {epoch+1:3d}/{total_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"{metric_str} | "
                  f"LR: {lr:.6f} | Time: {epoch_time:.2f}s")
        else:
            print(f"  → Epoch {epoch+1:3d}/{total_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"{metric_str}")
    
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
    
    def log_evaluation_start(self, model_name: str, target_type: str, task_label: str = "classification"):
        """Log the start of evaluation. task_label is 'classification' or 'regression'."""
        print(f"Evaluating {model_name} on {target_type} {task_label}...")
    
    def log_evaluation_success(self, results):
        """Log successful evaluation completion."""
        print(f"✅ Evaluation completed")
        # Check if this is a regression task (has 'mae' instead of 'accuracy')
        if 'mae' in results.metrics:
            print(f"   MAE: {results.metrics['mae']:.4f} years")
            print(f"   RMSE: {results.metrics['rmse']:.4f} years")
            print(f"   R²: {results.metrics['r2']:.4f}")
        else:
            acc = results.metrics.get('accuracy')
            f1 = results.metrics.get('f1_weighted')
            acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else "N/A"
            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else "N/A"
            print(f"   Accuracy: {acc_str}")
            print(f"   F1-Score: {f1_str}")
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
        
        # For multi-output models, save gender and age predictions separately
        if results.target_type == 'multi_output' and results.gender_predictions is not None:
            predictions_data = convert_numpy_types({
                'gender': {
                    'predictions': results.gender_predictions.tolist() if isinstance(results.gender_predictions, np.ndarray) else results.gender_predictions,
                    'true_labels': results.gender_true_labels.tolist() if isinstance(results.gender_true_labels, np.ndarray) else results.gender_true_labels,
                    'probabilities': results.gender_probabilities.tolist() if isinstance(results.gender_probabilities, np.ndarray) else results.gender_probabilities,
                    'class_names': ['Female', 'Male']
                },
                'age': {
                    'predictions': results.age_predictions.tolist() if isinstance(results.age_predictions, np.ndarray) else results.age_predictions,
                    'true_labels': results.age_true_labels.tolist() if isinstance(results.age_true_labels, np.ndarray) else results.age_true_labels,
                    'probabilities': results.age_probabilities.tolist() if isinstance(results.age_probabilities, np.ndarray) else results.age_probabilities,
                    'class_names': ['<8.5 years', '8.5-12.5 years', '>12.5 years']
                }
            })
        else:
            # Single-output model
            predictions_data = convert_numpy_types({
                'predictions': results.predictions,
                'true_labels': results.true_labels,
                'probabilities': results.probabilities,
                'class_names': results.class_names
            })
        safe_json_dump(predictions_data, predictions_file)
        
        # Save confusion matrix plot (only for classification tasks)
        # Regression tasks don't have confusion matrices
        if 'confusion_matrix' in results.metrics:
            self._plot_confusion_matrix(results, eval_dir)
        
        print(f"   Evaluation results saved to: {eval_dir}")
    
    def _plot_confusion_matrix(self, results, save_dir: str):
        """Plot and save confusion matrix."""
        # This method is only called when 'confusion_matrix' exists in results.metrics
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
        if result.segment_metrics is not None and result.metrics:
            # Side-by-side: segment-level vs participant-level (mean prob) vs participant-level (majority vote)
            def _fmt_acc(m):
                return f"{m.get('accuracy', 0):.4f}" if isinstance(m.get('accuracy'), (int, float)) else "N/A"
            def _fmt_mae(m):
                return f"{m.get('mae', 0):.4f} years" if isinstance(m.get('mae'), (int, float)) else "N/A"
            seg = result.segment_metrics
            mean_prob = getattr(result, 'participant_mean_prob_metrics', None) or {}
            part = result.metrics
            if 'mae' in part or (seg and 'mae' in seg):
                rmse_s = f"{seg.get('rmse', 0):.4f}" if isinstance(seg.get('rmse'), (int, float)) else "N/A"
                r2_s = f"{seg.get('r2', 0):.4f}" if isinstance(seg.get('r2'), (int, float)) else "N/A"
                rmse_p = f"{part.get('rmse', 0):.4f}" if isinstance(part.get('rmse'), (int, float)) else "N/A"
                r2_p = f"{part.get('r2', 0):.4f}" if isinstance(part.get('r2'), (int, float)) else "N/A"
                print(f"   Segment-level (per-window):  MAE: {_fmt_mae(seg)}, RMSE: {rmse_s}, R²: {r2_s}")
                print(f"   Participant (mean prob):     MAE: {_fmt_mae(mean_prob)}, RMSE: ..., R²: ...")
                print(f"   Participant (majority vote): MAE: {_fmt_mae(part)}, RMSE: {rmse_p}, R²: {r2_p}")
            elif 'gender_head' in part and 'age_head' in part:
                g_seg = seg.get('gender_head', {})
                a_seg = seg.get('age_head', {})
                g_mp = mean_prob.get('gender_head', {})
                a_mp = mean_prob.get('age_head', {})
                g_part = part.get('gender_head', part)
                a_part = part.get('age_head', part)
                print(f"   Segment-level (per-window):  Gender: {_fmt_acc(g_seg)}, Age: {_fmt_acc(a_seg)}, Avg: {_fmt_acc(seg)}")
                print(f"   Participant (mean prob):     Gender: {_fmt_acc(g_mp)}, Age: {_fmt_acc(a_mp)}, Avg: {_fmt_acc(mean_prob)}")
                print(f"   Participant (majority vote): Gender: {_fmt_acc(g_part)}, Age: {_fmt_acc(a_part)}, Avg: {_fmt_acc(part)}")
            else:
                print(f"   Segment-level (per-window):  Accuracy: {_fmt_acc(seg)}")
                print(f"   Participant (mean prob):     Accuracy: {_fmt_acc(mean_prob)}")
                print(f"   Participant (majority vote): Accuracy: {_fmt_acc(part)}")
        elif result.metrics:
            if 'mae' in result.metrics:
                mae = result.metrics.get('mae')
                rmse = result.metrics.get('rmse')
                r2 = result.metrics.get('r2')
                mae_str = f"{mae:.4f} years" if isinstance(mae, (int, float)) else "N/A"
                rmse_str = f"{rmse:.4f} years" if isinstance(rmse, (int, float)) else "N/A"
                r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else "N/A"
                print(f"   MAE: {mae_str}")
                print(f"   RMSE: {rmse_str}")
                print(f"   R²: {r2_str}")
            else:
                acc = result.metrics.get('accuracy')
                acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else "N/A"
                print(f"   Accuracy: {acc_str}")
    
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
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.is_cross_validation = False

    def cleanup(self):
        """
        Release data loaders and trigger worker shutdown so the next experiment
        does not see 'terminate called without an active exception' or worker
        killed (e.g. from leftover multiprocessing workers).
        """
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.data_loaders = None
        gc.collect()
    
    def setup(self):
        """
        Setup the experiment components (data loaders, model, trainer, evaluator).
        """
        print(f"Setting up experiment: {self.config.name}")
        
        # Create data loaders using the DataConfig
        self._create_data_loaders()
        
        # For classification (gender/age), handle imbalanced data: set num_classes and optionally class weights.
        data_config = self.config.data_config
        training_config = self.config.training_config
        prediction_type = getattr(training_config, 'prediction_type', 'classification')
        balance_method = getattr(training_config, 'balance_method', None)
        target_type = self.config.target_type
        if target_type in ('gender', 'age') and prediction_type == 'classification':
            num_classes = 2 if target_type == 'gender' else 3
            self.config.model_config.num_classes = num_classes
            # When not using oversample and no explicit class_weight, compute from train data (inverse frequency).
            if (
                balance_method != 'oversample'
                and getattr(training_config, 'class_weight', None) is None
            ):
                segment_length_str = f"{data_config.segment_length // 200}s"
                weight_power = getattr(training_config, 'class_weight_power', None)
                weight_power = weight_power if weight_power is not None else 1.0
                weights = compute_class_weights_from_train_hdf5(
                    data_config.hdf5_dir,
                    segment_length_str,
                    target_type,
                    task_type=data_config.task_type,
                    weight_power=weight_power,
                )
                training_config.class_weight = weights
                self.config.model_config.num_classes = len(weights)
        
        # Create model using ModelFactory
        model_kwargs = self.config.model_config.to_dict()
        
        # Handle LaBraM checkpoint if specified
        if self.config.model_type == 'user_identification_labram':
            if hasattr(self.config.model_config, 'labram_checkpoint'):
                model_kwargs['pretrained_path'] = self.config.model_config.labram_checkpoint
        
        self.model = ModelFactory.create_model(
            self.config.model_type,
            **model_kwargs
        )
        
        # Create trainer
        # Use num_gpus from system_config, default to 1 if not specified
        num_gpus = self.system_config.num_gpus if self.system_config else 1
        device_preference = self.system_config.device if self.system_config else 'auto'
        
        # Pass age range to trainer for displaying MAE in years (for regression tasks)
        age_min = getattr(self, '_age_min', None)
        age_max = getattr(self, '_age_max', None)
        
        self.trainer = EEGTrainer(
            self.model, 
            self.config.training_config, 
            self.config.name, 
            num_gpus=num_gpus,
            age_min=age_min, 
            age_max=age_max,
            device_preference=device_preference
        )
        
        # Create evaluator
        # Determine prediction_type from training config
        prediction_type = getattr(self.config.training_config, 'prediction_type', 'classification')
        
        # Pass age min/max to evaluator if available (for regression)
        age_min = getattr(self, '_age_min', None)
        age_max = getattr(self, '_age_max', None)
        
        # Get device preference from system config
        device_preference = self.system_config.device if self.system_config else 'auto'
        
        aggregate_by_participant = getattr(
            self.config.training_config, 'aggregate_by_participant', None
        )
        self.evaluator = EEGEvaluator(
            self.model,
            self.config.target_type,
            prediction_type=prediction_type,
            age_min=age_min,
            age_max=age_max,
            device_preference=device_preference,
            aggregate_by_participant=aggregate_by_participant,
        )
        
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
        user_identification_transform = None
        
        if target_type == 'gender':
            gender_transform = gender_classification_transform
        elif target_type == 'age':
            # Check if this is a regression task
            prediction_type = getattr(self.config.training_config, 'prediction_type', 'classification')
            if prediction_type == 'regression':
                # Get train file path(s) to compute age range
                # IMPORTANT: Only use training data to compute min/max to avoid data leakage
                # The same min/max will be used for normalizing train/val/test data
                hdf5_dir_path = Path(data_config.hdf5_dir)
                segment_length_str = f"{data_config.segment_length // 200}s"
                train_files, _, _ = EEGDataLoader._get_hdf5_file_paths(hdf5_dir_path, segment_length_str)
                
                # For 1s segments, train_files is a list - pass ALL files to compute age range
                # across the entire training dataset. For 2s/4s, it's a single file.
                train_file_or_files = train_files if isinstance(train_files, (list, tuple)) else [train_files]
                
                # Create transform with dataset-specific min/max from ALL TRAINING data
                # This ensures we get the correct age range across all files for 1s segments
                age_transform, age_min, age_max = create_age_regression_transform_from_hdf5(
                    [str(f) for f in train_file_or_files]
                )
                
                # Store age range for evaluator to use (same values for train/val/test)
                self._age_min = age_min
                self._age_max = age_max
                num_files = len(train_file_or_files)
                file_str = f"{num_files} file(s)" if num_files > 1 else "file"
                print(f"📊 Using training data age range for regression normalization: min={age_min:.2f}, max={age_max:.2f} years")
                print(f"   (Computed from {file_str}, used for all splits to avoid data leakage)")
            else:
                age_transform = age_classification_transform
        elif target_type == 'combined':
            combined_transform = combined_gender_age_classification_transform
        elif target_type == 'multi_output':
            # For multi-output, we need both individual transforms
            gender_transform = gender_classification_transform
            age_transform = age_classification_transform
            combined_transform = None
        elif target_type == 'user_identification':
            # Create user identification transform from HDF5 directory
            # This will load the participant_id_to_class_idx.json mapping
            hdf5_dir_path = Path(data_config.hdf5_dir)
            user_identification_transform, num_classes = create_user_identification_transform_from_hdf5(str(hdf5_dir_path))
            
            # Update model config with correct number of classes
            self.config.model_config.num_classes = num_classes
            print(f"📊 User identification: {num_classes} user classes detected")
            print(f"   Model will be configured with num_classes={num_classes}")
        
        # Handle different experiment types based on data config
        # Note: Cross-validation and cross-task experiments are commented out
        # Only standard train/val/test split with task_type="both" is used
        if data_config.use_cross_validation and data_config.train_task_type and data_config.val_test_task_type:
            # # Cross-task cross-validation experiments
            # print(f"Creating cross-task cross-validation loaders: {data_config.train_task_type} -> {data_config.val_test_task_type}")
            # self.data_loaders = EEGDataLoader.create_cross_task_cross_validation_loaders(
            #     pickle_dir=data_config.pickle_dir,
            #     train_task_type=data_config.train_task_type,
            #     val_test_task_type=data_config.val_test_task_type,
            #     n_folds=data_config.n_folds,
            #     batch_size=data_config.batch_size,
            #     num_workers=data_config.num_workers,
            #     random_seed=data_config.random_seed,
            #     stratify_by=data_config.cv_strategy,
            #     target_type=target_type,
            #     gender_transform=gender_transform,
            #     age_transform=age_transform,
            #     combined_transform=combined_transform
            # )
            # self.is_cross_validation = True
            
            # Cross-task cross-validation experiments (commented out)
            raise NotImplementedError("Cross-task cross-validation experiments are not currently supported")
            
        elif data_config.use_cross_validation:
            # # Standard cross-validation experiments
            # print(f"Creating cross-validation loaders for {target_type} classification")
            # self.data_loaders = EEGDataLoader.create_cross_validation_loaders(
            #     pickle_dir=data_config.pickle_dir,
            #     n_folds=data_config.n_folds,
            #     batch_size=data_config.batch_size,
            #     num_workers=data_config.num_workers,
            #     random_seed=data_config.random_seed,
            #     stratify_by=data_config.cv_strategy,
            #     task_type=data_config.task_type,
            #     target_type=target_type,
            #     gender_transform=gender_transform,
            #     age_transform=age_transform,
            #     combined_transform=combined_transform
            # )
            # self.is_cross_validation = True

            # Standard cross-validation experiments (commented out)
            raise NotImplementedError("Cross-validation experiments are not currently supported")
            
        elif data_config.train_task_type and data_config.val_test_task_type:
            # # Cross-task experiments
            # print(f"Creating cross-task loaders: {data_config.train_task_type} -> {data_config.val_test_task_type}")
            # self.data_loaders = EEGDataLoader.create_cross_task_loaders(
            #     pickle_dir=data_config.pickle_dir,
            #     train_task_type=data_config.train_task_type,
            #     val_test_task_type=data_config.val_test_task_type,
            #     batch_size=data_config.batch_size,
            #     num_workers=data_config.num_workers,
            #     random_seed=data_config.random_seed,
            #     target_type=target_type,
            #     gender_transform=gender_transform,
            #     age_transform=age_transform,
            #     combined_transform=combined_transform
            # )
            # self.is_cross_validation = False

            # Cross-task experiments (commented out)
            raise NotImplementedError("Cross-task experiments are not currently supported")
            
        else:
            # Standard train/val/test split experiments
            # Train on both task types (task_type="both")
            prediction_type = getattr(self.config.training_config, "prediction_type", "classification")
            task_label = "classification" if prediction_type == "classification" else "regression"
            print(
                f"Creating standard loaders for {target_type} {task_label} (train on both task types)"
            )
            segment_length_str = f"{data_config.segment_length // 200}s"  # Convert 200->1s, 800->4s
            self.data_loaders = EEGDataLoader.create_train_val_test_loaders(
                hdf5_dir=data_config.hdf5_dir,
                segment_length=segment_length_str,
                batch_size=data_config.batch_size,
                num_workers=data_config.num_workers,
                task_type=data_config.task_type,
                target_type=target_type,
                transform=None,
                gender_transform=gender_transform,
                age_transform=age_transform,
                combined_transform=combined_transform,
                user_identification_transform=user_identification_transform,
                shuffle_train=True,
                random_seed=data_config.random_seed,
                use_stratified_train_batches=(target_type in ("gender", "age")),
                train_balance_method=getattr(self.config.training_config, "balance_method", None),
                prediction_type=getattr(self.config.training_config, "prediction_type", "classification"),
            )
            self.is_cross_validation = False
            # Store segment_length for later use in task-type-specific evaluation
            self.segment_length_str = segment_length_str

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
            self.cleanup()  # Release loaders immediately so workers can shut down (avoids DataLoader worker killed)
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

    def run_eval_only(self) -> ExperimentResult:
        """
        Load the saved checkpoint and run evaluation only (no training).
        Use this to re-evaluate with different settings (e.g. majority vote)
        without retraining. Checkpoint path is taken from config name and results_dir.
        """
        if self.model is None or self.trainer is None or self.evaluator is None:
            self.setup()
        self.logger.log_experiment_start(self.config)
        self.train_loader, self.val_loader, self.test_loader = self.data_loaders
        evaluation_start = time.time()
        evaluation_results = self._evaluate()
        evaluation_time = time.time() - evaluation_start
        self.result = ExperimentResult(
            experiment_name=self.config.name,
            model_type=self.config.model_type,
            target_type=self.config.target_type,
            success=True,
            total_time=evaluation_time,
            training_time=0.0,
            evaluation_time=evaluation_time,
            description=self.config.description,
            metrics=evaluation_results.metrics,
            segment_metrics=getattr(evaluation_results, 'segment_metrics', None),
            participant_mean_prob_metrics=getattr(evaluation_results, 'participant_mean_prob_metrics', None),
            model_checkpoint_path=self._get_checkpoint_path(),
        )
        if hasattr(evaluation_results, 'task_type_metrics'):
            self.result.metrics['task_type_metrics'] = evaluation_results.task_type_metrics
        self.logger.log_experiment_success(self.result)
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
            segment_metrics=getattr(evaluation_results, 'segment_metrics', None),
            participant_mean_prob_metrics=getattr(evaluation_results, 'participant_mean_prob_metrics', None),
            model_checkpoint_path=self._get_checkpoint_path()
        )

        # Add training metrics to result
        result.training_metrics = self.logger.get_training_metrics()
        
        # Add task-type-specific metrics if available
        if hasattr(evaluation_results, 'task_type_metrics'):
            result.metrics['task_type_metrics'] = evaluation_results.task_type_metrics
        
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
        prediction_type = getattr(self.config.training_config, 'prediction_type', 'classification')
        task_label = "regression" if prediction_type == "regression" else "classification"
        print(f"Training {self.config.model_type} for {self.config.target_type} {task_label}...")

        is_regression = prediction_type == "regression"
        age_min = getattr(self, '_age_min', None)
        age_max = getattr(self, '_age_max', None)

        def progress_callback(epoch, total_epochs, train_loss, val_loss, train_acc, val_acc, lr, epoch_time):
            self.logger.log_experiment_progress(
                epoch, total_epochs, train_loss, val_loss, train_acc, val_acc, lr, epoch_time,
                is_regression=is_regression, age_min=age_min, age_max=age_max,
            )
        
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
        """Evaluate the model on overall test set and separately by task type."""
        prediction_type = getattr(self.config.training_config, 'prediction_type', 'classification')
        task_label = "regression" if prediction_type == "regression" else "classification"
        self.logger.log_evaluation_start(self.config.model_type, self.config.target_type, task_label=task_label)
        
        # Load best model
        checkpoint_path = self._get_checkpoint_path()
        if os.path.exists(checkpoint_path):
            self.evaluator.load_checkpoint(checkpoint_path)
        
        # Evaluate on overall test set
        results = self.evaluator.evaluate(self.test_loader)
        
        # Log and save overall results
        self.logger.log_evaluation_success(results)
        self.logger.save_evaluation_results(results, self.config.name)
        
        # Evaluate separately by task type (active and passive)
        print(f"\nEvaluating separately by task type...")
        task_type_results = self._evaluate_by_task_type()
        
        # Store task-type-specific metrics in results
        results.task_type_metrics = {
            task_type: task_results.metrics 
            for task_type, task_results in task_type_results.items()
        }
        
        # Log task-type-specific results
        for task_type, task_results in task_type_results.items():
            # Display appropriate metrics based on prediction type
            if hasattr(self.config.training_config, 'prediction_type') and self.config.training_config.prediction_type == 'regression':
                print(f"  {task_type.upper()} tasks: MAE = {task_results.metrics['mae']:.4f} years, "
                      f"RMSE = {task_results.metrics['rmse']:.4f} years, "
                      f"R² = {task_results.metrics['r2']:.4f}")
            else:
                print(f"  {task_type.upper()} tasks: Accuracy = {task_results.metrics['accuracy']:.4f}, "
                      f"F1-Score = {task_results.metrics['f1_weighted']:.4f}")
        
        return results
    
    def _evaluate_by_task_type(self) -> Dict[str, EvaluationResults]:
        """Evaluate model separately on active and passive tasks."""
        
        data_config = self.config.data_config
        target_type = self.config.target_type
        
        # Select appropriate transforms based on target type
        gender_transform = None
        age_transform = None
        combined_transform = None
        user_identification_transform = None
        
        if target_type == 'gender':
            gender_transform = gender_classification_transform
        elif target_type == 'age':
            # Check if this is a regression task
            prediction_type = getattr(self.config.training_config, 'prediction_type', 'classification')
            if prediction_type == 'regression':
                # For task-type-specific evaluation, use the same age range computed during setup
                # This should always be set in _create_data_loaders() for regression tasks
                if not hasattr(self, '_age_min') or not hasattr(self, '_age_max'):
                    raise ValueError(
                        "age_min and age_max must be set for age regression tasks. "
                        "This should happen automatically in _create_data_loaders(). "
                        "If you see this error, there's a bug in the experiment setup."
                    )
                # Use the age range computed from TRAINING data during data loader creation
                # Create a picklable transform class instance for multiprocessing compatibility
                age_transform = AgeRegressionTransform(self._age_min, self._age_max)
            else:
                age_transform = age_classification_transform
        elif target_type == 'combined':
            combined_transform = combined_gender_age_classification_transform
        elif target_type == 'multi_output':
            gender_transform = gender_classification_transform
            age_transform = age_classification_transform
        elif target_type == 'user_identification':
            # Load user identification transform for task-type-specific evaluation
            hdf5_dir_path = Path(data_config.hdf5_dir)
            user_identification_transform, _ = create_user_identification_transform_from_hdf5(str(hdf5_dir_path))
        
        # Get test file path(s) - may be single file or list of files for 1s segments
        hdf5_dir_path = Path(data_config.hdf5_dir)
        train_files, val_files, test_files = EEGDataLoader._get_hdf5_file_paths(
            hdf5_dir_path, self.segment_length_str
        )
        
        # Create datasets for each task type
        # Handle both single file and multiple files (for 1s segments)
        if isinstance(test_files, (list, tuple)):
            # Multiple files - use MultiFileEEGDataset
            test_file_paths = [str(f) for f in test_files]
            
            active_dataset = MultiFileEEGDataset(
                hdf5_files=test_file_paths,
                task_type="active",
                target_type=target_type,
                gender_transform=gender_transform,
                age_transform=age_transform,
                combined_transform=combined_transform,
                user_identification_transform=user_identification_transform,
                shuffle=False
            )
            passive_dataset = MultiFileEEGDataset(
                hdf5_files=test_file_paths,
                task_type="passive",
                target_type=target_type,
                gender_transform=gender_transform,
                age_transform=age_transform,
                combined_transform=combined_transform,
                user_identification_transform=user_identification_transform,
                shuffle=False
            )
        else:
            # Single file
            active_dataset = EEGDataset(
                hdf5_file=str(test_files),
                task_type="active",
                target_type=target_type,
                gender_transform=gender_transform,
                age_transform=age_transform,
                combined_transform=combined_transform,
                user_identification_transform=user_identification_transform,
                shuffle=False
            )
            passive_dataset = EEGDataset(
                hdf5_file=str(test_files),
                task_type="passive",
                target_type=target_type,
                gender_transform=gender_transform,
                age_transform=age_transform,
                combined_transform=combined_transform,
                user_identification_transform=user_identification_transform,
                shuffle=False
            )
        
        # Create loaders
        active_loader = EEGDataLoader.create_dataloader(
            active_dataset, 
            batch_size=data_config.batch_size,
            num_workers=data_config.num_workers
        )
        passive_loader = EEGDataLoader.create_dataloader(
            passive_dataset,
            batch_size=data_config.batch_size,
            num_workers=data_config.num_workers
        )
        
        # Evaluate separately
        test_loaders_by_task = {'active': active_loader, 'passive': passive_loader}
        results_by_task = self.evaluator.evaluate_by_task_type(test_loaders_by_task)
        
        # Save task-type-specific results
        for task_type, task_results in results_by_task.items():
            self.logger.save_evaluation_results(
                task_results, 
                f"{self.config.name}_task_{task_type}"
            )
        
        return results_by_task
    
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
                   generate_reports: bool = True,
                   eval_only: bool = False) -> List[ExperimentResult]:
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
        eval_only: If True, skip training and only run evaluation (load saved checkpoint).

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
    if eval_only:
        print(f"Mode: EVAL ONLY (load checkpoint, no training)")
    print(f"{'='*80}")

    results = []

    for i, config in enumerate(experiments):
        print(f"\nProgress: {i+1}/{len(experiments)}")
        print(f"Experiment: {config.name}")

        experiment = None
        try:
            # Create and run experiment (data loaders created internally)
            experiment = Experiment(config, system_config)
            experiment.setup()  # Setup data loaders, model, trainer, evaluator
            if eval_only:
                result = experiment.run_eval_only()
            else:
                result = experiment.run()
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
        finally:
            # Release data loaders so workers shut down before the next experiment.
            # Avoids "terminate called without an active exception" / worker killed.
            if experiment is not None:
                experiment.cleanup()
    
    # Save summary of all results
    _save_experiment_summary(results, system_config.results_dir)
    
    # Generate reports if requested
    if generate_reports:
        from CNN.report_generator import ReportGenerator
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
    from CNN.evaluator import compare_models
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
    # Determine if we're comparing regression or classification models
    is_regression = any('mae' in model_info for model_info in comparison['models'])
    
    print("\n" + "="*120)
    print("MODEL COMPARISON")
    print("="*120)
    
    if is_regression:
        print(f"{'Model':<20} {'Target':<10} {'MAE (years)':<15} {'RMSE (years)':<15} {'R²':<10} {'Time (s)':<10} {'Description':<50}")
        print("-"*120)
        
        for model_info in comparison['models']:
            description = model_info.get('description', 'Unknown experiment')
            # Truncate description if too long
            if len(description) > 47:
                description = description[:44] + "..."
            
            mae = model_info.get('mae', None)
            rmse = model_info.get('rmse', None)
            r2 = model_info.get('r2', None)
            
            mae_str = f"{mae:.4f}" if mae is not None else "N/A"
            rmse_str = f"{rmse:.4f}" if rmse is not None else "N/A"
            r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
            print(f"{model_info['name']:<20} {model_info['target_type']:<10} "
                  f"{mae_str:<15} {rmse_str:<15} {r2_str:<10} "
                  f"{model_info['evaluation_time']:<10.2f} {description:<50}")
        
        best_metric = comparison.get('best_mae', None)
        metric_name = 'MAE'
    else:
        print(f"{'Model':<20} {'Target':<10} {'Accuracy':<10} {'F1-Score':<10} {'Time (s)':<10} {'Description':<50}")
        print("-"*120)
        
        for model_info in comparison['models']:
            description = model_info.get('description', 'Unknown experiment')
            # Truncate description if too long
            if len(description) > 47:
                description = description[:44] + "..."
            
            acc = model_info.get('accuracy', None)
            f1 = model_info.get('f1_weighted', None)
            acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else "N/A"
            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else "N/A"
            print(f"{model_info['name']:<20} {model_info['target_type']:<10} "
                  f"{acc_str:<10} {f1_str:<10} "
                  f"{model_info['evaluation_time']:<10.2f} {description:<50}")
        
        best_metric = comparison.get('best_accuracy', None)
        metric_name = 'Accuracy'
    
    best_metric_str = f"{best_metric:.4f}" if best_metric is not None else "N/A"
    print(f"\nBest Model: {comparison['best_model']} ({metric_name}: {best_metric_str})")
    print("="*120)
