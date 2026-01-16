"""
Comprehensive Evaluation Module for EEG Classification Models
Provides systematic evaluation with detailed metrics and reporting.
"""

import os
import json
import time
from typing import Dict, Any, List, Tuple, Optional
import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

from CNN.models import BaseEEGCNN
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/data_processing')
from constants import DEFAULT_MIN_AGE, DEFAULT_MAX_AGE
from CNN.utils import safe_json_dump, convert_numpy_types
from gpu_utils import get_underlying_model as gpu_get_underlying_model, get_device

class EvaluationResults:
    """Container for evaluation results with comprehensive metrics."""
    
    def __init__(self, model_name: str, target_type: str):
        self.model_name = model_name
        self.target_type = target_type
        self.metrics = {}
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
        self.class_names = []
        self.evaluation_time = 0.0
        
        # Multi-output specific attributes
        self.gender_predictions = None
        self.gender_true_labels = None
        self.gender_probabilities = None
        self.age_predictions = None
        self.age_true_labels = None
        self.age_probabilities = None
        # Separate metrics for each head
        self.gender_metrics = None
        self.age_metrics = None
    
    def add_metrics(self, metrics: Dict[str, Any]):
        """Add computed metrics."""
        self.metrics.update(metrics)
    
    def add_predictions(self, predictions: List[int], true_labels: List[int], 
                       probabilities: Optional[List[List[float]]] = None):
        """Add prediction results."""
        self.predictions.extend(predictions)
        self.true_labels.extend(true_labels)
        if probabilities:
            self.probabilities.extend(probabilities)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        # For multi-output models, use gender_predictions length (both heads have same number of samples)
        num_samples = len(self.predictions) if len(self.predictions) > 0 else (
            len(self.gender_predictions) if self.gender_predictions is not None else 0
        )
        return convert_numpy_types({
            'model_name': self.model_name,
            'target_type': self.target_type,
            'metrics': self.metrics,
            'evaluation_time': self.evaluation_time,
            'num_samples': num_samples,
            'class_names': self.class_names
        })

class EEGEvaluator:
    """
    Comprehensive evaluator for EEG classification models.
    Provides detailed metrics, confusion matrices, and classification reports.
    """
    
    def __init__(self, model: BaseEEGCNN, target_type: str, 
                 class_names: Optional[List[str]] = None, prediction_type: str = 'classification',
                 age_min: Optional[float] = None, age_max: Optional[float] = None,
                 device_preference: str = 'auto'):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate (may be wrapped with DataParallel)
            target_type: Type of target ('gender', 'age', 'combined', 'multi_output')
            class_names: Optional list of class names
            prediction_type: 'classification' or 'regression'
            age_min: Minimum age for regression tasks (required for age regression)
            age_max: Maximum age for regression tasks (required for age regression)
            device_preference: Device preference ('auto', 'cuda', 'cpu')
        
        Raises:
            ValueError: If age_min/age_max are missing for regression tasks
            RuntimeError: If CUDA is requested but not available
        """
        self.model = model
        self.target_type = target_type
        self.prediction_type = prediction_type  # 'classification' or 'regression'
        self.class_names = class_names or self._get_default_class_names()
        
        # Setup device (explicit, no fallbacks)
        self.device = get_device(device_preference)
        
        # Move model to device (handles DataParallel models correctly)
        self.model.to(self.device)
        self.model.eval()
        
        # Age normalization parameters for denormalization (required for regression)
        # These should be computed from TRAINING data only and passed during initialization
        # to avoid data leakage
        if prediction_type == 'regression' and target_type == 'age':
            if age_min is None or age_max is None:
                raise ValueError(
                    "age_min and age_max must be provided for age regression tasks. "
                    "These should be computed from TRAINING data only to avoid data leakage. "
                    "Use create_age_regression_transform_from_hdf5() on the training HDF5 file."
                )
        self.age_min = age_min
        self.age_max = age_max
    
    def _get_underlying_model(self):
        """Get the underlying model, unwrapping DataParallel and torch.compile wrappers if needed."""
        return gpu_get_underlying_model(self.model)
    
    def _get_default_class_names(self) -> List[str]:
        """Get default class names based on target type."""
        if self.target_type == 'gender':
            return ['Female', 'Male']
        elif self.target_type == 'age':
            return ['<8.5 years', '8.5-12.5 years', '>12.5 years']
        elif self.target_type == 'user_identification':
            # For user identification, class names are participant IDs
            # We'll use generic class names since we don't have access to the mapping here
            # The actual participant IDs can be retrieved from the mapping file if needed
            underlying_model = self._get_underlying_model()
            return [f'User {i}' for i in range(underlying_model.num_classes)]
        else:
            underlying_model = self._get_underlying_model()
            return [f'Class {i}' for i in range(underlying_model.num_classes)]
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model weights from checkpoint using the existing load_checkpoint function.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if os.path.exists(checkpoint_path):
            # Import here to avoid circular imports
            from trainer import load_checkpoint as trainer_load_checkpoint
            
            # Use the existing load_checkpoint function from trainer.py
            checkpoint = trainer_load_checkpoint(checkpoint_path, self.model, self.device)
            self.model.eval()  # Set to evaluation mode
            print(f"✅ Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
    
    def evaluate(self, test_loader: DataLoader) -> EvaluationResults:
        """
        Evaluate model on test dataset.
        
        Args:
            test_loader: DataLoader for test dataset
            
        Returns:
            EvaluationResults object with comprehensive metrics
        """
        
        start_time = time.time()
        results = EvaluationResults(self._get_underlying_model().__class__.__name__, self.target_type)
        results.class_names = self.class_names
        
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['eeg_data'].to(self.device)
                
                # Handle multi_output case where we have separate gender and age keys
                if self.target_type == 'multi_output' and 'gender' in batch and 'age' in batch:
                    labels = {
                        'gender': batch['gender'].to(self.device),
                        'age': batch['age'].to(self.device)
                    }
                elif self.target_type == 'combined':
                    gender = batch['gender'].to(self.device)
                    age = batch['age'].to(self.device)
                    # For combined experiments, age should be classification (0, 1, 2), not regression
                    # Gender should also be classification (0, 1)
                    # Optimized: Use vectorized operations instead of Python list comprehension
                    gender_int = gender.long()  # Already 0 or 1
                    age_int = age.long()  # Already 0, 1, or 2
                    # Combined class = gender * 3 + age (vectorized)
                    combined_classes = gender_int * 3 + age_int
                    labels = combined_classes
                else:
                    # Get labels from batch - validate key exists
                    if self.target_type not in batch:
                        raise KeyError(
                            f"Target type '{self.target_type}' not found in batch during evaluation. "
                            f"Available keys: {list(batch.keys())}. "
                            f"This indicates a data loading issue."
                        )
                    labels = batch[self.target_type].to(self.device)
                    
                    # Validate labels for classification tasks
                    if self.prediction_type == 'classification':
                        # Ensure labels are long integers for classification
                        if labels.dtype != torch.long:
                            labels = labels.long()
                        
                        # Validate label range for classification tasks
                        num_classes = self._get_underlying_model().num_classes
                        if labels.max() >= num_classes or labels.min() < 0:
                            invalid_labels = labels[(labels >= num_classes) | (labels < 0)]
                            raise ValueError(
                                f"Invalid labels detected during evaluation: {invalid_labels.cpu().numpy()}. "
                                f"Labels must be in range [0, {num_classes-1}], but found values outside this range. "
                                f"This indicates a data inconsistency issue. "
                                f"Check that the transform is correctly mapping participant IDs to class indices."
                            )
                
                outputs = self.model(inputs)
                
                # Handle multi-output models
                if isinstance(outputs, dict):
                    # Multi-output model (e.g., MultiOutputCNN)
                    # Evaluate both outputs separately
                    gender_outputs = outputs['gender']
                    age_outputs = outputs['age']
                    gender_labels = labels['gender']
                    age_labels = labels['age']
                    
                    # Process gender predictions
                    gender_probabilities = torch.softmax(gender_outputs, dim=1)
                    _, gender_predicted = torch.max(gender_outputs, 1)
                    
                    # Process age predictions
                    age_probabilities = torch.softmax(age_outputs, dim=1)
                    _, age_predicted = torch.max(age_outputs, 1)
                    
                    # Store predictions for each head separately
                    if results.gender_predictions is None:
                        # Initialize lists for first batch
                        results.gender_predictions = []
                        results.gender_true_labels = []
                        results.age_predictions = []
                        results.age_true_labels = []
                        results.gender_probabilities = []
                        results.age_probabilities = []
                    
                    results.gender_predictions.extend(gender_predicted.cpu().numpy())
                    results.gender_true_labels.extend(gender_labels.cpu().numpy())
                    results.gender_probabilities.extend(gender_probabilities.cpu().numpy())
                    results.age_predictions.extend(age_predicted.cpu().numpy())
                    results.age_true_labels.extend(age_labels.cpu().numpy())
                    results.age_probabilities.extend(age_probabilities.cpu().numpy())
                    
                    # For multi-output models, we don't populate all_predictions
                    # Use gender_predictions and age_predictions separately instead
                    
                else:
                    # Single-output model (e.g., EEGCNN, CombinedCNN, EEGAgeRegressionCNN)
                    if self.prediction_type == 'regression':
                        # For regression: outputs are already in [0, 1] range
                        # Reshape if needed: (batch_size, 1) -> (batch_size,)
                        if outputs.dim() > 1 and outputs.size(1) == 1:
                            outputs = outputs.squeeze(1)
                        if labels.dim() > 1 and labels.size(1) == 1:
                            labels = labels.squeeze(1)
                        # Ensure labels are float
                        if labels.dtype != torch.float32:
                            labels = labels.float()
                        
                        # Store normalized predictions and labels
                        all_predictions.extend(outputs.cpu().numpy())
                        all_true_labels.extend(labels.cpu().numpy())
                        # For regression, we don't use probabilities
                        all_probabilities.extend([[p] for p in outputs.cpu().numpy()])  # Dummy probabilities for compatibility
                    else:
                        # For classification: use softmax and argmax
                        probabilities = torch.softmax(outputs, dim=1)
                        _, predicted = torch.max(outputs, 1)
                        
                        all_predictions.extend(predicted.cpu().numpy())
                        all_true_labels.extend(labels.cpu().numpy())
                        all_probabilities.extend(probabilities.cpu().numpy())
        
        # Add predictions to results (only for single-output models)
        # For multi-output models, predictions are stored separately in gender_predictions and age_predictions
        if len(all_predictions) > 0:
            results.add_predictions(all_predictions, all_true_labels, all_probabilities)
        
        # For multi-output models, compute metrics separately for each head
        if self.target_type == 'multi_output' and results.gender_predictions is not None:
            # Convert lists to arrays
            results.gender_predictions = np.array(results.gender_predictions)
            results.gender_true_labels = np.array(results.gender_true_labels)
            results.gender_probabilities = np.array(results.gender_probabilities)
            results.age_predictions = np.array(results.age_predictions)
            results.age_true_labels = np.array(results.age_true_labels)
            results.age_probabilities = np.array(results.age_probabilities)
            
            # Compute metrics for gender head
            gender_class_names = ['Female', 'Male']
            results.gender_metrics = self._compute_metrics(
                results.gender_predictions.tolist(),
                results.gender_true_labels.tolist(),
                results.gender_probabilities.tolist(),
                class_names=gender_class_names
            )
            
            # Compute metrics for age head
            age_class_names = ['<8.5 years', '8.5-12.5 years', '>12.5 years']
            results.age_metrics = self._compute_metrics(
                results.age_predictions.tolist(),
                results.age_true_labels.tolist(),
                results.age_probabilities.tolist(),
                class_names=age_class_names
            )
            
            # Compute average metrics to match trainer's calculation
            # Trainer computes: (gender_acc + age_acc) / 2 per sample, averaged across all samples
            gender_roc = results.gender_metrics.get('roc_auc')
            age_roc = results.age_metrics.get('roc_auc')
            avg_roc_auc = None
            if gender_roc is not None and age_roc is not None:
                avg_roc_auc = (gender_roc + age_roc) / 2.0
            elif gender_roc is not None:
                avg_roc_auc = gender_roc
            elif age_roc is not None:
                avg_roc_auc = age_roc
            
            avg_metrics = {
                'accuracy': (results.gender_metrics['accuracy'] + results.age_metrics['accuracy']) / 2.0,
                'precision_weighted': (results.gender_metrics['precision_weighted'] + results.age_metrics['precision_weighted']) / 2.0,
                'recall_weighted': (results.gender_metrics['recall_weighted'] + results.age_metrics['recall_weighted']) / 2.0,
                'f1_weighted': (results.gender_metrics['f1_weighted'] + results.age_metrics['f1_weighted']) / 2.0,
                'roc_auc': avg_roc_auc,
                # Note: The following metrics cannot be meaningfully averaged since they have different structures
                # (different numbers of classes: gender=2, age=3). These are included for backward compatibility
                # with existing code that expects them, but true per-head metrics are available in 'gender_head' and 'age_head'.
                # The primary scalar metrics (accuracy, precision, recall, f1, roc_auc) above use the average.
                'confusion_matrix': results.gender_metrics.get('confusion_matrix'),
                'classification_report': results.gender_metrics.get('classification_report'),
                'precision_per_class': results.gender_metrics.get('precision_per_class'),
                'recall_per_class': results.gender_metrics.get('recall_per_class'),
                'f1_per_class': results.gender_metrics.get('f1_per_class'),
                'support_per_class': results.gender_metrics.get('support_per_class'),
            }
            
            # Use average metrics as primary (matching trainer behavior)
            results.add_metrics(avg_metrics)
            
            # Add head-specific metrics to main metrics dict for easy access
            results.metrics['gender_head'] = results.gender_metrics
            results.metrics['age_head'] = results.age_metrics
        else:
            # Single-output model or combined model - compute metrics normally
            if self.prediction_type == 'regression':
                results.add_metrics(self._compute_regression_metrics(
                    all_predictions,
                    all_true_labels
                ))
            else:
                metrics = self._compute_metrics(all_predictions, all_true_labels, all_probabilities)
                results.add_metrics(metrics)
        
        # Set evaluation time
        results.evaluation_time = time.time() - start_time
        
        return results
    
    def evaluate_by_task_type(self, task_type_loaders: Dict[str, DataLoader]) -> Dict[str, EvaluationResults]:
        """
        Evaluate model separately on different task types (e.g., active vs passive).
        
        This method is useful when you train on all task types but want to evaluate
        performance separately for each task type to understand task-specific performance.
        
        Example usage:
            # Train on all task types
            train_loader, val_loader, test_loader = EEGDataLoader.create_train_val_test_loaders(
                hdf5_dir="processed_eeg_data_hdf5",
                task_type="both"  # Train on both active and passive
            )
            
            # ... train model ...
            
            # Evaluate separately on active and passive tasks
            from eeg_dataset import EEGDataset, EEGDataLoader
            from pathlib import Path
            
            # Get the test file path(s) - may be single file or list for 1s segments
            hdf5_dir_path = Path("processed_eeg_data_hdf5")
            train_files, val_files, test_files = EEGDataLoader._get_hdf5_file_paths(
                hdf5_dir_path, segment_length="1s"
            )
            
            # Create datasets for each task type
            # Use _create_dataset_from_hdf5 which handles both single files and lists
            active_dataset = EEGDataLoader._create_dataset_from_hdf5(
                hdf5_file_or_files=test_files,
                task_type="active",
                target_type="gender",  # Example target type
                transform=None,
                gender_transform=None,
                age_transform=None,
                combined_transform=None,
                user_identification_transform=None,
                shuffle=False
            )
            passive_dataset = EEGDataLoader._create_dataset_from_hdf5(
                hdf5_file_or_files=test_files,
                task_type="passive",
                target_type="gender",  # Example target type
                transform=None,
                gender_transform=None,
                age_transform=None,
                combined_transform=None,
                user_identification_transform=None,
                shuffle=False
            )
            
            # Create loaders
            active_loader = EEGDataLoader.create_dataloader(active_dataset, batch_size=32)
            passive_loader = EEGDataLoader.create_dataloader(passive_dataset, batch_size=32)
            
            # Evaluate separately
            test_loaders_by_task = {'active': active_loader, 'passive': passive_loader}
            results_by_task = evaluator.evaluate_by_task_type(test_loaders_by_task)
            # Returns: {'active': EvaluationResults, 'passive': EvaluationResults}
            
            # Access results
            active_accuracy = results_by_task['active'].metrics['accuracy']
            passive_accuracy = results_by_task['passive'].metrics['accuracy']
        
        Args:
            task_type_loaders: Dictionary mapping task type names to DataLoaders
                             (e.g., {'active': active_loader, 'passive': passive_loader})
        
        Returns:
            Dictionary mapping task type names to EvaluationResults
            (e.g., {'active': EvaluationResults, 'passive': EvaluationResults})
        """
        results_by_task_type = {}
        
        for task_type, loader in task_type_loaders.items():
            print(f"\n{'='*60}")
            print(f"Evaluating on {task_type} tasks...")
            print(f"{'='*60}")
            
            # Evaluate on this task type
            results = self.evaluate(loader)
            results_by_task_type[task_type] = results
            
            # Calculate and store sample count
            num_samples = len(results.predictions) if results.predictions else (
                len(results.gender_predictions) if results.gender_predictions is not None else 0
            )
            # Store sample count in metrics for reporting
            results.metrics['num_samples'] = num_samples
            
            # Print summary
            print(f"\n{task_type.upper()} Task Results:")
            # Check if this is a regression task (has 'mae' instead of 'accuracy')
            if 'mae' in results.metrics:
                print(f"  MAE: {results.metrics['mae']:.4f} years")
                print(f"  RMSE: {results.metrics['rmse']:.4f} years")
                print(f"  R²: {results.metrics['r2']:.4f}")
            else:
                acc = results.metrics.get('accuracy')
                acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else "N/A"
                print(f"  Accuracy: {acc_str}")
                if 'f1_weighted' in results.metrics:
                    print(f"  F1-Score (weighted): {results.metrics['f1_weighted']:.4f}")
                if 'roc_auc' in results.metrics and results.metrics['roc_auc'] is not None:
                    print(f"  ROC-AUC: {results.metrics['roc_auc']:.4f}")
            print(f"  Number of samples: {num_samples}")
        
        # Print comparison summary
        if len(results_by_task_type) > 1:
            print(f"\n{'='*60}")
            print("Task Type Comparison:")
            print(f"{'='*60}")
            for task_type, results in results_by_task_type.items():
                # Check if this is a regression task (has 'mae' instead of 'accuracy')
                if 'mae' in results.metrics:
                    print(f"  {task_type.upper()}: MAE = {results.metrics['mae']:.4f} years, "
                          f"RMSE = {results.metrics['rmse']:.4f} years, "
                          f"R² = {results.metrics['r2']:.4f}")
                else:
                    acc = results.metrics.get('accuracy')
                    f1 = results.metrics.get('f1_weighted')
                    acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else "N/A"
                    f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else "N/A"
                    print(f"  {task_type.upper()}: Accuracy = {acc_str}, F1 = {f1_str}")
        
        return results_by_task_type
    
    def _compute_metrics(self, predictions: List[int], true_labels: List[int], 
                        probabilities: List[List[float]], class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics for classification."""
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        probabilities = np.array(probabilities)
        
        # Use provided class_names or fall back to default
        metric_class_names = class_names if class_names is not None else self.class_names
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Precision, recall, F1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # ROC AUC (for binary classification or multi-class)
        try:
            if len(metric_class_names) == 2:
                # Binary classification
                roc_auc = roc_auc_score(true_labels, probabilities[:, 1])
            else:
                # Multi-class classification
                roc_auc = roc_auc_score(true_labels, probabilities, multi_class='ovr', average='weighted')
        except ValueError:
            roc_auc = None
        
        # Classification report
        class_report = classification_report(
            true_labels, predictions, 
            target_names=metric_class_names, 
            output_dict=True, 
            zero_division=0
        )
        
        return {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision),
            'recall_weighted': float(recall),
            'f1_weighted': float(f1),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'support_per_class': support_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'classification_report': class_report
        }
    
    def _compute_regression_metrics(self, predictions: List[float], true_labels: List[float]) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics for regression."""
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Denormalize predictions and labels back to actual age values
        # Both are in [0, 1] range, need to convert back to [age_min, age_max]
        pred_age = predictions * (self.age_max - self.age_min) + self.age_min
        true_age = true_labels * (self.age_max - self.age_min) + self.age_min
        
        # Compute regression metrics
        mae = mean_absolute_error(true_age, pred_age)
        mse = mean_squared_error(true_age, pred_age)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_age, pred_age)
        
        # Also compute metrics on normalized values for comparison
        mae_norm = mean_absolute_error(true_labels, predictions)
        mse_norm = mean_squared_error(true_labels, predictions)
        rmse_norm = np.sqrt(mse_norm)
        
        return {
            'mae': float(mae),  # Mean Absolute Error in years
            'mse': float(mse),  # Mean Squared Error in years²
            'rmse': float(rmse),  # Root Mean Squared Error in years
            'r2': float(r2),  # R² score
            'mae_normalized': float(mae_norm),  # MAE on normalized [0, 1] values
            'mse_normalized': float(mse_norm),  # MSE on normalized [0, 1] values
            'rmse_normalized': float(rmse_norm),  # RMSE on normalized [0, 1] values
            'age_min': float(self.age_min),
            'age_max': float(self.age_max)
        }
    

def compare_models(results_list: List[EvaluationResults]) -> Dict[str, Any]:
    """
    Compare multiple model evaluation results.
    
    Args:
        results_list: List of EvaluationResults objects
        
    Returns:
        Comparison summary dictionary
    """
    comparison = {
        'models': [],
        'metrics_comparison': {},
        'best_model': None
    }
    
    # Determine if we're comparing regression or classification models
    is_regression = any('mae' in r.metrics for r in results_list)
    
    if is_regression:
        # For regression: use MAE (lower is better)
        best_metric = float('inf')
        best_model_name = None
        metric_key = 'mae'
    else:
        # For classification: use accuracy (higher is better)
        best_metric = 0
        best_model_name = None
        metric_key = 'accuracy'
    
    for results in results_list:
        if is_regression:
            model_info = {
                'name': results.model_name,
                'target_type': results.target_type,
                'mae': results.metrics.get('mae', None),
                'rmse': results.metrics.get('rmse', None),
                'r2': results.metrics.get('r2', None),
                'evaluation_time': results.evaluation_time
            }
            # For regression, lower MAE is better
            current_metric = results.metrics.get('mae', float('inf'))
            if current_metric < best_metric:
                best_metric = current_metric
                best_model_name = results.model_name
        else:
            model_info = {
                'name': results.model_name,
                'target_type': results.target_type,
                'accuracy': results.metrics.get('accuracy', None),
                'f1_weighted': results.metrics.get('f1_weighted', None),
                'evaluation_time': results.evaluation_time
            }
            # For classification, higher accuracy is better
            current_metric = results.metrics.get('accuracy', 0)
            if current_metric > best_metric:
                best_metric = current_metric
                best_model_name = results.model_name
        
        comparison['models'].append(model_info)
    
    comparison['best_model'] = best_model_name
    comparison[f'best_{metric_key}'] = best_metric
    
    return comparison
