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
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score
)

from models import BaseEEGCNN
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# For combined experiments, compute combined class from gender and age
sys.path.append('/home/mojtabam/projects/def-aghodsib/mojtabam/EEG/data_processing')
from constants import get_combined_class
from utils import safe_json_dump, convert_numpy_types

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
                 class_names: Optional[List[str]] = None):
        self.model = model
        self.target_type = target_type
        self.class_names = class_names or self._get_default_class_names()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def _get_underlying_model(self):
        """Get the underlying model, unwrapping DataParallel if needed."""
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module
        return self.model
    
    def _get_default_class_names(self) -> List[str]:
        """Get default class names based on target type."""
        if self.target_type == 'gender':
            return ['Female', 'Male']
        elif self.target_type == 'age':
            return ['<8.5 years', '8.5-12.5 years', '>12.5 years']
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
                    combined_classes = torch.tensor([get_combined_class(g.item(), a.item()) for g, a in zip(gender, age)], 
                                                  dtype=torch.long, device=self.device)
                    labels = combined_classes
                else:
                    labels = batch[self.target_type].to(self.device)
                
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
                    # Single-output model (e.g., EEGCNN, CombinedCNN)
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
            metrics = self._compute_metrics(all_predictions, all_true_labels, all_probabilities)
            results.add_metrics(metrics)
        
        # Set evaluation time
        results.evaluation_time = time.time() - start_time
        
        return results
    
    def _compute_metrics(self, predictions: List[int], true_labels: List[int], 
                        probabilities: List[List[float]], class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics."""
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
    
    best_accuracy = 0
    best_model_name = None
    
    for results in results_list:
        model_info = {
            'name': results.model_name,
            'target_type': results.target_type,
            'accuracy': results.metrics['accuracy'],
            'f1_weighted': results.metrics['f1_weighted'],
            'evaluation_time': results.evaluation_time
        }
        comparison['models'].append(model_info)
        
        if results.metrics['accuracy'] > best_accuracy:
            best_accuracy = results.metrics['accuracy']
            best_model_name = results.model_name
    
    comparison['best_model'] = best_model_name
    comparison['best_accuracy'] = best_accuracy
    
    return comparison
