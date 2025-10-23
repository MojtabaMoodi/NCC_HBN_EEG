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
        return convert_numpy_types({
            'model_name': self.model_name,
            'target_type': self.target_type,
            'metrics': self.metrics,
            'evaluation_time': self.evaluation_time,
            'num_samples': len(self.predictions),
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
    
    def _get_default_class_names(self) -> List[str]:
        """Get default class names based on target type."""
        if self.target_type == 'gender':
            return ['Female', 'Male']
        elif self.target_type == 'age':
            return ['<8.5 years', '8.5-12.5 years', '>12.5 years']
        else:
            return [f'Class {i}' for i in range(self.model.num_classes)]
    
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
        results = EvaluationResults(self.model.__class__.__name__, self.target_type)
        results.class_names = self.class_names
        
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['eeg_data'].to(self.device)
                labels = batch[self.target_type].to(self.device)
                
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Add predictions to results
        results.add_predictions(all_predictions, all_true_labels, all_probabilities)
        
        # Compute metrics
        metrics = self._compute_metrics(all_predictions, all_true_labels, all_probabilities)
        results.add_metrics(metrics)
        
        # Set evaluation time
        results.evaluation_time = time.time() - start_time
        
        return results
    
    def _compute_metrics(self, predictions: List[int], true_labels: List[int], 
                        probabilities: List[List[float]]) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics."""
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        probabilities = np.array(probabilities)
        
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
            if len(self.class_names) == 2:
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
            target_names=self.class_names, 
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
