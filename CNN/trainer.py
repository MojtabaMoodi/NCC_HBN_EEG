"""
Systematic Training Module for EEG Classification Models
Provides model-agnostic training functionality with comprehensive logging.
"""

import os
import time
import json
import sys
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from CNN.models import BaseEEGCNN
from CNN.models.arcface_loss import ArcFaceLoss
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/data_processing')
from CNN.utils import safe_json_dump, convert_numpy_types
from CNN.config import TrainingConfig
from CNN.gpu_utils import (
    get_device, determine_num_gpus_to_use, setup_model_for_gpus,
    get_underlying_model as gpu_get_underlying_model, print_gpu_info
)


class EEGTrainer:
    """
    Systematic trainer for EEG classification models.
    Provides model-agnostic training with comprehensive logging and early stopping.
    """
    
    def __init__(self, model: BaseEEGCNN, config: TrainingConfig, 
                 experiment_name: str = None, num_gpus: int = 1,
                 age_min: float = None, age_max: float = None,
                 device_preference: str = 'auto'):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            experiment_name: Name of the experiment
            num_gpus: Number of GPUs to use (from config, defaults to 1)
            age_min: Minimum age for regression tasks
            age_max: Maximum age for regression tasks
            device_preference: Device preference ('auto', 'cuda', 'cpu')
        
        Raises:
            RuntimeError: If GPU configuration is invalid
            ValueError: If num_gpus is invalid
        """
        self.model = model
        self.config = config
        self.experiment_name = experiment_name or f"{model.__class__.__name__}_{int(time.time())}"
        # Store age range for displaying MAE in years (for regression tasks)
        self.age_min = age_min
        self.age_max = age_max
        
        # Determine number of GPUs to use (explicit, no fallbacks)
        self.num_gpus_used = determine_num_gpus_to_use(num_gpus)
        
        # Setup device (explicit, no fallbacks)
        self.device = get_device(device_preference)
        
        # Print GPU information
        print_gpu_info(num_gpus, self.num_gpus_used, self.device)
        
        # Setup model for GPUs (explicit, raises errors if invalid)
        self.model = setup_model_for_gpus(self.model, self.num_gpus_used, self.device)
        
        # Compile model for faster execution (PyTorch 2.0+)
        # This can provide 10-30% speedup on modern GPUs
        try:
            if hasattr(torch, 'compile'):
                print("Compiling model with torch.compile() for faster execution...")
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("✅ Model compiled successfully")
        except Exception as e:
            print(f"⚠️  Model compilation not available or failed: {e}")
        
        # Setup training components
        # Use MSE loss for regression, CrossEntropyLoss or ArcFace for classification
        self.is_regression = hasattr(config, 'prediction_type') and config.prediction_type == 'regression'
        self.use_arcface = False  # Flag to track if using ArcFace
        
        if self.is_regression:
            self.criterion = nn.MSELoss()
        else:
            # For large classification tasks (e.g., user identification with 3000+ classes),
            # ArcFace loss is more suitable than standard softmax + CrossEntropyLoss
            # ArcFace uses angular margin to improve discrimination for many classes
            if hasattr(config, 'target_key') and config.target_key == 'user_identification':
                # Get number of classes from model
                num_classes = self._get_underlying_model().num_classes
                
                # Use ArcFace for very large classification (2000+ classes)
                # ArcFace is designed for large-scale face recognition (10K+ classes)
                # and is highly suitable for user identification with many classes
                if num_classes > 2000:
                    # Check if model supports feature extraction (required for ArcFace)
                    model = self._get_underlying_model()
                    if hasattr(model, 'extract_features'):
                        # Dynamically determine embedding dimension from model architecture
                        # For EEGUserIdentificationCNN, this is fc2_intermediate.out_features
                        if hasattr(model, 'fc2_intermediate'):
                            # Direct access to embedding dimension from model architecture
                            # This is the preferred method - no dummy input needed
                            embedding_dim = model.fc2_intermediate.out_features
                        else:
                            # Model has extract_features but not fc2_intermediate
                            # This should not happen for EEGUserIdentificationCNN, but handle gracefully
                            raise AttributeError(
                                f"Model {model.__class__.__name__} has extract_features method "
                                f"but missing 'fc2_intermediate' attribute. "
                                f"For ArcFace to work, the model must have an intermediate FC layer "
                                f"that can be accessed via 'fc2_intermediate.out_features'. "
                                f"Please ensure the model architecture supports feature extraction."
                            )
                        
                        # Validate embedding dimension
                        if embedding_dim <= 0:
                            raise ValueError(
                                f"Invalid embedding dimension: {embedding_dim}. "
                                f"Must be > 0."
                            )
                        
                        # ArcFace hyperparameters from config (no hardcoding)
                        margin = config.arcface_margin
                        scale = config.arcface_scale
                        easy_margin = config.arcface_easy_margin
                        
                        self.criterion = ArcFaceLoss(
                            num_classes=num_classes,
                            embedding_dim=embedding_dim,
                            margin=margin,
                            scale=scale,
                            easy_margin=easy_margin
                        )
                        self.use_arcface = True
                        # Move ArcFace to device
                        self.criterion = self.criterion.to(self.device)
                        print(f"✅ Using ArcFace loss for very large classification ({num_classes} classes)")
                        print(f"   Embedding dimension: {embedding_dim} (determined from model architecture)")
                        print(f"   ArcFace hyperparameters: margin={margin}, scale={scale}, easy_margin={easy_margin}")
                        print(f"   ArcFace advantages: angular margin, feature normalization, better discrimination")
                    else:
                        # Fallback to CrossEntropyLoss if model doesn't support feature extraction
                        label_smoothing = config.label_smoothing_very_large
                        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
                        print(f"⚠️  Model doesn't support feature extraction, using CrossEntropyLoss with label_smoothing={label_smoothing}")
                elif num_classes > 100:
                    # For moderately large classification, use label smoothing from config
                    label_smoothing = config.label_smoothing_large
                    self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
                    print(f"✅ Using CrossEntropyLoss with label_smoothing={label_smoothing} for large classification ({num_classes} classes)")
                else:
                    self.criterion = nn.CrossEntropyLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
        
        # Use fused Adam optimizer if available (faster on modern GPUs)
        # Weight decay from config (no hardcoding)
        weight_decay = config.weight_decay
        if weight_decay > 0.0:
            print(f"✅ Using weight decay={weight_decay} for regularization (L2)")
        
        # CRITICAL: Adjust learning rate for ArcFace BEFORE creating optimizer
        # ArcFace is more sensitive to learning rate than CrossEntropyLoss
        # High learning rate can cause immediate model collapse (all predictions to one class)
        # Use configurable multiplier (no hardcoding)
        actual_learning_rate = config.learning_rate
        if self.use_arcface:
            arcface_lr_multiplier = config.arcface_lr_multiplier
            actual_learning_rate = config.learning_rate * arcface_lr_multiplier
            print(f"✅ Learning rate adjusted for ArcFace: {config.learning_rate} -> {actual_learning_rate} (multiplier: {arcface_lr_multiplier})")
        
        # Collect all parameters (model + ArcFace if using)
        params_to_optimize = list(self.model.parameters())
        if self.use_arcface:
            # ArcFace has its own weight matrix that needs to be optimized
            params_to_optimize.extend(list(self.criterion.parameters()))
        
        try:
            self.optimizer = optim.Adam(params_to_optimize, lr=actual_learning_rate, weight_decay=weight_decay, fused=True)
            print("✅ Using fused Adam optimizer")
        except TypeError:
            # Fused optimizer not available, fall back to regular Adam
            self.optimizer = optim.Adam(params_to_optimize, lr=actual_learning_rate, weight_decay=weight_decay)
        
        # Enable mixed precision training (AMP) for 1.5-2x speedup
        self.use_amp = True
        # Use new torch.amp API (PyTorch 2.0+)
        self.scaler = torch.amp.GradScaler('cuda')
        print("✅ Mixed precision training (AMP) enabled")
        
        # Gradient clipping configuration
        # Prevents exploding gradients, especially important for large-scale classification
        max_grad_norm = config.max_grad_norm
        if max_grad_norm is not None and max_grad_norm > 0.0:
            print(f"✅ Gradient clipping enabled: max_norm={max_grad_norm}")
        else:
            print("⚠️  Gradient clipping disabled (max_grad_norm=None or <= 0)")
        
        # Learning rate scheduler hyperparameters from config (no hardcoding)
        scheduler_patience = config.scheduler_patience
        if scheduler_patience is None:
            # Default: use 3 for user identification, 10 for others
            scheduler_patience = 3 if (hasattr(config, 'target_key') and config.target_key == 'user_identification') else 10
        scheduler_factor = config.scheduler_factor
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True
        )
        print(f"✅ Learning rate scheduler: factor={scheduler_factor}, patience={scheduler_patience}")
        
        # Learning rate warmup configuration
        # Gradually increases LR from 0 to target over warmup_epochs
        # This prevents large gradient updates in early epochs that can cause collapse
        self.warmup_epochs = config.warmup_epochs if hasattr(config, 'warmup_epochs') else 0
        self.base_learning_rate = actual_learning_rate  # Store base LR for warmup calculation
        if self.warmup_epochs > 0:
            print(f"✅ Learning rate warmup enabled: {self.warmup_epochs} epochs")
            # Initialize LR to 0 for warmup
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.0
        else:
            print("⚠️  Learning rate warmup disabled")
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0
    
    def _get_best_checkpoint_path(self, experiment_results_dir: str = None) -> str:
        """Get the path to the best checkpoint file."""
        if experiment_results_dir:
            checkpoint_dir = os.path.join(experiment_results_dir, 'checkpoints')
        else:
            model_name = self._get_underlying_model().__class__.__name__
            checkpoint_dir = os.path.join(self.config.checkpoint_dir, model_name)
        
        best_path = os.path.join(checkpoint_dir, f'{self.experiment_name}_best.pth')
        return best_path
    
    def _get_underlying_model(self):
        """Get the underlying model, unwrapping DataParallel and torch.compile wrappers if needed."""
        return gpu_get_underlying_model(self.model)
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        # Reset collapse counter at start of each epoch
        # This allows model to recover if collapse was temporary
        if hasattr(self, '_collapse_batch_count'):
            self._collapse_batch_count = 0
        
        batch_start_time = time.time()
        num_batches = 0
        first_batch_loaded = False
        for batch_idx, batch in enumerate(train_loader):
            num_batches = batch_idx + 1
            if not first_batch_loaded:
                elapsed = time.time() - batch_start_time
                print(f"  ✅ First batch loaded! (took {elapsed:.1f}s)")
                sys.stdout.flush()
                first_batch_loaded = True
                batch_start_time = time.time()  # Reset timer for batch processing
            if batch_idx % 1000 == 0 and batch_idx > 0:
                elapsed = time.time() - batch_start_time
                print(f"  Processed {batch_idx} batches, elapsed: {elapsed:.1f}s")
                sys.stdout.flush()
            
            # Non-blocking transfer for faster data loading
            inputs = batch['eeg_data'].to(self.device, non_blocking=True)
            
            # Handle multi_output case where we have separate gender and age keys
            if self.config.target_key == 'multi_output' and 'gender' in batch and 'age' in batch:
                labels = {
                    'gender': batch['gender'].to(self.device, non_blocking=True),
                    'age': batch['age'].to(self.device, non_blocking=True)
                }
            elif self.config.target_key == 'combined':
                gender = batch['gender'].to(self.device, non_blocking=True)
                age = batch['age'].to(self.device, non_blocking=True)
                # For combined experiments, age should be classification (0, 1, 2), not regression
                # Gender should also be classification (0, 1)
                # Optimized: Use vectorized operations instead of Python list comprehension
                # Convert to int tensors (gender: 0/1, age: 0/1/2)
                gender_int = gender.long()  # Already 0 or 1
                age_int = age.long()  # Already 0, 1, or 2
                # Combined class = gender * 3 + age (vectorized)
                combined_classes = gender_int * 3 + age_int
                labels = combined_classes
            else:
                # Get labels from batch - validate key exists
                if self.config.target_key not in batch:
                    raise KeyError(
                        f"Target key '{self.config.target_key}' not found in batch. "
                        f"Available keys: {list(batch.keys())}. "
                        f"This indicates a data loading issue."
                    )
                labels = batch[self.config.target_key].to(self.device, non_blocking=True)
                
                # For regression, ensure labels are float and have correct shape
                if self.is_regression:
                    if labels.dtype != torch.float32:
                        labels = labels.float()
                    # Reshape if needed: (batch_size,) -> (batch_size, 1)
                    if labels.dim() == 1:
                        labels = labels.unsqueeze(1)
                else:
                    # For classification: validate labels
                    # Ensure labels are long integers for classification
                    if labels.dtype != torch.long:
                        labels = labels.long()
                    
                    # Validate label range for classification tasks
                    num_classes = self._get_underlying_model().num_classes
                    if labels.max() >= num_classes or labels.min() < 0:
                        invalid_labels = labels[(labels >= num_classes) | (labels < 0)]
                        raise ValueError(
                            f"Invalid labels detected in training batch: {invalid_labels.cpu().numpy()}. "
                            f"Labels must be in range [0, {num_classes-1}], but found values outside this range. "
                            f"This indicates a data inconsistency issue. "
                            f"Check that the transform is correctly mapping participant IDs to class indices."
                        )
            
            self.optimizer.zero_grad()
            
            # Mixed precision training (AMP)
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    # For ArcFace, extract features instead of final logits
                    if self.use_arcface:
                        # Extract feature embeddings (before final FC layer)
                        features = self._get_underlying_model().extract_features(inputs)
                        # ArcFace loss computes logits internally from normalized features
                        loss = self.criterion(features, labels)
                        # For accuracy calculation, compute logits from ArcFace
                        with torch.no_grad():
                            outputs = self.criterion.compute_logits(features)
                            # Convert to float32 for accuracy calculation (ArcFace outputs may be in float16)
                            outputs = outputs.float() if outputs.dtype == torch.float16 else outputs
                    else:
                        # Standard forward pass
                        outputs = self.model(inputs)
                
                # Convert outputs to float32 for loss computation to ensure numerical stability
                # Mixed precision outputs may be in float16, which can cause numerical issues
                # with large number of classes and label smoothing
                if not self.use_arcface:  # ArcFace already computed loss and converted outputs
                    if isinstance(outputs, dict):
                        outputs = {k: v.float() if v.dtype == torch.float16 else v for k, v in outputs.items()}
                    else:
                        outputs = outputs.float() if outputs.dtype == torch.float16 else outputs
                    
                    # Handle multi-output models
                    if isinstance(outputs, dict):
                        # Multi-output model (e.g., MultiOutputCNN)
                        gender_loss = self.criterion(outputs['gender'], labels['gender'])
                        age_loss = self.criterion(outputs['age'], labels['age'])
                        loss = gender_loss + age_loss  # Combined loss
                    else:
                        # Single-output model (e.g., EEGCNN, CombinedCNN, EEGAgeRegressionCNN)
                        # Clamp outputs to prevent extreme values that cause numerical overflow
                        # This is especially important with mixed precision and large number of classes
                        # Use config values (no hardcoding)
                        outputs_clamped = torch.clamp(outputs, min=self.config.output_clamp_min, max=self.config.output_clamp_max)
                        loss = self.criterion(outputs_clamped, labels)
                
                # Calculate metrics outside autocast
                # For ArcFace, outputs were computed and converted to float32 in the autocast block
                # No additional processing needed
                
                if isinstance(outputs, dict):
                    # Multi-output model: get predicted class indices (argmax)
                    gender_pred = torch.argmax(outputs['gender'], dim=1)
                    age_pred = torch.argmax(outputs['age'], dim=1)
                    gender_correct = (gender_pred == labels['gender']).sum().item()
                    age_correct = (age_pred == labels['age']).sum().item()
                    correct += (gender_correct + age_correct) / 2  # Average of both accuracies
                else:
                    # Single-output model
                    if self.is_regression:
                        # For regression: accumulate sum of absolute errors
                        # We'll divide by total later to get overall MAE
                        sum_absolute_errors = torch.abs(outputs - labels).sum().item()
                        correct += sum_absolute_errors
                    else:
                        # For classification: calculate accuracy
                        predicted = torch.argmax(outputs, dim=1)
                        batch_correct = (predicted == labels).sum().item()
                        correct += batch_correct
                        
                        # CRITICAL: Detect model collapse (all predictions to same class)
                        # Track collapse across batches to avoid false positives from temporary collapses
                        unique_predictions = len(torch.unique(predicted))
                        if unique_predictions == 1:
                            # Track collapse occurrences
                            if not hasattr(self, '_collapse_batch_count'):
                                self._collapse_batch_count = 0
                            self._collapse_batch_count += 1
                            
                            # Only raise error if collapse persists across multiple batches
                            # This prevents false positives from temporary collapses in early training
                            # Increased from 5 to 10 to allow more recovery time, especially during warmup
                            # For very large classification (3000+ classes), model needs more time to recover
                            collapse_threshold = 10  # Allow up to 10 consecutive batches with collapse
                            if self._collapse_batch_count >= collapse_threshold:
                                collapsed_class = predicted[0].item()
                                raise RuntimeError(
                                    f"Model collapse detected in training: {self._collapse_batch_count} consecutive batches "
                                    f"with all samples predicted as class {collapsed_class}. "
                                    f"This indicates a serious training issue. Possible causes: "
                                    f"1) ArcFace weight initialization problem, "
                                    f"2) Learning rate too high causing early collapse, "
                                    f"3) Feature extraction producing identical features, "
                                    f"4) Numerical instability, "
                                    f"5) Batch correlation (samples from same participant in batch). "
                                    f"Check model initialization, learning rate, feature diversity, and data shuffling."
                                )
                            elif self._collapse_batch_count == 1:
                                # First occurrence - log warning but continue
                                collapsed_class = predicted[0].item()
                                print(f"⚠️  WARNING: Model collapse detected in batch (all predictions = class {collapsed_class}). "
                                      f"Monitoring for persistence...")
                        else:
                            # Reset collapse counter if predictions are diverse
                            if hasattr(self, '_collapse_batch_count'):
                                if self._collapse_batch_count > 0:
                                    print(f"✅ Model recovered: Predictions are now diverse (unique: {unique_predictions})")
                                self._collapse_batch_count = 0
                
                # Scale loss and backward pass for mixed precision
                self.scaler.scale(loss).backward()
                
                # Gradient clipping for training stability
                # Prevents exploding gradients, especially important for large-scale classification
                # Use config value (no hardcoding)
                max_grad_norm = self.config.max_grad_norm
                if max_grad_norm is not None and max_grad_norm > 0.0:
                    # Unscale gradients before clipping (required for mixed precision)
                    self.scaler.unscale_(self.optimizer)
                    # CRITICAL: Clip all parameters together (model + ArcFace if using)
                    # Clipping separately can cause issues - all parameters should be clipped as one group
                    params_to_clip = list(self.model.parameters())
                    if self.use_arcface:
                        params_to_clip.extend(list(self.criterion.parameters()))
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(params_to_clip, max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                # For ArcFace, extract features instead of final logits
                if self.use_arcface:
                    # Extract feature embeddings (before final FC layer)
                    features = self._get_underlying_model().extract_features(inputs)
                    # ArcFace loss computes logits internally from normalized features
                    loss = self.criterion(features, labels)
                    # For accuracy calculation, compute logits from ArcFace
                    with torch.no_grad():
                        outputs = self.criterion.compute_logits(features)
                else:
                    # Standard forward pass
                    outputs = self.model(inputs)
                
                # Handle multi-output models
                if isinstance(outputs, dict):
                    # Multi-output model (e.g., MultiOutputCNN)
                    if not self.use_arcface:  # ArcFace already computed loss
                        gender_loss = self.criterion(outputs['gender'], labels['gender'])
                        age_loss = self.criterion(outputs['age'], labels['age'])
                        loss = gender_loss + age_loss  # Combined loss
                    
                    # Calculate combined accuracy: get predicted class indices (argmax)
                    gender_pred = torch.argmax(outputs['gender'], dim=1)
                    age_pred = torch.argmax(outputs['age'], dim=1)
                    gender_correct = (gender_pred == labels['gender']).sum().item()
                    age_correct = (age_pred == labels['age']).sum().item()
                    correct += (gender_correct + age_correct) / 2  # Average of both accuracies
                else:
                    # Single-output model (e.g., EEGCNN, CombinedCNN, EEGAgeRegressionCNN)
                    if not self.use_arcface:  # ArcFace already computed loss
                        loss = self.criterion(outputs, labels)
                    
                    if self.is_regression:
                        # For regression: accumulate sum of absolute errors
                        # We'll divide by total later to get overall MAE
                        sum_absolute_errors = torch.abs(outputs - labels).sum().item()
                        correct += sum_absolute_errors
                    else:
                        # For classification: calculate accuracy
                        predicted = torch.argmax(outputs, dim=1)
                        batch_correct = (predicted == labels).sum().item()
                        correct += batch_correct
                        
                        # CRITICAL: Detect model collapse (all predictions to same class)
                        # Track collapse across batches to avoid false positives from temporary collapses
                        unique_predictions = len(torch.unique(predicted))
                        if unique_predictions == 1:
                            # Track collapse occurrences
                            if not hasattr(self, '_collapse_batch_count'):
                                self._collapse_batch_count = 0
                            self._collapse_batch_count += 1
                            
                            # Only raise error if collapse persists across multiple batches
                            # This prevents false positives from temporary collapses in early training
                            # Increased from 5 to 10 to allow more recovery time, especially during warmup
                            # For very large classification (3000+ classes), model needs more time to recover
                            collapse_threshold = 10  # Allow up to 10 consecutive batches with collapse
                            if self._collapse_batch_count >= collapse_threshold:
                                collapsed_class = predicted[0].item()
                                raise RuntimeError(
                                    f"Model collapse detected in training: {self._collapse_batch_count} consecutive batches "
                                    f"with all samples predicted as class {collapsed_class}. "
                                    f"This indicates a serious training issue. Possible causes: "
                                    f"1) ArcFace weight initialization problem, "
                                    f"2) Learning rate too high causing early collapse, "
                                    f"3) Feature extraction producing identical features, "
                                    f"4) Numerical instability, "
                                    f"5) Batch correlation (samples from same participant in batch). "
                                    f"Check model initialization, learning rate, feature diversity, and data shuffling."
                                )
                            elif self._collapse_batch_count == 1:
                                # First occurrence - log warning but continue
                                collapsed_class = predicted[0].item()
                                print(f"⚠️  WARNING: Model collapse detected in batch (all predictions = class {collapsed_class}). "
                                      f"Monitoring for persistence...")
                        else:
                            # Reset collapse counter if predictions are diverse
                            if hasattr(self, '_collapse_batch_count'):
                                if self._collapse_batch_count > 0:
                                    print(f"✅ Model recovered: Predictions are now diverse (unique: {unique_predictions})")
                                self._collapse_batch_count = 0
                
                loss.backward()
                
                # Gradient clipping for training stability
                # Prevents exploding gradients, especially important for large-scale classification
                # Use config value (no hardcoding)
                max_grad_norm = self.config.max_grad_norm
                if max_grad_norm is not None and max_grad_norm > 0.0:
                    # CRITICAL: Clip all parameters together (model + ArcFace if using)
                    # Clipping separately can cause issues - all parameters should be clipped as one group
                    params_to_clip = list(self.model.parameters())
                    if self.use_arcface:
                        params_to_clip.extend(list(self.criterion.parameters()))
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(params_to_clip, max_grad_norm)
                
                self.optimizer.step()
            
            total_loss += loss.item()
            total += labels.size(0) if not isinstance(labels, dict) else labels['gender'].size(0)
        
        # Use num_batches instead of len(train_loader) since IterableDataset doesn't support __len__()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Calculate metric based on task type
        # For regression: correct = sum(|outputs - labels|) across all batches, so correct/total = overall MAE
        # For classification: correct = sum(correct_predictions), so correct/total = accuracy
        metric = correct / total if total > 0 else 0.0
        return avg_loss, metric
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        # Reset collapse counter at start of each validation epoch
        # This allows model to recover if collapse was temporary
        if hasattr(self, '_val_collapse_batch_count'):
            self._val_collapse_batch_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                num_batches = batch_idx + 1
                # Non-blocking transfer for faster data loading
                inputs = batch['eeg_data'].to(self.device, non_blocking=True)
                
                # Handle multi_output case where we have separate gender and age keys
                if self.config.target_key == 'multi_output' and 'gender' in batch and 'age' in batch:
                    labels = {
                        'gender': batch['gender'].to(self.device, non_blocking=True),
                        'age': batch['age'].to(self.device, non_blocking=True)
                    }
                elif self.config.target_key == 'combined':
                    # For combined experiments, compute combined class from gender and age
                    gender = batch['gender'].to(self.device, non_blocking=True)
                    age = batch['age'].to(self.device, non_blocking=True)
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
                    if self.config.target_key not in batch:
                        raise KeyError(
                            f"Target key '{self.config.target_key}' not found in batch. "
                            f"Available keys: {list(batch.keys())}. "
                            f"This indicates a data loading issue."
                        )
                    labels = batch[self.config.target_key].to(self.device, non_blocking=True)
                    
                    # Validate labels for classification tasks
                    if not self.is_regression:
                        # Ensure labels are long integers for classification
                        if labels.dtype != torch.long:
                            labels = labels.long()
                        
                        # Validate label range for classification tasks
                        num_classes = self._get_underlying_model().num_classes
                        if labels.max() >= num_classes or labels.min() < 0:
                            invalid_labels = labels[(labels >= num_classes) | (labels < 0)]
                            raise ValueError(
                                f"Invalid labels detected: {invalid_labels.cpu().numpy()}. "
                                f"Labels must be in range [0, {num_classes-1}], but found values outside this range. "
                                f"This indicates a data inconsistency issue. "
                                f"Check that the transform is correctly mapping participant IDs to class indices."
                            )
                    else:
                        # For regression, ensure labels are float and have correct shape
                        if labels.dtype != torch.float32:
                            labels = labels.float()
                        # Reshape if needed: (batch_size,) -> (batch_size, 1)
                        if labels.dim() == 1:
                            labels = labels.unsqueeze(1)
                
                # For very large classification with label smoothing, use full precision for validation
                # to avoid numerical instability. Mixed precision can cause overflow with 3000+ classes.
                # However, ArcFace handles normalization internally, so it's more stable
                use_amp_for_val = self.use_amp
                if (hasattr(self.config, 'target_key') and 
                    self.config.target_key == 'user_identification' and
                    self._get_underlying_model().num_classes > 2000 and
                    not self.use_arcface):  # ArcFace is more stable, can use AMP
                    # Disable AMP for validation with very large classification to ensure numerical stability
                    use_amp_for_val = False
                
                # For ArcFace, extract features instead of final logits
                if self.use_arcface:
                    # Extract feature embeddings (before final FC layer)
                    features = self._get_underlying_model().extract_features(inputs)
                    # ArcFace loss computes logits internally from normalized features
                    loss = self.criterion(features, labels)
                    # For accuracy calculation, compute logits from ArcFace
                    outputs = self.criterion.compute_logits(features)
                    # CRITICAL: Convert to float32 for accuracy calculation (ArcFace outputs may be in float16)
                    # This ensures numerical stability and correct argmax computation
                    outputs = outputs.float() if outputs.dtype == torch.float16 else outputs
                elif use_amp_for_val:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(inputs)
                    # Convert outputs to float32 for loss computation to ensure numerical stability
                    # Mixed precision outputs may be in float16, which can cause numerical issues
                    # with large number of classes and label smoothing
                    if isinstance(outputs, dict):
                        outputs = {k: v.float() if v.dtype == torch.float16 else v for k, v in outputs.items()}
                    else:
                        outputs = outputs.float() if outputs.dtype == torch.float16 else outputs
                else:
                    outputs = self.model(inputs)
                
                # Handle multi-output models
                if isinstance(outputs, dict):
                    # Multi-output model (e.g., MultiOutputCNN)
                    if not self.use_arcface:  # ArcFace already computed loss
                        gender_loss = self.criterion(outputs['gender'], labels['gender'])
                        age_loss = self.criterion(outputs['age'], labels['age'])
                        loss = gender_loss + age_loss  # Combined loss
                    
                    # Calculate combined accuracy: get predicted class indices (argmax)
                    gender_pred = torch.argmax(outputs['gender'], dim=1)
                    age_pred = torch.argmax(outputs['age'], dim=1)
                    gender_correct = (gender_pred == labels['gender']).sum().item()
                    age_correct = (age_pred == labels['age']).sum().item()
                    correct += (gender_correct + age_correct) / 2  # Average of both accuracies
                else:
                    # Single-output model (e.g., EEGCNN, CombinedCNN, EEGAgeRegressionCNN)
                    
                    if not self.use_arcface:  # ArcFace already computed loss
                        # Validate outputs don't contain NaN/Inf
                        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                            nan_count = torch.isnan(outputs).sum().item()
                            inf_count = torch.isinf(outputs).any().item()
                            raise ValueError(
                                f"NaN/Inf detected in model outputs: NaN count={nan_count}, Inf detected={inf_count}. "
                                f"This indicates a numerical instability issue. "
                                f"Check model architecture, learning rate, and input data."
                            )
                        
                        # Clamp outputs to prevent extreme values that cause numerical overflow
                        # This is especially important with mixed precision and large number of classes
                        # Clamp to reasonable range to prevent overflow in softmax and log operations
                        # Use config values (no hardcoding)
                        outputs_clamped = torch.clamp(outputs, min=self.config.output_clamp_min, max=self.config.output_clamp_max)
                        
                        loss = self.criterion(outputs_clamped, labels)
                    else:
                        # For ArcFace, outputs are already computed from logits
                        pass
                    
                    # Validate loss is finite BEFORE adding to total_loss
                    # This is critical - we must check before accumulation
                    loss_value = loss.item() if hasattr(loss, 'item') else float(loss)
                    
                    if not np.isfinite(loss_value):
                        # Get diagnostic information before raising error
                        num_classes = self._get_underlying_model().num_classes
                        label_min = labels.min().item() if labels.numel() > 0 else None
                        label_max = labels.max().item() if labels.numel() > 0 else None
                        output_min = outputs.min().item() if outputs.numel() > 0 else None
                        output_max = outputs.max().item() if outputs.numel() > 0 else None
                        output_mean = outputs.mean().item() if outputs.numel() > 0 else None
                        
                        error_msg = (
                            f"Non-finite loss detected during validation: {loss_value}. "
                            f"This usually indicates: "
                            f"1) Labels out of range for classification (should be [0, {num_classes-1}]), "
                            f"2) Numerical instability in model outputs, or "
                            f"3) Issue with loss function (e.g., label smoothing with invalid labels). "
                            f"Diagnostics: "
                            f"Model has {num_classes} classes, "
                            f"Label range: [{label_min}, {label_max}], "
                            f"Output range: [{output_min}, {output_max}], "
                            f"Output mean: {output_mean}, "
                            f"Batch size: {labels.size(0)}"
                        )
                        raise ValueError(error_msg)
                    
                    if self.is_regression:
                        # For regression: accumulate sum of absolute errors
                        # We'll divide by total later to get overall MAE
                        sum_absolute_errors = torch.abs(outputs - labels).sum().item()
                        correct += sum_absolute_errors
                    else:
                        # For classification: calculate accuracy
                        predicted = torch.argmax(outputs, dim=1)
                        batch_correct = (predicted == labels).sum().item()
                        correct += batch_correct
                        
                        # CRITICAL: Detect model collapse (all predictions to same class)
                        # Track collapse across batches to avoid false positives from temporary collapses
                        # Use same threshold mechanism as training for consistency
                        unique_predictions = len(torch.unique(predicted))
                        if unique_predictions == 1:
                            # Track collapse occurrences
                            if not hasattr(self, '_val_collapse_batch_count'):
                                self._val_collapse_batch_count = 0
                            self._val_collapse_batch_count += 1
                            
                            # Only raise error if collapse persists across multiple batches
                            # This prevents false positives from temporary collapses
                            # For very large classification (3000+ classes), model needs more time to recover
                            collapse_threshold = 10  # Allow up to 10 consecutive batches with collapse
                            if self._val_collapse_batch_count >= collapse_threshold:
                                collapsed_class = predicted[0].item()
                                raise RuntimeError(
                                    f"Model collapse detected in validation: {self._val_collapse_batch_count} consecutive batches "
                                    f"with all samples predicted as class {collapsed_class}. "
                                    f"This indicates a serious training issue. Possible causes: "
                                    f"1) ArcFace weight initialization problem, "
                                    f"2) Learning rate too high causing early collapse, "
                                    f"3) Feature extraction producing identical features, "
                                    f"4) Numerical instability. "
                                    f"Check model initialization, learning rate, and feature diversity."
                                )
                            elif self._val_collapse_batch_count == 1:
                                # First occurrence - log warning but continue
                                collapsed_class = predicted[0].item()
                                print(f"⚠️  WARNING: Model collapse detected in validation batch (all predictions = class {collapsed_class}). "
                                      f"Monitoring for persistence...")
                        else:
                            # Reset collapse counter if predictions are diverse
                            if hasattr(self, '_val_collapse_batch_count'):
                                if self._val_collapse_batch_count > 0:
                                    print(f"✅ Model recovered in validation: Predictions are now diverse (unique: {unique_predictions})")
                                self._val_collapse_batch_count = 0
                        
                        # Diagnostic logging for debugging validation accuracy issue
                        # Only log first batch of first validation epoch to avoid spam
                        if (not hasattr(self, '_val_diagnostic_logged') and 
                            batch_idx == 0 and 
                            self.use_arcface):
                            self._val_diagnostic_logged = True
                            # Extract features for diagnostic
                            with torch.no_grad():
                                features = self._get_underlying_model().extract_features(inputs)
                                # Check feature statistics
                                feature_norm = torch.norm(features, p=2, dim=1)
                                feature_mean_norm = feature_norm.mean().item()
                                feature_std_norm = feature_norm.std().item()
                                # Check feature diversity (how different features are)
                                feature_cosine = torch.nn.functional.cosine_similarity(
                                    features.unsqueeze(1), features.unsqueeze(0), dim=2
                                )
                                # Remove diagonal (self-similarity)
                                mask = ~torch.eye(feature_cosine.size(0), dtype=torch.bool, device=feature_cosine.device)
                                feature_similarity_mean = feature_cosine[mask].mean().item()
                            
                            print(f"\n🔍 Validation Diagnostic (first batch, epoch 1):")
                            print(f"   Batch size: {labels.size(0)}")
                            print(f"   Outputs shape: {outputs.shape}, dtype: {outputs.dtype}")
                            print(f"   Outputs stats: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}, std={outputs.std().item():.4f}")
                            print(f"   Labels range: [{labels.min().item()}, {labels.max().item()}], unique: {len(torch.unique(labels))}")
                            print(f"   Predicted range: [{predicted.min().item()}, {predicted.max().item()}], unique: {unique_predictions}")
                            print(f"   Batch correct: {batch_correct}/{labels.size(0)} = {batch_correct/labels.size(0)*100:.4f}%")
                            unique_preds, counts = torch.unique(predicted, return_counts=True)
                            top_pred = unique_preds[counts.argmax()].item()
                            top_count = counts.max().item()
                            print(f"   Most common prediction: class {top_pred} ({top_count}/{labels.size(0)} = {top_count/labels.size(0)*100:.2f}%)")
                            print(f"   📊 Feature Quality:")
                            print(f"      Feature norm: mean={feature_mean_norm:.4f}, std={feature_std_norm:.4f}")
                            print(f"      Feature similarity (cosine): mean={feature_similarity_mean:.4f} (lower is better, <0.5 is good)")
                            if feature_similarity_mean > 0.7:
                                print(f"      ⚠️  WARNING: Features are too similar (mean similarity > 0.7)")
                                print(f"         This suggests features lack discrimination - model may need more capacity or better training")
                            if batch_correct == 0:
                                print(f"   ⚠️  WARNING: Zero correct predictions in first validation batch!")
                
                # Only add to total_loss if loss is finite (double-check)
                if np.isfinite(loss_value):
                    total_loss += loss_value
                else:
                    # This should never happen due to check above, but add safety check
                    raise ValueError(f"Attempted to add non-finite loss to total: {loss_value}")
                total += labels.size(0) if not isinstance(labels, dict) else labels['gender'].size(0)
        
        # Use num_batches instead of len(val_loader) since IterableDataset doesn't support __len__()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Calculate metric based on task type
        # For regression: correct = sum(|outputs - labels|) across all batches, so correct/total = overall MAE
        # For classification: correct = sum(correct_predictions), so correct/total = accuracy
        metric = correct / total if total > 0 else 0.0
        return avg_loss, metric
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, is_last: bool = False, experiment_results_dir: str = None):
        """Save model checkpoint. Only saves best and last checkpoints."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
            'model_info': self._get_underlying_model().get_model_info(),
            'use_arcface': self.use_arcface  # Save ArcFace flag for checkpoint loading
        }
        
        # Save ArcFace state if using ArcFace
        if self.use_arcface:
            checkpoint['arcface_state_dict'] = self.criterion.state_dict()
        
        # Add optional fields if they exist (for backward compatibility with old checkpoints)
        # These fields help with resume functionality but aren't strictly required
        if hasattr(self, 'best_epoch'):
            checkpoint['best_epoch'] = self.best_epoch
        if hasattr(self, 'patience_counter'):
            checkpoint['patience_counter'] = self.patience_counter
        
        # Use experiment-specific directory if provided, otherwise fall back to global
        if experiment_results_dir:
            checkpoint_dir = os.path.join(experiment_results_dir, 'checkpoints')
        else:
            # Fallback to global checkpoint directory
            model_name = self._get_underlying_model().__class__.__name__
            checkpoint_dir = os.path.join(self.config.checkpoint_dir, model_name)
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save best model
        if is_best:
            best_path = os.path.join(
                checkpoint_dir, 
                f'{self.experiment_name}_best.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"New best model saved at epoch {epoch+1} in {checkpoint_dir}")
        
        # Save last model
        if is_last:
            last_path = os.path.join(
                checkpoint_dir, 
                f'{self.experiment_name}_last.pth'
            )
            torch.save(checkpoint, last_path)
            print(f"Last model saved at epoch {epoch+1} in {checkpoint_dir}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, progress_callback=None, experiment_results_dir: str = None, resume_from_best: bool = True) -> Dict[str, Any]:
        """
        Train the model with early stopping and comprehensive logging.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            progress_callback: Optional callback function for progress updates
            experiment_results_dir: Directory to save experiment-specific checkpoints
            resume_from_best: If True, resume from best checkpoint if it exists
        
        Returns:
            Dictionary with training results and metrics
        """
        print(f"Starting training for {self.experiment_name}")
        print(f"Model: {self._get_underlying_model().__class__.__name__}")
        print(f"Target: {self.config.target_key}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("-" * 60)
        
        # Check for existing checkpoint to resume from
        start_epoch = 0
        resumed_from_checkpoint = False
        if resume_from_best:
            checkpoint_path = self._get_best_checkpoint_path(experiment_results_dir)
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"📂 Found existing checkpoint: {checkpoint_path}")
                print("   Attempting to resume training from best checkpoint...")
                try:
                    # Try strict loading first
                    checkpoint = load_checkpoint(checkpoint_path, self._get_underlying_model(), self.device, strict=True)
                    print("   ✅ Checkpoint loaded successfully (architecture matches)")
                    
                    # Restore training state
                    if 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'scheduler_state_dict' in checkpoint:
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    if 'best_val_loss' in checkpoint:
                        self.best_val_loss = checkpoint['best_val_loss']
                    if 'epoch' in checkpoint:
                        start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
                        # For old checkpoints: if best_epoch is missing, use the checkpoint's epoch
                        # (since this is the best checkpoint, the epoch it was saved at is the best epoch)
                        # Explicit check instead of .get() to avoid silent errors
                        if 'best_epoch' in checkpoint:
                            self.best_epoch = checkpoint['best_epoch']
                        else:
                            # Fallback for backward compatibility with old checkpoints
                            self.best_epoch = checkpoint['epoch']
                            print("   ⚠️  Warning: Checkpoint missing 'best_epoch', using 'epoch' as fallback")
                    # For old checkpoints: if patience_counter is missing, initialize to 0
                    # (we don't know the exact value, but starting at 0 is safe)
                    if 'patience_counter' in checkpoint:
                        self.patience_counter = checkpoint['patience_counter']
                    # Restore ArcFace state if present
                    # Explicit check instead of .get() to avoid silent errors
                    if 'use_arcface' in checkpoint:
                        checkpoint_use_arcface = checkpoint['use_arcface']
                    else:
                        # Old checkpoint without use_arcface flag - assume CrossEntropyLoss
                        checkpoint_use_arcface = False
                        print("   ⚠️  Warning: Checkpoint missing 'use_arcface' flag, assuming CrossEntropyLoss")
                    
                    if checkpoint_use_arcface != self.use_arcface:
                        # Checkpoint was saved with different loss function
                        if checkpoint_use_arcface and not self.use_arcface:
                            raise RuntimeError(
                                f"Checkpoint was saved with ArcFace loss, but current configuration uses CrossEntropyLoss. "
                                f"This is incompatible. Please use the same loss function configuration or start fresh training."
                            )
                        elif not checkpoint_use_arcface and self.use_arcface:
                            raise RuntimeError(
                                f"Checkpoint was saved with CrossEntropyLoss, but current configuration uses ArcFace loss. "
                                f"This is incompatible. Please use the same loss function configuration or start fresh training."
                            )
                    
                    if 'arcface_state_dict' in checkpoint:
                        if self.use_arcface:
                            try:
                                self.criterion.load_state_dict(checkpoint['arcface_state_dict'])
                                print("   ✅ ArcFace state restored from checkpoint")
                            except Exception as e:
                                raise RuntimeError(
                                    f"Failed to load ArcFace state from checkpoint. "
                                    f"This may indicate an architecture mismatch. Error: {e}"
                                ) from e
                        else:
                            # Checkpoint has ArcFace state but we're not using ArcFace
                            # This shouldn't happen if use_arcface flag is correct, but handle gracefully
                            print("   ⚠️  Warning: Checkpoint contains ArcFace state but current configuration doesn't use ArcFace")
                    
                    # For old checkpoints: if patience_counter is missing, initialize to 0
                    # (we don't know the exact value, but starting at 0 is safe)
                    if 'patience_counter' in checkpoint:
                        self.patience_counter = checkpoint['patience_counter']
                    else:
                        self.patience_counter = 0
                    
                    resumed_from_checkpoint = True
                    print(f"   Resuming from epoch {start_epoch + 1}/{self.config.epochs}")
                    print(f"   Best validation loss so far: {self.best_val_loss:.4f} (epoch {self.best_epoch + 1})")
                except RuntimeError as e:
                    # Architecture mismatch - automatically skip checkpoint and start fresh
                    print(f"   ⚠️  Architecture mismatch detected!")
                    print(f"   💡 The model architecture has changed since this checkpoint was saved.")
                    print(f"   💡 The old checkpoint will be ignored and training will start from scratch.")
                    print(f"   💡 Old checkpoint location: {checkpoint_path}")
                    print(f"   💡 To permanently remove it, run: rm {checkpoint_path}")
                    # Continue with fresh training (don't raise error)
                    print("   ✅ Starting fresh training with new architecture...")
            else:
                print("   No existing checkpoint found. Starting fresh training.")
        
        start_time = time.time()
        
        # Log that we're about to start loading first batch
        print(f"⏳ Loading first batch (batch_size={train_loader.batch_size}, num_workers={train_loader.num_workers})...")
        # Note: IterableDataset doesn't support __len__(), so we can't print dataset size
        sys.stdout.flush()
        
        for epoch in range(start_epoch, self.config.epochs):
            epoch_start = time.time()
            print(f"\n🔄 Starting epoch {epoch+1}/{self.config.epochs}")
            sys.stdout.flush()
            
            # Learning rate warmup: gradually increase LR from 0 to base_learning_rate
            # This prevents large gradient updates in early epochs that can cause collapse
            if hasattr(self, 'warmup_epochs') and self.warmup_epochs > 0 and epoch < self.warmup_epochs:
                # Linear warmup: LR = base_lr * (epoch + 1) / warmup_epochs
                warmup_lr = self.base_learning_rate * (epoch + 1) / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                if epoch == 0 or (epoch + 1) % max(1, self.warmup_epochs // 3) == 0:
                    print(f"   🔥 Warmup: LR = {warmup_lr:.6f} (target: {self.base_learning_rate:.6f})")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Validate val_loss is finite before passing to scheduler
            if not np.isfinite(val_loss):
                raise ValueError(
                    f"Validation loss is non-finite: {val_loss}. "
                    f"This indicates a serious issue with validation. "
                    f"Training cannot continue. Check validation data and model outputs."
                )
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(epoch, self.config.epochs, train_loss, val_loss, train_acc, val_acc, current_lr, epoch_time)
            else:
                # For regression, convert normalized MAE to years if age range is available
                if self.is_regression and self.age_min is not None and self.age_max is not None:
                    age_range = self.age_max - self.age_min
                    train_mae_years = train_acc * age_range
                    val_mae_years = val_acc * age_range
                    print(f"  → Epoch {epoch+1:3d}/{self.config.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                        f"Train MAE: {train_mae_years:.2f} years ({train_acc:.4f} norm) | Val MAE: {val_mae_years:.2f} years ({val_acc:.4f} norm) | "
                        f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")
                else:
                    print(f"  → Epoch {epoch+1:3d}/{self.config.epochs} | "
                        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                        f"Train {'MAE' if self.is_regression else 'Acc'}: {train_acc:.4f} | Val {'MAE' if self.is_regression else 'Acc'}: {val_acc:.4f} | "
                        f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")
            
            # Check for improvement
            # Note: We compare with min_delta to avoid saving checkpoints for tiny improvements
            # When resuming, best_val_loss should be loaded from checkpoint, so this comparison
            # ensures we only save a new best if there's a meaningful improvement
            improvement_threshold = self.best_val_loss - self.config.min_delta
            if val_loss < improvement_threshold:
                # Additional check: if we just resumed and this is the first epoch after resume,
                # and the improvement is very small (within 2*min_delta), it might be due to
                # floating-point precision or data order differences. Only save if improvement is significant.
                is_first_epoch_after_resume = (epoch == start_epoch and resumed_from_checkpoint)
                improvement_magnitude = self.best_val_loss - val_loss
                
                if is_first_epoch_after_resume and improvement_magnitude < 2 * self.config.min_delta:
                    # Very small improvement on first epoch after resume - likely due to precision/data order
                    # Don't save new best, but still update best_val_loss to the current (slightly better) value
                    print(f"   ⚠️  Small improvement ({improvement_magnitude:.6f}) on first epoch after resume - likely due to precision/data order. Not saving new best checkpoint.")
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.best_epoch = epoch
                else:
                    # Significant improvement - save new best checkpoint
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.best_epoch = epoch
                    self.save_checkpoint(epoch, is_best=True, experiment_results_dir=experiment_results_dir)
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch+1} (patience: {self.config.patience})")
                # Save last checkpoint before breaking
                self.save_checkpoint(epoch, is_last=True, experiment_results_dir=experiment_results_dir)
                break
        
        # Save last checkpoint at the end of training (if we didn't already save it due to early stopping)
        if self.patience_counter < self.config.patience:
            # Training completed normally (reached max epochs)
            self.save_checkpoint(epoch, is_last=True, experiment_results_dir=experiment_results_dir)
        
        total_time = time.time() - start_time
        
        # Save final metrics
        # Metrics saving now handled by ExperimentLogger
        
        # Return training results
        results = {
            'experiment_name': self.experiment_name,
            'model_info': self._get_underlying_model().get_model_info(),
            'config': self.config.to_dict(),
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'total_time': total_time,
            'training_completed': True
        }
        
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch+1}")
        
        return results

def load_checkpoint(checkpoint_path: str, model: BaseEEGCNN, 
                   device: torch.device = None, strict: bool = True) -> Dict[str, Any]:
    """
    Load model from checkpoint.
    Handles DataParallel checkpoints by stripping 'module.' prefix if needed.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load weights into
        device: Device to load model on
        strict: If True, requires exact match of state_dict keys. If False, allows partial loading.
                Default True for backward compatibility, but will raise error on mismatch.
    
    Returns:
        Checkpoint information dictionary
        
    Raises:
        RuntimeError: If strict=True and state_dict doesn't match model architecture
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # Handle torch.compile and DataParallel wrappers in checkpoint
    # Checkpoint may have keys like: "_orig_mod.module.conv_layers.0.weight"
    # We need to strip both "_orig_mod." and "module." prefixes to get to the base model
    new_state_dict = {}
    for k, v in state_dict.items():
        # Strip torch.compile wrapper prefix (_orig_mod.)
        if k.startswith('_orig_mod.'):
            k = k[10:]  # Remove '_orig_mod.' prefix (10 characters)
        
        # Strip DataParallel wrapper prefix (module.)
        if k.startswith('module.'):
            k = k[7:]  # Remove 'module.' prefix (7 characters)
        
        new_state_dict[k] = v
    
    state_dict = new_state_dict
    
    # Handle DataParallel wrapper mismatch if model is currently wrapped
    model_is_wrapped = isinstance(model, nn.DataParallel)
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    
    if has_module_prefix and not model_is_wrapped:
        # State dict has 'module.' prefix but model is not wrapped - strip prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix (7 characters)
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    elif not has_module_prefix and model_is_wrapped:
        # State dict doesn't have 'module.' prefix but model is wrapped - add prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[f'module.{k}'] = v
        state_dict = new_state_dict
    
    # Try to load state_dict - handle architecture mismatches gracefully
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        if strict:
            # In strict mode, raise the error with helpful message
            error_msg = (
                f"Failed to load checkpoint due to architecture mismatch. "
                f"This usually happens when the model architecture has changed. "
                f"Original error: {e}\n"
                f"To start fresh training, delete or rename the checkpoint file: {checkpoint_path}"
            )
            raise RuntimeError(error_msg) from e
        else:
            # In non-strict mode, log warnings but continue
            missing_keys = []
            unexpected_keys = []
            error_str = str(e)
            if "Missing key(s)" in error_str:
                # Extract missing keys from error message
                import re
                missing_match = re.search(r'Missing key\(s\) in state_dict: (.+?)(?:\.|size mismatch)', error_str)
                if missing_match:
                    missing_keys = [k.strip() for k in missing_match.group(1).split(',')]
            if "Unexpected key(s)" in error_str:
                unexpected_match = re.search(r'Unexpected key\(s\) in state_dict: (.+?)(?:\.|size mismatch)', error_str)
                if unexpected_match:
                    unexpected_keys = [k.strip() for k in unexpected_match.group(1).split(',')]
            
            print(f"⚠️  Warning: Partial checkpoint loading (architecture mismatch detected)")
            if missing_keys:
                print(f"   Missing keys (will use random initialization): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"   Unexpected keys (ignored): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
            print(f"   Continuing with partial weights...")
            
            # Try partial loading
            model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    
    return checkpoint
