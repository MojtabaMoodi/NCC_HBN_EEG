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

from models import BaseEEGCNN
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/data_processing')
from utils import safe_json_dump, convert_numpy_types
from config import TrainingConfig


class EEGTrainer:
    """
    Systematic trainer for EEG classification models.
    Provides model-agnostic training with comprehensive logging and early stopping.
    """
    
    def __init__(self, model: BaseEEGCNN, config: TrainingConfig, 
                 experiment_name: str = None, num_gpus: int = 2,
                 age_min: float = None, age_max: float = None):
        self.model = model
        self.config = config
        self.experiment_name = experiment_name or f"{model.__class__.__name__}_{int(time.time())}"
        # Store age range for displaying MAE in years (for regression tasks)
        self.age_min = age_min
        self.age_max = age_max
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Setup model - use specified number of GPUs for DataParallel
        if torch.cuda.device_count() > 1:
            num_gpus_to_use = min(num_gpus, torch.cuda.device_count())
            if num_gpus_to_use < num_gpus:
                print(f"Warning: Requested {num_gpus} GPUs but only {torch.cuda.device_count()} available. Using {num_gpus_to_use} GPUs.")
            else:
                print(f"Using {num_gpus_to_use} GPUs (out of {torch.cuda.device_count()} available)")
            self.model = nn.DataParallel(model, device_ids=list(range(num_gpus_to_use)))
        self.model.to(self.device)
        
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
        # Use MSE loss for regression, CrossEntropyLoss for classification
        self.is_regression = hasattr(config, 'prediction_type') and config.prediction_type == 'regression'
        if self.is_regression:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Use fused Adam optimizer if available (faster on modern GPUs)
        try:
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, fused=True)
            print("✅ Using fused Adam optimizer")
        except TypeError:
            # Fused optimizer not available, fall back to regular Adam
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # Enable mixed precision training (AMP) for 1.5-2x speedup
        self.use_amp = True
        # Use new torch.amp API (PyTorch 2.0+)
        self.scaler = torch.amp.GradScaler('cuda')
        print("✅ Mixed precision training (AMP) enabled")
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0
    
    def _get_underlying_model(self):
        """Get the underlying model, unwrapping DataParallel and torch.compile wrappers if needed."""
        model = self.model
        
        # Recursively unwrap until we get to the base model
        while True:
            # Unwrap torch.compile wrapper (has _orig_mod attribute)
            if hasattr(model, '_orig_mod'):
                model = model._orig_mod
                continue
            
            # Unwrap DataParallel wrapper
            if isinstance(model, nn.DataParallel):
                model = model.module
                continue
            
            # No more wrappers to unwrap
            break
        
        return model
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
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
                labels = batch[self.config.target_key].to(self.device, non_blocking=True)
                # For regression, ensure labels are float and have correct shape
                if self.is_regression:
                    if labels.dtype != torch.float32:
                        labels = labels.float()
                    # Reshape if needed: (batch_size,) -> (batch_size, 1)
                    if labels.dim() == 1:
                        labels = labels.unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training (AMP)
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(inputs)
                    
                    # Handle multi-output models
                    if isinstance(outputs, dict):
                        # Multi-output model (e.g., MultiOutputCNN)
                        gender_loss = self.criterion(outputs['gender'], labels['gender'])
                        age_loss = self.criterion(outputs['age'], labels['age'])
                        loss = gender_loss + age_loss  # Combined loss
                    else:
                        # Single-output model (e.g., EEGCNN, CombinedCNN, EEGAgeRegressionCNN)
                        loss = self.criterion(outputs, labels)
                
                # Calculate metrics outside autocast
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
                        correct += (predicted == labels).sum().item()
                
                # Scale loss and backward pass for mixed precision
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                outputs = self.model(inputs)
                
                # Handle multi-output models
                if isinstance(outputs, dict):
                    # Multi-output model (e.g., MultiOutputCNN)
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
                    loss = self.criterion(outputs, labels)
                    if self.is_regression:
                        # For regression: accumulate sum of absolute errors
                        # We'll divide by total later to get overall MAE
                        sum_absolute_errors = torch.abs(outputs - labels).sum().item()
                        correct += sum_absolute_errors
                    else:
                        # For classification: calculate accuracy
                        predicted = torch.argmax(outputs, dim=1)
                        correct += (predicted == labels).sum().item()
                
                loss.backward()
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
                    labels = batch[self.config.target_key].to(self.device, non_blocking=True)
                    # For regression, ensure labels are float and have correct shape
                    if self.is_regression:
                        if labels.dtype != torch.float32:
                            labels = labels.float()
                        # Reshape if needed: (batch_size,) -> (batch_size, 1)
                        if labels.dim() == 1:
                            labels = labels.unsqueeze(1)
                
                # Use mixed precision for validation too (faster, minimal accuracy impact)
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)
                
                # Handle multi-output models
                if isinstance(outputs, dict):
                    # Multi-output model (e.g., MultiOutputCNN)
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
                    loss = self.criterion(outputs, labels)
                    if self.is_regression:
                        # For regression: accumulate sum of absolute errors
                        # We'll divide by total later to get overall MAE
                        sum_absolute_errors = torch.abs(outputs - labels).sum().item()
                        correct += sum_absolute_errors
                    else:
                        # For classification: calculate accuracy
                        predicted = torch.argmax(outputs, dim=1)
                        correct += (predicted == labels).sum().item()
                
                total_loss += loss.item()
                total += labels.size(0) if not isinstance(labels, dict) else labels['gender'].size(0)
        
        # Use num_batches instead of len(val_loader) since IterableDataset doesn't support __len__()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Calculate metric based on task type
        # For regression: correct = sum(|outputs - labels|) across all batches, so correct/total = overall MAE
        # For classification: correct = sum(correct_predictions), so correct/total = accuracy
        metric = correct / total if total > 0 else 0.0
        return avg_loss, metric
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, experiment_results_dir: str = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
            'model_info': self._get_underlying_model().get_model_info()
        }
        
        # Use experiment-specific directory if provided, otherwise fall back to global
        if experiment_results_dir:
            checkpoint_dir = os.path.join(experiment_results_dir, 'checkpoints')
        else:
            # Fallback to global checkpoint directory
            model_name = self._get_underlying_model().__class__.__name__
            checkpoint_dir = os.path.join(self.config.checkpoint_dir, model_name)
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir, 
            f'{self.experiment_name}_epoch_{epoch+1}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(
                checkpoint_dir, 
                f'{self.experiment_name}_best.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"New best model saved at epoch {epoch+1} in {checkpoint_dir}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, progress_callback=None, experiment_results_dir: str = None) -> Dict[str, Any]:
        """
        Train the model with early stopping and comprehensive logging.
        
        Returns:
            Dictionary with training results and metrics
        """
        print(f"Starting training for {self.experiment_name}")
        print(f"Model: {self._get_underlying_model().__class__.__name__}")
        print(f"Target: {self.config.target_key}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("-" * 60)
        
        start_time = time.time()
        
        # Log that we're about to start loading first batch
        print(f"⏳ Loading first batch (batch_size={train_loader.batch_size}, num_workers={train_loader.num_workers})...")
        # Note: IterableDataset doesn't support __len__(), so we can't print dataset size
        sys.stdout.flush()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            print(f"\n🔄 Starting epoch {epoch+1}/{self.config.epochs}")
            sys.stdout.flush()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            
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
            if val_loss < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True, experiment_results_dir=experiment_results_dir)
            else:
                self.patience_counter += 1
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch, experiment_results_dir=experiment_results_dir)
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch+1} (patience: {self.config.patience})")
                break
        
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
                   device: torch.device = None) -> Dict[str, Any]:
    """
    Load model from checkpoint.
    Handles DataParallel checkpoints by stripping 'module.' prefix if needed.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load weights into
        device: Device to load model on
    
    Returns:
        Checkpoint information dictionary
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
    
    model.load_state_dict(state_dict)
    model.to(device)
    
    return checkpoint
