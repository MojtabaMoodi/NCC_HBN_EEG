"""
Systematic Training Module for EEG Classification Models
Provides model-agnostic training functionality with comprehensive logging.
"""

import os
import time
import json
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
from utils import safe_json_dump, convert_numpy_types
from config import TrainingConfig


class EEGTrainer:
    """
    Systematic trainer for EEG classification models.
    Provides model-agnostic training with comprehensive logging and early stopping.
    """
    
    def __init__(self, model: BaseEEGCNN, config: TrainingConfig, 
                 experiment_name: str = None):
        self.model = model
        self.config = config
        self.experiment_name = experiment_name or f"{model.__class__.__name__}_{int(time.time())}"
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Setup model
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(model)
        self.model.to(self.device)
        
        # Setup training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            inputs = batch['eeg_data'].to(self.device)
            labels = batch[self.config.target_key].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['eeg_data'].to(self.device)
                labels = batch[self.config.target_key].to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
            'model_info': self.model.get_model_info()
        }
        
        # Create model-specific directory
        model_name = self.model.__class__.__name__
        model_checkpoint_dir = os.path.join(self.config.checkpoint_dir, model_name)
        os.makedirs(model_checkpoint_dir, exist_ok=True)
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            model_checkpoint_dir, 
            f'{self.experiment_name}_epoch_{epoch+1}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(
                model_checkpoint_dir, 
                f'{self.experiment_name}_best.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"New best model saved at epoch {epoch+1} in {model_checkpoint_dir}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, progress_callback=None) -> Dict[str, Any]:
        """
        Train the model with early stopping and comprehensive logging.
        
        Returns:
            Dictionary with training results and metrics
        """
        print(f"Starting training for {self.experiment_name}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Target: {self.config.target_key}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
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
                print(f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                    f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")
            
            # Check for improvement
            if val_loss < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch)
            
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
            'model_info': self.model.get_model_info(),
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
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load weights into
        device: Device to load model on
        
    Returns:
        Checkpoint information dictionary
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return checkpoint
