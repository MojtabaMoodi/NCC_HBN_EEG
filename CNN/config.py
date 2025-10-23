"""
Centralized Configuration for EEG Classification Framework
All configuration parameters are defined here for easy maintenance and consistency.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    pickle_dir: str = "/home/mojtabam/projects/def-aghodsib/mojtabam/EEG/data_processing/processed_eeg_data"
    batch_size: int = 128  # Increased from 32 for better GPU utilization
    num_workers: int = 4
    random_seed: int = 42
    task_type: str = "both"  # "active", "passive", or "both"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Cross-validation support
    use_cross_validation: bool = False
    n_folds: int = 5
    cv_strategy: str = "stratified"  # "stratified", "kfold", "group"
    
    # Cross-task evaluation support
    train_task_type: str = None  # "active" or "passive" for cross-task experiments
    val_test_task_type: str = None  # "active" or "passive" for cross-task experiments
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'pickle_dir': self.pickle_dir,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'random_seed': self.random_seed,
            'task_type': self.task_type,
            'train_split': self.train_split,
            'val_split': self.val_split,
            'test_split': self.test_split,
            'use_cross_validation': self.use_cross_validation,
            'n_folds': self.n_folds,
            'cv_strategy': self.cv_strategy,
            'train_task_type': self.train_task_type,
            'val_test_task_type': self.val_test_task_type
        }

@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    num_channels: int = 60
    dropout_rate: float = 0.5
    use_layer_norm: bool = True
    num_classes: int = 2  # Will be overridden based on target_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'num_channels': self.num_channels,
            'dropout_rate': self.dropout_rate,
            'use_layer_norm': self.use_layer_norm,
            'num_classes': self.num_classes
        }

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    epochs: int = 50
    learning_rate: float = 0.001
    patience: int = 10
    min_delta: float = 1e-4
    save_every: int = 5
    checkpoint_dir: str = 'checkpoints'
    target_key: str = 'gender'  # 'gender' or 'age'

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'save_every': self.save_every,
            'checkpoint_dir': self.checkpoint_dir,
            'target_key': self.target_key
        }

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    model_type: str  # 'gender_cnn' or 'age_cnn'
    target_type: str  # 'gender' or 'age'
    data_config: Optional[DataConfig] = None
    model_config: Optional[ModelConfig] = None
    training_config: Optional[TrainingConfig] = None
    
    def __post_init__(self):
        if self.data_config is None:
            self.data_config = DataConfig()
        if self.model_config is None:
            self.model_config = ModelConfig()
        if self.training_config is None:
            self.training_config = TrainingConfig(target_key=self.target_type)
        
        # Set num_classes based on target_type
        if self.target_type == 'gender':
            self.model_config.num_classes = 2
        elif self.target_type == 'age':
            self.model_config.num_classes = 3

@dataclass
class SystemConfig:
    """System-wide configuration."""
    results_dir: str = 'experiment_results'
    reports_dir: str = 'reports'
    evaluation_dir: str = 'evaluation_results'
    log_dir: str = 'logs'
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    verbose: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'results_dir': self.results_dir,
            'reports_dir': self.reports_dir,
            'evaluation_dir': self.evaluation_dir,
            'log_dir': self.log_dir,
            'device': self.device,
            'verbose': self.verbose
        }
