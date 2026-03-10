"""
Centralized Configuration for EEG Classification Framework
All configuration parameters are defined here for easy maintenance and consistency.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    hdf5_dir: str = "path/to/hdf5_dir"  # Changed from pickle_dir to hdf5_dir
    segment_length: int = None  # 200 for 1s, 800 for 4s
    
    # Data loading parameters
    batch_size: int = 128
    num_workers: int = 4
    random_seed: int = 42
    task_type: str = "both"  # "active", "passive", or "both"
    
    # Train/val/test splits
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
            'hdf5_dir': self.hdf5_dir,
            'segment_length': self.segment_length,
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
    learning_rate: float = 0.001  # Default learning rate (will be adjusted for ArcFace if needed)
    patience: int = 10
    min_delta: float = 1e-4
    save_every: int = 5
    checkpoint_dir: str = 'checkpoints'
    target_key: str = 'gender'  # 'gender', 'age', 'combined', 'multi_output', or 'user_identification'
    prediction_type: str = 'classification'  # 'classification' or 'regression'

    # Evaluation: aggregate segment-level predictions to participant level (no retraining)
    # 'majority_vote' = classification: confidence-weighted majority (or mode with tie-break if no probs); regression: median
    # None = segment-level evaluation (default)
    aggregate_by_participant: Optional[str] = None  # None or 'majority_vote'

    # ArcFace-specific learning rate multiplier
    # ArcFace is more sensitive to learning rate than CrossEntropyLoss
    # DIAGNOSIS: LR of 0.00005 (0.5 multiplier) caused model collapse in epoch 18
    # Reduced to 0.3 to prevent collapse while still allowing learning
    # This results in effective LR of 0.00003 after warmup (0.0001 * 0.3)
    # Lower LR helps prevent collapse but may slow convergence
    arcface_lr_multiplier: float = 0.3  # Multiply base LR by this for ArcFace (default: 0.3 for very large-scale classification)
    
    # Gradient clipping for training stability
    # Prevents exploding gradients, especially important for large-scale classification
    # Set to None to disable gradient clipping
    # For very large classification (3000+ classes), use more aggressive clipping
    # Further reduced from 0.3 to 0.2 for maximum stability
    max_grad_norm: float = 0.2  # Maximum gradient norm for clipping (default: 0.2 for very large-scale classification)
    
    # Learning rate warmup for stable training
    # Gradually increases learning rate from 0 to target LR over warmup_epochs
    # This prevents large gradient updates in early epochs that can cause collapse
    # DIAGNOSIS: 8 epochs was too short - increased to 10 for more gradual ramp-up
    # With deeper model and 3145 classes, we need more gradual warmup to prevent collapse
    warmup_epochs: int = 10  # Number of epochs for warmup (default: 10 for very large-scale classification)
    
    # Optimizer hyperparameters
    weight_decay: float = 0.0  # L2 regularization (default: 0.0, set to 1e-4 for user identification)
    
    # Learning rate scheduler hyperparameters
    scheduler_factor: float = 0.5  # Factor by which learning rate is reduced
    scheduler_patience: int = None  # Patience for scheduler (None = use default: 10, or 3 for user identification)
    
    # Loss function hyperparameters
    # Label smoothing for CrossEntropyLoss (used for large classification tasks)
    label_smoothing_large: float = 0.1  # For num_classes > 100
    label_smoothing_very_large: float = 0.05  # For num_classes > 2000

    # Whether to use stratified train batches (each batch has balanced class counts). Only for gender/age classification.
    # None = use default (True for gender/age). Set False to disable when building the sample list is slow (e.g. large 4s HDF5).
    use_stratified_train_batches: Optional[bool] = None  # None = default (True for gender/age), False = plain shuffle

    # How to balance classes for gender/age classification training. Options:
    # - None: no balancing (iterable dataset); class_weight in loss is set from data if not provided.
    # - 'stratified': stratified batch sampling (each batch has equal counts per class); class_weight can still be used.
    # - 'oversample': sampling with replacement (minority classes oversampled); class_weight is NOT set (balance via data).
    balance_method: Optional[str] = None  # None | 'stratified' | 'oversample'

    # Class weights for imbalanced classification (e.g. gender: Female=0, Male=1). Ignored when balance_method == 'oversample'.
    # Example: [1.0, 2.0] upweights Male so the model is penalized more for missing Male.
    class_weight: Optional[List[float]] = None  # None = no weighting; list length must match num_classes
    # When class_weight is computed from data: exponent applied to inverse-frequency weights (>1 upweights minorities more).
    class_weight_power: Optional[float] = None  # None = 1.0 (standard balanced); set e.g. 1.5 for stronger minority weighting.

    # ArcFace hyperparameters (for very large classification: num_classes > 2000)
    arcface_margin: float = 0.5  # Angular margin in radians (~28.6 degrees)
    # DIAGNOSIS: Scale of 128.0 was insufficient for 3145 classes
    # With loss decreasing but accuracy near zero, logits were too small, softmax too flat
    # Increased to 256.0 for better discrimination power (2x increase from 128.0)
    # Higher scale makes logits larger, softmax probabilities more peaked, better discrimination
    # This should help model make more confident predictions and improve accuracy
    arcface_scale: float = 256.0  # Feature scale parameter (increased for very large-scale classification)
    arcface_easy_margin: bool = False  # Whether to use easier margin computation
    
    # Numerical stability hyperparameters
    output_clamp_min: float = -50.0  # Minimum value for output clamping (prevents numerical overflow)
    output_clamp_max: float = 50.0  # Maximum value for output clamping (prevents numerical overflow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'save_every': self.save_every,
            'checkpoint_dir': self.checkpoint_dir,
            'target_key': self.target_key,
            'prediction_type': self.prediction_type,
            'aggregate_by_participant': self.aggregate_by_participant,
            'balance_method': self.balance_method,
            'weight_decay': self.weight_decay,
            'scheduler_factor': self.scheduler_factor,
            'scheduler_patience': self.scheduler_patience,
            'label_smoothing_large': self.label_smoothing_large,
            'label_smoothing_very_large': self.label_smoothing_very_large,
            'class_weight': self.class_weight,
            'class_weight_power': self.class_weight_power,
            'arcface_margin': self.arcface_margin,
            'arcface_scale': self.arcface_scale,
            'arcface_easy_margin': self.arcface_easy_margin,
            'arcface_lr_multiplier': self.arcface_lr_multiplier,
            'max_grad_norm': self.max_grad_norm,
            'output_clamp_min': self.output_clamp_min,
            'output_clamp_max': self.output_clamp_max
        }

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    model_type: str  # 'gender_cnn' or 'age_cnn'
    target_type: str  # 'gender' or 'age'
    description: Optional[str] = None  # Human-readable description of the experiment
    data_config: Optional[DataConfig] = None
    model_config: Optional[ModelConfig] = None
    training_config: Optional[TrainingConfig] = None
    # When set (e.g. for eval_only), load this path instead of results_dir/checkpoints/<name>_best.pth
    checkpoint_path: Optional[str] = None
    # When set, load from this directory (expects <name>_best.pth inside). Ignored if checkpoint_path is set.
    checkpoint_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.data_config is None:
            self.data_config = DataConfig()
        if self.model_config is None:
            self.model_config = ModelConfig()
        if self.training_config is None:
            self.training_config = TrainingConfig(target_key=self.target_type)
        
        # Set num_classes based on target_type and prediction_type
        if self.target_type == 'gender':
            self.model_config.num_classes = 2
        elif self.target_type == 'age':
            # Check if this is a regression task
            if hasattr(self.training_config, 'prediction_type') and self.training_config.prediction_type == 'regression':
                self.model_config.num_classes = 1  # Regression outputs 1 value
            else:
                self.model_config.num_classes = 3  # Classification has 3 classes
        elif self.target_type == 'combined':
            self.model_config.num_classes = 6  # 2 genders × 3 age groups = 6 classes
        elif self.target_type == 'multi_output':
            self.model_config.num_classes = 2  # Will be overridden by multi-output heads
        elif self.target_type == 'user_identification':
            # num_classes will be set dynamically based on number of participants
            # This is handled in experiment.py when loading the transform
            pass

@dataclass
class SystemConfig:
    """System-wide configuration."""
    results_dir: str = 'experiment_results'
    reports_dir: str = 'reports'
    evaluation_dir: str = 'evaluation_results'
    log_dir: str = 'logs'
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    verbose: bool = True
    num_gpus: int = 1  # Number of GPUs to use for DataParallel (default: 1, will auto-detect up to 4 if available)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'results_dir': self.results_dir,
            'reports_dir': self.reports_dir,
            'evaluation_dir': self.evaluation_dir,
            'log_dir': self.log_dir,
            'device': self.device,
            'verbose': self.verbose,
            'num_gpus': self.num_gpus
        }
