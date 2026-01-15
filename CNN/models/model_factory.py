"""
Model Factory for EEG Classification Models
Provides a centralized way to create and configure models.
"""

from typing import Dict, Any, Type
from .base_model import BaseEEGCNN
from .model import EEGCNN, EEGGenderCNN, EEGAgeCNN, CombinedCNN, MultiOutputCNN, EEGAgeRegressionCNN, EEGUserIdentificationCNN

# Import LaBraM wrapper (optional - only if LaBraM is available)
# Import from user_identification directory
try:
    import sys
    import importlib.util
    from pathlib import Path
    _user_id_dir = Path(__file__).parent.parent.parent / "user_identification"
    _labram_model_path = _user_id_dir / "labram_model.py"
    
    if _labram_model_path.exists():
        # Use importlib to explicitly load the module
        spec = importlib.util.spec_from_file_location("labram_model", _labram_model_path)
        labram_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(labram_model)
        LaBraMUserIdentificationWrapper = labram_model.LaBraMUserIdentificationWrapper
        LABRAM_AVAILABLE = True
    else:
        # Fallback: try direct import if path is set
        if str(_user_id_dir) not in sys.path:
            sys.path.insert(0, str(_user_id_dir))
        from labram_model import LaBraMUserIdentificationWrapper  # type: ignore
        LABRAM_AVAILABLE = True
except ImportError:
    LaBraMUserIdentificationWrapper = None
    LABRAM_AVAILABLE = False

# Import ResNet models
# Use absolute import from CNN directory
# ResNet module is required - raise error if not available
import sys
import os
# Add CNN directory to path if not already there
cnn_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if cnn_dir not in sys.path:
    sys.path.insert(0, cnn_dir)

from resnet.resnet_model import (
    EEGResNet, EEGResNet18, EEGResNet34, EEGResNet50,
    EEGGenderResNet, EEGGenderResNet34, EEGGenderResNet50,
    EEGAgeResNet, EEGAgeResNet34, EEGAgeResNet50,
    CombinedResNet, MultiOutputResNet, EEGAgeRegressionResNet
)

class ModelFactory:
    """
    Factory class for creating EEG classification models.
    Implements the Factory pattern for systematic model creation.
    """
    
    # Registry of available models
    _models: Dict[str, Type[BaseEEGCNN]] = {
        # Standard CNN models
        'eeg_cnn': EEGCNN,           # Unified model for any number of classes
        'gender_cnn': EEGGenderCNN,  # Backward compatibility (2 classes)
        'age_cnn': EEGAgeCNN,        # Backward compatibility (3 classes)
        'age_regression_cnn': EEGAgeRegressionCNN,  # Age regression (1 output)
        'combined_cnn': CombinedCNN, # Combined age+gender classification (6 classes)
        'multi_output_cnn': MultiOutputCNN, # Multi-output model (2 heads: gender + age)
        'user_identification_cnn': EEGUserIdentificationCNN,  # User identification (N classes = number of participants)
    }
    
    # Add LaBraM models if available
    if LABRAM_AVAILABLE and LaBraMUserIdentificationWrapper is not None:
        _models['user_identification_labram'] = LaBraMUserIdentificationWrapper  # LaBraM for user identification
    
    @classmethod
    def _get_models_dict(cls) -> Dict[str, Type[BaseEEGCNN]]:
        """Get models dictionary, including ResNet models."""
        models = cls._models.copy()
        
        # Add ResNet models (required - import error will be raised if not available)
        models.update({
            'resnet': EEGResNet,         # Base ResNet model (configurable)
            'resnet18': EEGResNet18,     # ResNet-18 architecture (generic)
            'resnet34': EEGResNet34,     # ResNet-34 architecture (generic)
            'resnet50': EEGResNet50,     # ResNet-50 architecture (generic)
            'gender_resnet': EEGGenderResNet,  # ResNet18 for gender classification (2 classes)
            'gender_resnet34': EEGGenderResNet34,  # ResNet34 for gender classification (2 classes)
            'gender_resnet50': EEGGenderResNet50,  # ResNet50 for gender classification (2 classes)
            'age_resnet': EEGAgeResNet,  # ResNet18 for age classification (3 classes)
            'age_resnet34': EEGAgeResNet34,  # ResNet34 for age classification (3 classes)
            'age_resnet50': EEGAgeResNet50,  # ResNet50 for age classification (3 classes)
            'age_regression_resnet': EEGAgeRegressionResNet,  # ResNet18 for age regression (1 output)
            'combined_resnet': CombinedResNet,  # ResNet18 for combined classification (6 classes)
            'multi_output_resnet': MultiOutputResNet,  # ResNet18 multi-output (2 heads: gender + age)
        })
        
        return models
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseEEGCNN:
        """
        Create a model instance by type.
        
        Args:
            model_type: Type of model to create 
                       Standard CNN: 'eeg_cnn', 'gender_cnn', 'age_cnn', 'age_regression_cnn', 'combined_cnn', 'multi_output_cnn'
                       ResNet: 'resnet', 'resnet18', 'resnet34', 'resnet50', 'gender_resnet', 'age_resnet', 
                               'age_regression_resnet', 'combined_resnet', 'multi_output_resnet'
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model_type is not supported
        """
        models = cls._get_models_dict()
        if model_type not in models:
            available_models = ', '.join(models.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")
        
        model_class = models[model_type]
        return model_class(**kwargs)
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model types."""
        return list(cls._get_models_dict().keys())
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """
        Get information about a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary with model information
        """
        models = cls._get_models_dict()
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = models[model_type]
        return {
            'name': model_class.__name__,
            'module': model_class.__module__,
            'description': model_class.__doc__ or 'No description available'
        }
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseEEGCNN]):
        """
        Register a new model type.
        
        Args:
            name: Name to register the model under
            model_class: Model class that inherits from BaseEEGCNN
        """
        if not issubclass(model_class, BaseEEGCNN):
            raise ValueError("Model class must inherit from BaseEEGCNN")
        
        cls._models[name] = model_class
        # Also update the dynamic dict (ResNet models are always included)
        cls._get_models_dict()[name] = model_class


# Predefined model configurations for common use cases
MODEL_CONFIGS = {
    'gender_classification': {
        'model_type': 'gender_cnn',
        'num_classes': 2,
        'dropout_rate': 0.5,
        'use_layer_norm': True
    },
    'age_classification': {
        'model_type': 'age_cnn',
        'num_classes': 3,
        'dropout_rate': 0.5,
        'use_layer_norm': True
    }
}

def create_model_from_config(config_name: str, num_channels: int = 60, **overrides) -> BaseEEGCNN:
    """
    Create a model from a predefined configuration.
    
    Args:
        config_name: Name of the configuration ('gender_classification', 'age_classification')
        num_channels: Number of EEG channels
        **overrides: Override any configuration parameters
        
    Returns:
        Configured model instance
    """
    if config_name not in MODEL_CONFIGS:
        available_configs = ', '.join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown config: {config_name}. Available: {available_configs}")
    
    config = MODEL_CONFIGS[config_name].copy()
    config.update(overrides)
    config['num_channels'] = num_channels
    
    return ModelFactory.create_model(**config)
