"""
Model Factory for EEG Classification Models
Provides a centralized way to create and configure models.
"""

from typing import Dict, Any, Type
from .base_model import BaseEEGCNN
from .model import EEGCNN, EEGGenderCNN, EEGAgeCNN, CombinedCNN, MultiOutputCNN, EEGAgeRegressionCNN

class ModelFactory:
    """
    Factory class for creating EEG classification models.
    Implements the Factory pattern for systematic model creation.
    """
    
    # Registry of available models
    _models: Dict[str, Type[BaseEEGCNN]] = {
        'eeg_cnn': EEGCNN,           # Unified model for any number of classes
        'gender_cnn': EEGGenderCNN,  # Backward compatibility (2 classes)
        'age_cnn': EEGAgeCNN,        # Backward compatibility (3 classes)
        'age_regression_cnn': EEGAgeRegressionCNN,  # Age regression (1 output)
        'combined_cnn': CombinedCNN, # Combined age+gender classification (6 classes)
        'multi_output_cnn': MultiOutputCNN, # Multi-output model (2 heads: gender + age)
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseEEGCNN:
        """
        Create a model instance by type.
        
        Args:
            model_type: Type of model to create 
                       ('eeg_cnn', 'gender_cnn', 'age_cnn', 'age_regression_cnn', 'combined_cnn', 'multi_output_cnn')
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in cls._models:
            available_models = ', '.join(cls._models.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")
        
        model_class = cls._models[model_type]
        return model_class(**kwargs)
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model types."""
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """
        Get information about a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary with model information
        """
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls._models[model_type]
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
