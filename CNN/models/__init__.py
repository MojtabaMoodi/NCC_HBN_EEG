"""
Models package for EEG Classification Framework
Contains base models, specific implementations, and model factory.
"""

from .base_model import BaseEEGCNN
from .model import EEGGenderCNN, EEGAgeCNN
from .model_factory import ModelFactory, create_model_from_config

__all__ = [
    'BaseEEGCNN',
    'EEGGenderCNN', 
    'EEGAgeCNN',
    'ModelFactory',
    'create_model_from_config'
]
