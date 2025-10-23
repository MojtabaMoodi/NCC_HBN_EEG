"""
Models package for EEG Classification Framework
Contains base models, specific implementations, and model factory.
"""

from .base_model import BaseEEGCNN
from .model import EEGCNN, EEGGenderCNN, EEGAgeCNN
from .model_factory import ModelFactory, create_model_from_config

__all__ = [
    'BaseEEGCNN',
    'EEGCNN',           # Unified model for any number of classes
    'EEGGenderCNN',     # Backward compatibility alias (2 classes)
    'EEGAgeCNN',        # Backward compatibility alias (3 classes)
    'ModelFactory',
    'create_model_from_config'
]
