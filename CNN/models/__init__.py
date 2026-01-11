"""
Models package for EEG Classification Framework
Contains base models, specific implementations, and model factory.
"""

from .base_model import BaseEEGCNN
from .model import EEGCNN, EEGGenderCNN, EEGAgeCNN, CombinedCNN, MultiOutputCNN, EEGUserIdentificationCNN
from .model_factory import ModelFactory, create_model_from_config
from .arcface_loss import ArcFaceLoss

__all__ = [
    'BaseEEGCNN',
    'EEGCNN',           # Unified model for any number of classes
    'EEGGenderCNN',     # Backward compatibility alias (2 classes)
    'EEGAgeCNN',        # Backward compatibility alias (3 classes)
    'CombinedCNN',      # Combined age+gender classification (6 classes)
    'MultiOutputCNN',   # Multi-output model (2 heads: gender + age)
    'EEGUserIdentificationCNN',  # User identification (N classes = number of participants)
    'ModelFactory',
    'create_model_from_config'
]
