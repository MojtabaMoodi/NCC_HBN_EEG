"""
ResNet Models for EEG Classification
Implements ResNet-like architectures adapted for EEG data.
"""

from .resnet_model import (
    BasicBlock,
    BottleneckBlock,
    EEGResNet,
    EEGResNet18,
    EEGResNet34,
    EEGResNet50,
    EEGGenderResNet,
    EEGGenderResNet34,
    EEGGenderResNet50,
    EEGAgeResNet,
    EEGAgeResNet34,
    EEGAgeResNet50,
    CombinedResNet,
    MultiOutputResNet,
    EEGAgeRegressionResNet
)

__all__ = [
    'BasicBlock',
    'BottleneckBlock',
    'EEGResNet',
    'EEGResNet18',
    'EEGResNet34',
    'EEGResNet50',
    'EEGGenderResNet',
    'EEGGenderResNet34',
    'EEGGenderResNet50',
    'EEGAgeResNet',
    'EEGAgeResNet34',
    'EEGAgeResNet50',
    'CombinedResNet',
    'MultiOutputResNet',
    'EEGAgeRegressionResNet'
]

# Backward compatibility alias
ResNetBlock = BasicBlock

