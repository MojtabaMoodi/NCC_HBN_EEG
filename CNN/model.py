import torch
import torch.nn as nn
from base_model import BaseEEGCNN

class EEGGenderCNN(BaseEEGCNN):
    """
    CNN model for EEG gender classification.
    Inherits from BaseEEGCNN for common functionality.
    """
    
    def __init__(self, num_channels=64, num_classes=2, dropout_rate=0.5, use_layer_norm=True):
        super(EEGGenderCNN, self).__init__(
            num_channels=num_channels, 
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm
        )
    
    def _build_conv_layers(self) -> nn.ModuleList:
        """Build the convolutional layers for gender classification."""
        return nn.ModuleList([
            nn.Conv2d(1, 16, kernel_size=(self.num_channels, 3)),
            nn.Conv2d(16, 32, kernel_size=(1, 3)),
            nn.Conv2d(32, 64, kernel_size=(1, 3)),
            nn.Conv2d(64, 128, kernel_size=(1, 3)),
            nn.Conv2d(128, 128, kernel_size=(1, 3)),
            nn.Conv2d(128, 256, kernel_size=(1, 3)),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.Conv2d(256, 256, kernel_size=(1, 3))
        ])


class EEGAgeCNN(BaseEEGCNN):
    """
    CNN model for EEG age classification.
    Uses 3 classes: <8.5, 8.5-12.5, >12.5 years.
    """
    
    def __init__(self, num_channels=64, num_classes=3, dropout_rate=0.5, use_layer_norm=True):
        super(EEGAgeCNN, self).__init__(
            num_channels=num_channels, 
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm
        )
    
    def _build_conv_layers(self) -> nn.ModuleList:
        """Build the convolutional layers for age classification."""
        return nn.ModuleList([
            nn.Conv2d(1, 16, kernel_size=(self.num_channels, 3)),
            nn.Conv2d(16, 32, kernel_size=(1, 3)),
            nn.Conv2d(32, 64, kernel_size=(1, 3)),
            nn.Conv2d(64, 128, kernel_size=(1, 3)),
            nn.Conv2d(128, 128, kernel_size=(1, 3)),
            nn.Conv2d(128, 256, kernel_size=(1, 3)),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.Conv2d(256, 256, kernel_size=(1, 3))
        ])