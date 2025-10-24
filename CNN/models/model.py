import torch
import torch.nn as nn
from .base_model import BaseEEGCNN

class EEGCNN(BaseEEGCNN):
    """
    Unified CNN model for EEG classification tasks.
    Can handle different numbers of classes (binary, multi-class, etc.).
    
    Examples:
        - Gender classification: num_classes=2 (Female/Male)
        - Age classification: num_classes=3 (<8.5, 8.5-12.5, >12.5 years)
        - Custom classification: num_classes=N (any number of classes)
    """
    
    def __init__(self, num_channels=64, num_classes=2, dropout_rate=0.5, use_layer_norm=True):
        super(EEGCNN, self).__init__(
            num_channels=num_channels, 
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm
        )
    
    def _build_conv_layers(self) -> nn.ModuleList:
        """Build the convolutional layers for EEG classification."""
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


# Backward compatibility aliases
class EEGGenderCNN(EEGCNN):
    """
    CNN model for EEG gender classification (binary).
    Backward compatibility alias for EEGCNN with num_classes=2.
    """
    def __init__(self, num_channels=64, num_classes=2, dropout_rate=0.5, use_layer_norm=True):
        super(EEGGenderCNN, self).__init__(
            num_channels=num_channels, 
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm
        )


class EEGAgeCNN(EEGCNN):
    """
    CNN model for EEG age classification (3-class).
    Backward compatibility alias for EEGCNN with num_classes=3.
    """
    def __init__(self, num_channels=64, num_classes=3, dropout_rate=0.5, use_layer_norm=True):
        super(EEGAgeCNN, self).__init__(
            num_channels=num_channels, 
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm
        )

class CombinedCNN(EEGCNN):
    """
    CNN model for combined age+gender classification (6 classes).
    """
    def __init__(self, num_channels=64, num_classes=6, dropout_rate=0.5, use_layer_norm=True):
        super(CombinedCNN, self).__init__(
            num_channels=num_channels, 
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm
        )