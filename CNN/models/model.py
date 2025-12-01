import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
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


class MultiOutputCNN(BaseEEGCNN):
    """
    Multi-output CNN model for simultaneous gender and age prediction.
    Has two separate output heads: one for gender (2 classes) and one for age (3 classes).
    """
    
    def __init__(self, num_channels=64, num_classes=2, dropout_rate=0.5, use_layer_norm=True):
        super(MultiOutputCNN, self).__init__(
            num_channels=num_channels,
            num_classes=2,  # Will be overridden by separate heads
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm
        )
        
        # Override the classifier with two separate heads
        # The base model uses 64 features after fc1 layer
        self.gender_classifier = nn.Linear(64, 2)  # 2 classes for gender
        self.age_classifier = nn.Linear(64, 3)     # 3 classes for age
        
        # Remove the original classifier
        del self.fc2
    
    def forward(self, x):
        """
        Forward pass with two separate outputs.
        
        Args:
            x: Input tensor of shape (batch_size, 1, num_channels, sequence_length)
            
        Returns:
            Dictionary with 'gender' and 'age' predictions
        """
        # Preprocess input
        x = self._preprocess_input(x)
        
        # Apply convolutional layers
        x = self._apply_conv_layers(x)
        
        # Global average pooling across spatial dimensions
        x = F.adaptive_avg_pool2d(x, (1, 1))  # (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)             # (batch, 256)
        
        # Apply first fully connected layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Separate predictions
        gender_output = self.gender_classifier(x)
        age_output = self.age_classifier(x)
        
        return {
            'gender': gender_output,
            'age': age_output
        }
    
    
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


class EEGAgeRegressionCNN(BaseEEGCNN):
    """
    CNN model for EEG age regression.
    Outputs a single continuous value (normalized age in [0, 1] range).
    """
    
    def __init__(self, num_channels=64, num_classes=1, dropout_rate=0.5, use_layer_norm=True):
        # For regression, num_classes is always 1 (single continuous output)
        # Accept num_classes as parameter for factory compatibility, but always use 1
        super(EEGAgeRegressionCNN, self).__init__(
            num_channels=num_channels,
            num_classes=1,  # Single output for regression (always 1, regardless of input)
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm
        )
        
        # Override fc2 to output single value with sigmoid activation for [0, 1] range
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for age regression.
        
        Args:
            x: Input tensor of shape (batch_size, num_channels, sequence_length)
            
        Returns:
            Normalized age predictions in [0, 1] range, shape (batch_size, 1)
        """
        x = self._preprocess_input(x)
        x = self._apply_conv_layers(x)
        
        # Global average pooling across spatial dimensions
        x = F.adaptive_avg_pool2d(x, (1, 1))  # (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)             # (batch, 256)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Apply sigmoid to ensure output is in [0, 1] range
        x = torch.sigmoid(x)
        
        return x
    
    def _build_conv_layers(self) -> nn.ModuleList:
        """Build the convolutional layers for EEG regression."""
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for reporting."""
        info = super().get_model_info()
        info['prediction_type'] = 'regression'
        info['output_dim'] = 1
        return info