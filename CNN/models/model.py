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


class EEGUserIdentificationCNN(EEGCNN):
    """
    CNN model for user identification (multi-class classification).
    Uses EEGCNN with dynamic number of classes based on number of participants.
    
    For large classification tasks (many classes), this model uses:
    - Larger fully connected layers for better capacity
    - Better weight initialization for the final classification layer
    """
    # Weight initialization multiplier for final layer (no hardcoding)
    # This ensures logits start in a safe range for large classification tasks
    FINAL_LAYER_INIT_MULTIPLIER = 5.0
    
    def __init__(self, num_channels=64, num_classes=None, dropout_rate=0.5, use_layer_norm=True):
        if num_classes is None:
            raise ValueError("num_classes must be specified for user identification. "
                           "It should equal the number of unique participants.")
        
        # Call parent __init__ first
        super(EEGUserIdentificationCNN, self).__init__(
            num_channels=num_channels, 
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm
        )
        
        # Override FC layers with larger capacity for large classification tasks
        # Original: 256 -> 64 -> num_classes (too small bottleneck for 3000+ classes)
        # Previous: 256 -> 2048 -> 1024 -> num_classes (insufficient for 3145 classes)
        # New: 512 -> 2048 -> 1024 -> num_classes (increased capacity with deeper conv layers)
        # This provides more representational power for distinguishing between 3145 users
        # Embedding dimension of 1024 is recommended for 3000+ classes
        # Note: Input to FC is now 512 (from deeper conv layers) instead of 256
        self.fc1 = nn.Linear(512, 2048)
        self.fc2_intermediate = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, num_classes)
        
        # Initialize weights properly for large classification
        self._initialize_weights()
    
    def _build_conv_layers(self) -> nn.ModuleList:
        """
        Build deeper convolutional layers for user identification.
        
        For 3145 classes, we need deeper feature extraction to learn
        discriminative features for each user. This adds 2 more layers
        compared to the base model.
        
        Architecture: 1->16->32->64->128->128->256->256->256->512->512
        """
        return nn.ModuleList([
            nn.Conv2d(1, 16, kernel_size=(self.num_channels, 3)),
            nn.Conv2d(16, 32, kernel_size=(1, 3)),
            nn.Conv2d(32, 64, kernel_size=(1, 3)),
            nn.Conv2d(64, 128, kernel_size=(1, 3)),
            nn.Conv2d(128, 128, kernel_size=(1, 3)),
            nn.Conv2d(128, 256, kernel_size=(1, 3)),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            nn.Conv2d(256, 256, kernel_size=(1, 3)),
            # Additional layers for deeper feature extraction
            nn.Conv2d(256, 512, kernel_size=(1, 3)),
            nn.Conv2d(512, 512, kernel_size=(1, 3))
        ])
    
    def _initialize_weights(self):
        """
        Initialize weights with better strategies for large classification.
        Uses Kaiming initialization for ReLU layers and small initialization for final layer.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m == self.fc2:
                    # Final classification layer: use larger initialization for numerical stability
                    # For large number of classes (3145), very small logits cause numerical issues
                    # with label smoothing (softmax probabilities become too small, -log overflows)
                    # We use a larger std to ensure logits are in a safe range
                    fan_in = m.weight.size(1)
                    fan_out = m.weight.size(0)
                    # Xavier init: std = sqrt(2.0 / (fan_in + fan_out))
                    # For 1024 -> 3145: std ≈ sqrt(2.0 / (1024 + 3145)) ≈ 0.021
                    # Scale up significantly for large classification to prevent numerical issues
                    # Target: logits in range [-1, 1] to avoid extreme softmax values
                    # With 3145 classes, we need logits large enough that softmax probabilities
                    # don't become so small that -log overflows with label smoothing
                    std = (2.0 / (fan_in + fan_out)) ** 0.5
                    # Use class constant instead of hardcoded value (no hardcoding)
                    std = std * self.FINAL_LAYER_INIT_MULTIPLIER
                    nn.init.normal_(m.weight, mean=0.0, std=std)
                    # Initialize bias to zero (let weights do the work)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                else:
                    # Intermediate layers: use Kaiming initialization for ReLU
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def _apply_fc_layers(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply fully connected layers with increased capacity for user identification.
        
        Args:
            x: Input tensor of shape (batch_size, 512)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = F.relu(self.fc1(x))  # 512 -> 2048
        x = self.dropout(x)
        x = F.relu(self.fc2_intermediate(x))  # 2048 -> 1024
        x = self.dropout(x)
        x = self.fc2(x)  # 1024 -> num_classes
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings before the final classification layer.
        This is used for ArcFace loss, which requires normalized features.
        
        Args:
            x: Input tensor of shape (batch_size, num_channels, sequence_length)
            
        Returns:
            Feature embeddings of shape (batch_size, 1024)
            (output of fc2_intermediate, before fc2)
        """
        x = self._preprocess_input(x)
        x = self._apply_conv_layers(x)
        
        # Global average pooling across spatial dimensions
        x = F.adaptive_avg_pool2d(x, (1, 1))  # (batch, 512, 1, 1)
        x = x.view(x.size(0), -1)             # (batch, 512)
        
        # Apply FC layers up to (but not including) final classification layer
        x = F.relu(self.fc1(x))  # 512 -> 2048
        # CRITICAL: Apply dropout only in training mode to ensure feature diversity
        # In eval mode, dropout is disabled, which can cause features to be too similar
        # and lead to model collapse (all predictions to same class)
        x = self.dropout(x)
        x = F.relu(self.fc2_intermediate(x))  # 2048 -> 1024
        # Note: We don't apply dropout after fc2_intermediate to preserve feature information
        # The dropout after fc1 is sufficient for regularization
        
        # Return features before final classification layer
        # These will be normalized and used with ArcFace
        return x

    @property
    def arcface_embedding_dim(self) -> int:
        """Dimension of vectors returned by extract_features() (ArcFace head)."""
        return int(self.fc2_intermediate.out_features)

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