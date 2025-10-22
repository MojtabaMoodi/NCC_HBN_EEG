import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseEEGCNN(nn.Module, ABC):
    """
    Base class for EEG CNN models providing common functionality.
    This class implements the DRY principle by centralizing common operations.
    """
    
    def __init__(self, num_channels: int = 64, num_classes: int = 2, 
                 dropout_rate: float = 0.5, use_layer_norm: bool = True):
        super(BaseEEGCNN, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        
        # Common convolutional layers
        self.conv_layers = self._build_conv_layers()
        
        # Common fully connected layers
        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)
        
    @abstractmethod
    def _build_conv_layers(self) -> nn.ModuleList:
        """Build the convolutional layers. Must be implemented by subclasses."""
        pass
    
    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input tensor for CNN processing."""
        # Input shape: (batch, 60, 200) -> need to add channel dimension
        if x.dim() == 3:  # (batch, 60, 200)
            x = x.unsqueeze(1)  # (batch, 1, 60, 200)
        
        if self.use_layer_norm:
            x = F.layer_norm(x, x.shape)
        
        return x
    
    def _apply_conv_layers(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all convolutional layers with ReLU activation."""
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))
        return x
    
    def _apply_fc_layers(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fully connected layers."""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self._preprocess_input(x)
        x = self._apply_conv_layers(x)
        
        # Global average pooling across spatial dimensions
        x = F.adaptive_avg_pool2d(x, (1, 1))  # (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)             # (batch, 256)
        
        x = self._apply_fc_layers(x)
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for reporting."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'num_channels': self.num_channels,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'use_layer_norm': self.use_layer_norm,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'conv_layers': len(self.conv_layers)
        }
