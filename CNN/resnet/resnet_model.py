"""
ResNet-like CNN Models for EEG Classification
Implements residual networks adapted for EEG time-series data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import sys
import os

# Add parent directory to path to import base model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CNN.models.base_model import BaseEEGCNN


class BasicBlock(nn.Module):
    """
    Basic ResNet block with two convolutional layers and skip connection.
    Used in ResNet-18 and ResNet-34.
    Adapted for EEG data with 1D temporal convolutions.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 use_batch_norm: bool = True, dropout_rate: float = 0.0):
        """
        Initialize ResNet block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel (default: 3)
            stride: Stride for convolution (default: 1)
            use_batch_norm: Whether to use batch normalization (default: True)
            dropout_rate: Dropout rate (default: 0.0)
        """
        super(BasicBlock, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        self.stride = stride
        
        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=(1, kernel_size), 
                               stride=(1, stride), 
                               padding=(0, kernel_size // 2),
                               bias=not use_batch_norm)
        
        # Batch normalization
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=(1, kernel_size),
                               stride=1,
                               padding=(0, kernel_size // 2),
                               bias=not use_batch_norm)
        
        if use_batch_norm:
            self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
        # Skip connection: 1x1 conv to match dimensions if needed
        self.skip_connection = None
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels,
                                             kernel_size=1,
                                             stride=(1, stride),
                                             bias=not use_batch_norm)
            if use_batch_norm:
                self.skip_bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet block.
        
        Args:
            x: Input tensor of shape (batch, in_channels, height, width)
            
        Returns:
            Output tensor of shape (batch, out_channels, height', width')
        """
        identity = x
        
        # First conv + BN + ReLU
        out = self.conv1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = F.relu(out)
        
        # Dropout if enabled
        if self.dropout is not None:
            out = self.dropout(out)
        
        # Second conv + BN
        out = self.conv2(out)
        if self.use_batch_norm:
            out = self.bn2(out)
        
        # Skip connection
        if self.skip_connection is not None:
            identity = self.skip_connection(identity)
            if self.use_batch_norm:
                identity = self.skip_bn(identity)
        
        # Add skip connection and apply ReLU
        out = out + identity
        out = F.relu(out)
        
        return out


class BottleneckBlock(nn.Module):
    """
    Bottleneck ResNet block with three convolutional layers (1x1, 3x3, 1x1).
    Used in ResNet-50 and deeper architectures.
    Adapted for EEG data with 1D temporal convolutions.
    
    Architecture: 1x1 conv (reduce) -> 3x3 conv (extract) -> 1x1 conv (expand)
    """
    
    expansion = 4  # Output channels = in_channels * expansion
    
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, use_batch_norm: bool = True):
        """
        Initialize Bottleneck block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Base number of output channels (will be multiplied by expansion)
            stride: Stride for the 3x3 convolution (default: 1)
            use_batch_norm: Whether to use batch normalization (default: True)
        """
        super(BottleneckBlock, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        self.stride = stride
        
        # Calculate actual output channels (with expansion)
        expanded_channels = out_channels * self.expansion
        
        # 1x1 conv: reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1,
                               stride=1,
                               bias=not use_batch_norm)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 conv: main feature extraction
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=(1, 3),
                               stride=(1, stride),
                               padding=(0, 1),
                               bias=not use_batch_norm)
        if use_batch_norm:
            self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 conv: expand channels
        self.conv3 = nn.Conv2d(out_channels, expanded_channels,
                               kernel_size=1,
                               stride=1,
                               bias=not use_batch_norm)
        if use_batch_norm:
            self.bn3 = nn.BatchNorm2d(expanded_channels)
        
        # Skip connection: 1x1 conv to match dimensions if needed
        self.skip_connection = None
        if stride != 1 or in_channels != expanded_channels:
            self.skip_connection = nn.Conv2d(in_channels, expanded_channels,
                                             kernel_size=1,
                                             stride=(1, stride),
                                             bias=not use_batch_norm)
            if use_batch_norm:
                self.skip_bn = nn.BatchNorm2d(expanded_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Bottleneck block.
        
        Args:
            x: Input tensor of shape (batch, in_channels, height, width)
            
        Returns:
            Output tensor of shape (batch, expanded_channels, height', width')
        """
        identity = x
        
        # 1x1 conv: reduce
        out = self.conv1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = F.relu(out)
        
        # 3x3 conv: extract
        out = self.conv2(out)
        if self.use_batch_norm:
            out = self.bn2(out)
        out = F.relu(out)
        
        # 1x1 conv: expand
        out = self.conv3(out)
        if self.use_batch_norm:
            out = self.bn3(out)
        
        # Skip connection
        if self.skip_connection is not None:
            identity = self.skip_connection(identity)
            if self.use_batch_norm:
                identity = self.skip_bn(identity)
        
        # Add skip connection and apply ReLU
        out = out + identity
        out = F.relu(out)
        
        return out


class EEGResNet(BaseEEGCNN):
    """
    ResNet-like CNN model for EEG classification.
    Uses residual blocks with skip connections for better gradient flow.
    
    Architecture:
    - Initial convolution to expand channels
    - Multiple ResNet blocks with increasing channels
    - Global average pooling
    - Fully connected layers for classification
    """
    
    def __init__(self, num_channels: int = 60, num_classes: int = 2,
                 dropout_rate: float = 0.5, use_layer_norm: bool = False,
                 use_batch_norm: bool = True, layers: list = None,
                 block_type: type = BasicBlock, base_channels: int = 64):
        """
        Initialize EEG ResNet model.
        
        Args:
            num_channels: Number of EEG channels (default: 60)
            num_classes: Number of output classes (default: 2)
            dropout_rate: Dropout rate for FC layers (default: 0.5)
            use_layer_norm: Whether to use layer normalization on input (default: False)
            use_batch_norm: Whether to use batch normalization in blocks (default: True)
            layers: List of number of blocks per stage, e.g., [2,2,2,2] for ResNet18 (default: None)
            block_type: Type of block to use (BasicBlock or BottleneckBlock) (default: BasicBlock)
            base_channels: Base number of channels (default: 64)
        """
        # Override use_layer_norm based on use_batch_norm preference
        # If using batch norm, we typically don't need layer norm
        # Set attributes BEFORE calling super().__init__() so they're available
        # if BaseEEGCNN.__init__() calls _build_conv_layers()
        self.use_batch_norm = use_batch_norm
        self.layers = layers or [2, 2, 2, 2]  # Default: ResNet18 structure
        self.block_type = block_type
        self.base_channels = base_channels
        
        # Calculate expansion factor (1 for BasicBlock, 4 for BottleneckBlock)
        self.expansion = getattr(block_type, 'expansion', 1)
        
        # Now call super().__init__() which may call _build_conv_layers()
        super(EEGResNet, self).__init__(
            num_channels=num_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm and not use_batch_norm
        )
        
        # Calculate final channels: last stage channels * expansion
        # Stages: [64, 128, 256, 512] for 4 stages
        num_stages = len(self.layers)
        final_channels = base_channels * (2 ** (num_stages - 1)) * self.expansion
        self.final_channels = final_channels
        
        # Rebuild conv layers with ResNet blocks
        self.conv_layers = self._build_conv_layers()
        
        # Update FC layers to match ResNet architecture
        # We'll use adaptive pooling, so input size is final_channels
        self.fc1 = nn.Linear(final_channels, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
    
    def _build_conv_layers(self) -> nn.ModuleList:
        """
        Build ResNet convolutional layers with residual blocks organized in stages.
        
        Standard ResNet architecture:
        - Stage 1: base_channels (64) channels
        - Stage 2: base_channels * 2 (128) channels
        - Stage 3: base_channels * 4 (256) channels
        - Stage 4: base_channels * 8 (512) channels
        
        Returns:
            ModuleList containing ResNet blocks organized in stages
        """
        layers = nn.ModuleList()
        
        # Initial convolution: expand from 1 channel to base_channels
        # Uses kernel_size=(num_channels, 7) to process all channels at once
        # This processes all 60 EEG channels simultaneously
        layers.append(nn.Conv2d(1, self.base_channels,
                                kernel_size=(self.num_channels, 7),
                                stride=(1, 2),
                                padding=(0, 3),
                                bias=not self.use_batch_norm))
        
        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(self.base_channels))
        
        # Build stages: each stage has multiple blocks
        in_channels = self.base_channels
        for stage_idx, num_blocks_in_stage in enumerate(self.layers):
            # Calculate base channels for this stage (before expansion)
            stage_base_channels = self.base_channels * (2 ** stage_idx)
            
            # First block in stage (except stage 0) uses stride=2 for downsampling
            stride = 2 if stage_idx > 0 else 1
            
            # Create blocks for this stage
            for block_idx in range(num_blocks_in_stage):
                # First block in stage: may need stride and channel change
                if block_idx == 0:
                    # First block: uses stride (may be 2 for downsampling)
                    block = self.block_type(in_channels, stage_base_channels,
                                           stride=stride,
                                           use_batch_norm=self.use_batch_norm)
                    # Update in_channels after first block
                    in_channels = stage_base_channels * self.expansion
                else:
                    # Subsequent blocks: stride=1, same channels
                    block = self.block_type(in_channels, stage_base_channels,
                                           stride=1,
                                           use_batch_norm=self.use_batch_norm)
                    # in_channels stays the same for subsequent blocks
                
                layers.append(block)
        
        return layers
    
    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input tensor for ResNet processing."""
        # Input shape: (batch, 60, 200) -> need to add channel dimension
        if x.dim() == 3:  # (batch, 60, 200)
            x = x.unsqueeze(1)  # (batch, 1, 60, 200)
        
        # Layer norm only if enabled and not using batch norm
        if self.use_layer_norm and not self.use_batch_norm:
            x = F.layer_norm(x, x.shape)
        
        return x
    
    def _apply_conv_layers(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all convolutional layers including ResNet blocks."""
        i = 0
        while i < len(self.conv_layers):
            layer = self.conv_layers[i]
            
            if isinstance(layer, (BasicBlock, BottleneckBlock)):
                # ResNet blocks handle their own forward pass
                x = layer(x)
            elif isinstance(layer, nn.Conv2d):
                # Initial conv layer
                x = layer(x)
                # Check if next layer is batch norm
                if i + 1 < len(self.conv_layers) and isinstance(self.conv_layers[i + 1], nn.BatchNorm2d):
                    x = self.conv_layers[i + 1](x)
                    i += 1  # Skip batch norm layer
                x = F.relu(x)
            # Skip batch norm layers (handled above)
            
            i += 1
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet.
        
        Args:
            x: Input tensor of shape (batch, num_channels, timepoints)
            
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        x = self._preprocess_input(x)
        x = self._apply_conv_layers(x)
        
        # Global average pooling across spatial dimensions
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for reporting."""
        info = super().get_model_info()
        info['architecture'] = 'ResNet'
        info['layers'] = self.layers
        info['total_blocks'] = sum(self.layers)
        info['block_type'] = self.block_type.__name__
        info['base_channels'] = self.base_channels
        info['final_channels'] = self.final_channels
        info['use_batch_norm'] = self.use_batch_norm
        return info


class EEGResNet18(EEGResNet):
    """
    ResNet-18 architecture for EEG classification.
    Uses 8 BasicBlocks arranged as [2,2,2,2] across 4 stages.
    
    Architecture:
    - Stage 1: 64 channels, 2 blocks
    - Stage 2: 128 channels, 2 blocks
    - Stage 3: 256 channels, 2 blocks
    - Stage 4: 512 channels, 2 blocks
    - Total: 8 BasicBlocks
    """
    def __init__(self, num_channels: int = 60, num_classes: int = 2,
                 dropout_rate: float = 0.5, use_layer_norm: bool = False,
                 use_batch_norm: bool = True):
        super(EEGResNet18, self).__init__(
            num_channels=num_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            layers=[2, 2, 2, 2],  # Standard ResNet-18: 8 BasicBlocks in 4 stages
            block_type=BasicBlock,
            base_channels=64
        )


class EEGResNet34(EEGResNet):
    """
    ResNet-34 architecture for EEG classification.
    Uses 16 BasicBlocks arranged as [3,4,6,3] across 4 stages.
    
    Architecture:
    - Stage 1: 64 channels, 3 blocks
    - Stage 2: 128 channels, 4 blocks
    - Stage 3: 256 channels, 6 blocks
    - Stage 4: 512 channels, 3 blocks
    - Total: 16 BasicBlocks
    """
    def __init__(self, num_channels: int = 60, num_classes: int = 2,
                 dropout_rate: float = 0.5, use_layer_norm: bool = False,
                 use_batch_norm: bool = True):
        super(EEGResNet34, self).__init__(
            num_channels=num_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            layers=[3, 4, 6, 3],  # Standard ResNet-34: 16 BasicBlocks in 4 stages
            block_type=BasicBlock,
            base_channels=64
        )


class EEGResNet50(EEGResNet):
    """
    ResNet-50 architecture for EEG classification.
    Uses 16 BottleneckBlocks arranged as [3,4,6,3] across 4 stages.
    
    Architecture:
    - Stage 1: 64*4=256 channels, 3 BottleneckBlocks
    - Stage 2: 128*4=512 channels, 4 BottleneckBlocks
    - Stage 3: 256*4=1024 channels, 6 BottleneckBlocks
    - Stage 4: 512*4=2048 channels, 3 BottleneckBlocks
    - Total: 16 BottleneckBlocks
    
    Note: Each BottleneckBlock has 3 conv layers (1x1, 3x3, 1x1) with expansion=4.
    """
    def __init__(self, num_channels: int = 60, num_classes: int = 2,
                 dropout_rate: float = 0.5, use_layer_norm: bool = False,
                 use_batch_norm: bool = True):
        super(EEGResNet50, self).__init__(
            num_channels=num_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            layers=[3, 4, 6, 3],  # Standard ResNet-50: 16 BottleneckBlocks in 4 stages
            block_type=BottleneckBlock,
            base_channels=64
        )


# Backward compatibility aliases and task-specific models
class EEGGenderResNet(EEGResNet18):
    """
    ResNet model for EEG gender classification (binary).
    """
    def __init__(self, num_channels: int = 60, num_classes: int = 2,
                 dropout_rate: float = 0.5, use_layer_norm: bool = False,
                 use_batch_norm: bool = True):
        super(EEGGenderResNet, self).__init__(
            num_channels=num_channels,
            num_classes=2,  # Force 2 classes for gender
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm
        )


class EEGAgeResNet(EEGResNet18):
    """
    ResNet model for EEG age classification (3-class).
    Uses ResNet18 architecture.
    """
    def __init__(self, num_channels: int = 60, num_classes: int = 3,
                 dropout_rate: float = 0.5, use_layer_norm: bool = False,
                 use_batch_norm: bool = True):
        super(EEGAgeResNet, self).__init__(
            num_channels=num_channels,
            num_classes=3,  # Force 3 classes for age
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm
        )


class EEGAgeResNet34(EEGResNet34):
    """
    ResNet34 model for EEG age classification (3-class).
    
    Uses ResNet34 architecture (16 BasicBlocks in [3,4,6,3] stages, 512 final channels).
    Inherits from EEGResNet34, which provides:
    - 16 BasicBlocks arranged as [3,4,6,3]
    - Final feature size: 512 channels
    - FC1: 512 -> 128
    - FC2: 128 -> 3 (age classes)
    """
    def __init__(self, num_channels: int = 60, num_classes: int = 3,
                 dropout_rate: float = 0.5, use_layer_norm: bool = False,
                 use_batch_norm: bool = True):
        super(EEGAgeResNet34, self).__init__(
            num_channels=num_channels,
            num_classes=3,  # Force 3 classes for age
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm
        )


class EEGGenderResNet34(EEGResNet34):
    """
    ResNet34 model for EEG gender classification (binary).
    
    Uses ResNet34 architecture (16 BasicBlocks in [3,4,6,3] stages, 512 final channels).
    Inherits from EEGResNet34, which provides:
    - 16 BasicBlocks arranged as [3,4,6,3]
    - Final feature size: 512 channels
    - FC1: 512 -> 128
    - FC2: 128 -> 2 (gender classes)
    """
    def __init__(self, num_channels: int = 60, num_classes: int = 2,
                 dropout_rate: float = 0.5, use_layer_norm: bool = False,
                 use_batch_norm: bool = True):
        super(EEGGenderResNet34, self).__init__(
            num_channels=num_channels,
            num_classes=2,  # Force 2 classes for gender
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm
        )


class EEGGenderResNet50(EEGResNet50):
    """
    ResNet50 model for EEG gender classification (binary).
    
    Uses ResNet50 architecture (16 BottleneckBlocks in [3,4,6,3] stages, 2048 final channels).
    Inherits from EEGResNet50, which provides:
    - 16 BottleneckBlocks arranged as [3,4,6,3]
    - Final feature size: 2048 channels
    - FC1: 2048 -> 128
    - FC2: 128 -> 2 (gender classes)
    """
    def __init__(self, num_channels: int = 60, num_classes: int = 2,
                 dropout_rate: float = 0.5, use_layer_norm: bool = False,
                 use_batch_norm: bool = True):
        super(EEGGenderResNet50, self).__init__(
            num_channels=num_channels,
            num_classes=2,  # Force 2 classes for gender
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm
        )


class EEGAgeResNet50(EEGResNet50):
    """
    ResNet50 model for EEG age classification (3-class).
    
    Uses ResNet50 architecture (16 BottleneckBlocks in [3,4,6,3] stages, 2048 final channels).
    Inherits from EEGResNet50, which provides:
    - 16 BottleneckBlocks arranged as [3,4,6,3] (vs BasicBlocks in ResNet34)
    - Final feature size: 2048 channels (vs 512 for ResNet34)
    - FC1: 2048 -> 128
    - FC2: 128 -> 3 (age classes)
    
    Difference from EEGAgeResNet34:
    - Uses BottleneckBlocks (3 conv layers) instead of BasicBlocks (2 conv layers)
    - Much larger feature representation (2048 vs 512 channels)
    - More parameters and capacity for complex patterns
    """
    def __init__(self, num_channels: int = 60, num_classes: int = 3,
                 dropout_rate: float = 0.5, use_layer_norm: bool = False,
                 use_batch_norm: bool = True):
        super(EEGAgeResNet50, self).__init__(
            num_channels=num_channels,
            num_classes=3,  # Force 3 classes for age
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm
        )


class CombinedResNet(EEGResNet18):
    """
    ResNet model for combined age+gender classification (6 classes).
    """
    def __init__(self, num_channels: int = 60, num_classes: int = 6,
                 dropout_rate: float = 0.5, use_layer_norm: bool = False,
                 use_batch_norm: bool = True):
        super(CombinedResNet, self).__init__(
            num_channels=num_channels,
            num_classes=6,  # Force 6 classes for combined
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm
        )


class MultiOutputResNet(EEGResNet18):
    """
    Multi-output ResNet model for simultaneous gender and age prediction.
    Has two separate output heads: one for gender (2 classes) and one for age (3 classes).
    Uses ResNet18 architecture (8 BasicBlocks in [2,2,2,2] stages).
    """
    def __init__(self, num_channels: int = 60, num_classes: int = 2,
                 dropout_rate: float = 0.5, use_layer_norm: bool = False,
                 use_batch_norm: bool = True):
        super(MultiOutputResNet, self).__init__(
            num_channels=num_channels,
            num_classes=2,  # Will be overridden by separate heads
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm
        )
        
        # Override the classifier with two separate heads
        # fc1 outputs 128 features (from base class)
        self.gender_classifier = nn.Linear(128, 2)  # 2 classes for gender
        self.age_classifier = nn.Linear(128, 3)     # 3 classes for age
        
        # Remove the original classifier
        del self.fc2
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with two separate outputs.
        
        Args:
            x: Input tensor of shape (batch_size, num_channels, sequence_length)
            
        Returns:
            Dictionary with 'gender' and 'age' predictions
        """
        x = self._preprocess_input(x)
        x = self._apply_conv_layers(x)
        
        # Global average pooling across spatial dimensions
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
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


class EEGAgeRegressionResNet(EEGResNet18):
    """
    ResNet model for EEG age regression.
    Outputs a single continuous value (normalized age in [0, 1] range).
    Uses ResNet18 architecture (8 BasicBlocks in [2,2,2,2] stages).
    """
    def __init__(self, num_channels: int = 60, num_classes: int = 1,
                 dropout_rate: float = 0.5, use_layer_norm: bool = False,
                 use_batch_norm: bool = True):
        super(EEGAgeRegressionResNet, self).__init__(
            num_channels=num_channels,
            num_classes=1,  # Single output for regression
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm
        )
        
        # Override fc2 to output single value
        # fc1 outputs 128 features (from base class)
        self.fc2 = nn.Linear(128, 1)
    
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
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Apply sigmoid to ensure output is in [0, 1] range
        x = torch.sigmoid(x)
        
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for reporting."""
        info = super().get_model_info()
        info['prediction_type'] = 'regression'
        info['output_dim'] = 1
        return info

