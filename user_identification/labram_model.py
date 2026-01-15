"""
LaBraM Model Wrapper for User Identification

This module provides a LaBraM wrapper that integrates with the existing CNN framework
for user identification tasks. It maintains compatibility with BaseEEGCNN interface
while using LaBraM's transformer architecture.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path for absolute imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import from CNN module using absolute imports
from CNN.models.base_model import BaseEEGCNN

# Add LaBraM directory to path for imports
_labram_dir = Path(__file__).parent.parent / "LaBraM"
if str(_labram_dir) not in sys.path:
    sys.path.insert(0, str(_labram_dir))

try:
    from timm.models import create_model
    # Import LaBraM modules (they're not a package, so we import directly after adding to path)
    import importlib.util
    _modeling_finetune_path = _labram_dir / "modeling_finetune.py"
    if _modeling_finetune_path.exists():
        spec = importlib.util.spec_from_file_location("modeling_finetune", _modeling_finetune_path)
        modeling_finetune = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modeling_finetune)
        NeuralTransformer = modeling_finetune.NeuralTransformer
    else:
        # Fallback to direct import (works when LaBraM is in path)
        from modeling_finetune import NeuralTransformer  # type: ignore
    
    import utils as labram_utils  # type: ignore
except ImportError as e:
    raise ImportError(
        f"Failed to import LaBraM modules. Make sure LaBraM directory is available. Error: {e}"
    )


class LaBraMUserIdentificationWrapper(BaseEEGCNN):
    """
    LaBraM wrapper for user identification tasks.
    
    This class wraps LaBraM's NeuralTransformer model to provide compatibility
    with the existing CNN framework while leveraging LaBraM's transformer architecture
    for better performance on large-scale user identification tasks.
    
    The wrapper handles:
    - Input format conversion: [B, 60, T] -> [B, 60, num_patches, patch_size]
    - Channel name mapping to LaBraM's input_chans format
    - Integration with existing training pipeline
    """
    
    # Standard 10-20 channel names (60 channels from your data)
    # These should match the channel order in your HDF5 files
    STANDARD_CHANNEL_NAMES = [
        'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'F1', 'F2', 'F5', 'F6', 'F9', 'F10', 
        'AF3', 'AF4', 'AF7', 'AF8', 'AFZ', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 
        'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'P7', 'P3', 'PZ', 'P4', 'P8', 'P1', 'P2', 
        'P5', 'P6', 'PO3', 'PO4', 'PO7', 'PO8', 'POZ', 'OZ', 'O1', 'O2', 'C3', 'C4', 
        'C1', 'C2', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPZ'
    ]
    
    def __init__(self, num_channels: int = 60, num_classes: int = None, 
                 dropout_rate: float = 0.5, use_layer_norm: bool = True,
                 pretrained_path: Optional[str] = None,
                 model_name: str = "labram_base_patch200_200",
                 patch_size: int = 200,
                 embed_dim: int = 200,
                 depth: int = 12,
                 num_heads: int = 10,
                 mlp_ratio: float = 4.0):
        """
        Initialize LaBraM wrapper for user identification.
        
        Args:
            num_channels: Number of EEG channels (default: 60)
            num_classes: Number of user classes (required for user identification)
            dropout_rate: Dropout rate (used for LaBraM's drop_rate)
            use_layer_norm: Whether to use layer normalization (LaBraM always uses it)
            pretrained_path: Path to pre-trained LaBraM checkpoint (optional)
            model_name: LaBraM model name (default: "labram_base_patch200_200")
            patch_size: Patch size for LaBraM (default: 200)
            embed_dim: Embedding dimension (default: 200)
            depth: Number of transformer layers (default: 12)
            num_heads: Number of attention heads (default: 10)
            mlp_ratio: MLP ratio (default: 4.0)
        """
        if num_classes is None:
            raise ValueError(
                "num_classes must be specified for user identification. "
                "It should equal the number of unique participants."
            )
        
        # Initialize base class (required for compatibility)
        # We'll override most methods, but need BaseEEGCNN for ModelFactory compatibility
        super(LaBraMUserIdentificationWrapper, self).__init__(
            num_channels=num_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm
        )
        
        self.patch_size = patch_size
        self.model_name = model_name
        
        # Create LaBraM model using timm's create_model
        # multi_output=False for single classification head (user identification)
        self.labram_model = create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            drop_rate=dropout_rate,
            drop_path_rate=0.1,  # Standard for fine-tuning
            attn_drop_rate=0.0,
            use_mean_pooling=True,  # Use mean pooling for classification
            init_scale=0.001,  # Standard initialization scale
            use_rel_pos_bias=False,  # Disable relative position bias for simplicity
            use_abs_pos_emb=True,  # Use absolute position embeddings
            init_values=0.1,  # Layer scale initialization
            qkv_bias=False,  # Disable QKV bias (standard for LaBraM)
            multi_output=False,  # Single output for user identification
            EEG_size=None,  # Will be determined from input
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )
        
        # Load pre-trained weights if provided
        if pretrained_path is not None:
            self._load_pretrained_weights(pretrained_path)
        
        # Compute input_chans from channel names
        # This maps channel names to LaBraM's standard_1020 indices
        self.input_chans = self._compute_input_chans()
        
        # Store num_classes for compatibility
        self.num_classes = num_classes
    
    def _compute_input_chans(self) -> List[int]:
        """
        Compute input_chans parameter for LaBraM from channel names.
        
        Returns:
            List of channel indices for LaBraM's input_chans parameter
        """
        try:
            # Use LaBraM's utility function to get input_chans
            input_chans = labram_utils.get_input_chans(self.STANDARD_CHANNEL_NAMES)
            return input_chans
        except Exception as e:
            # Fallback: if channel names don't match, use sequential indices
            # This assumes channels are in standard order
            print(f"Warning: Could not map channel names to LaBraM standard_1020. "
                  f"Using sequential indices. Error: {e}")
            return [0] + list(range(1, self.num_channels + 1))  # [0] for CLS token
    
    def _load_pretrained_weights(self, checkpoint_path: str):
        """
        Load pre-trained LaBraM weights from checkpoint.
        
        Args:
            checkpoint_path: Path to pre-trained checkpoint file
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load weights (may need to handle key name mismatches)
            self.labram_model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pre-trained weights from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load pre-trained weights from {checkpoint_path}: {e}")
            print("Continuing with random initialization...")
    
    def _reshape_input_for_labram(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape input from [B, 60, T] to [B, 60, num_patches, patch_size] format.
        
        LaBraM expects input shape: [batch_size, num_electrodes, num_patches, patch_size]
        For example, for 4s segments at 200Hz: [B, 60, 4, 200]
        
        Args:
            x: Input tensor of shape [batch_size, num_channels, sequence_length]
            
        Returns:
            Reshaped tensor of shape [batch_size, num_channels, num_patches, patch_size]
        """
        batch_size, num_channels, sequence_length = x.shape
        
        # Calculate number of patches
        num_patches = sequence_length // self.patch_size
        
        if sequence_length % self.patch_size != 0:
            # If sequence length is not divisible by patch_size, pad or truncate
            target_length = num_patches * self.patch_size
            if sequence_length < target_length:
                # Pad with zeros
                padding = target_length - sequence_length
                x = torch.nn.functional.pad(x, (0, padding), mode='constant', value=0)
            else:
                # Truncate
                x = x[:, :, :target_length]
        
        # Reshape: [B, 60, T] -> [B, 60, num_patches, patch_size]
        x = x.view(batch_size, num_channels, num_patches, self.patch_size)
        
        return x
    
    def forward(self, x: torch.Tensor, input_chans: Optional[List[int]] = None) -> torch.Tensor:
        """
        Forward pass through LaBraM model.
        
        Args:
            x: Input tensor of shape [batch_size, num_channels, sequence_length]
            input_chans: Optional channel indices (if None, uses self.input_chans)
            
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        # Reshape input to LaBraM format
        x = self._reshape_input_for_labram(x)
        
        # Use provided input_chans or default
        if input_chans is None:
            input_chans = self.input_chans
        
        # Forward through LaBraM model
        output = self.labram_model(x, input_chans=input_chans)
        
        return output
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings before the final classification layer.
        This is used for ArcFace loss, which requires normalized features.
        
        Args:
            x: Input tensor of shape [batch_size, num_channels, sequence_length]
            
        Returns:
            Feature embeddings of shape [batch_size, embed_dim]
        """
        # Reshape input to LaBraM format
        x = self._reshape_input_for_labram(x)
        
        # Extract features (before classification head)
        features = self.labram_model.forward_features(
            x, 
            input_chans=self.input_chans,
            return_patch_tokens=False,
            return_all_tokens=False
        )
        
        # Features shape: [batch_size, embed_dim]
        return features
    
    def _build_conv_layers(self) -> nn.ModuleList:
        """
        Override base class method - not used for LaBraM.
        LaBraM uses transformer blocks instead of convolutional layers.
        """
        return nn.ModuleList([])  # Empty - not used
    
    def _apply_conv_layers(self, x: torch.Tensor) -> torch.Tensor:
        """
        Override base class method - not used for LaBraM.
        """
        return x  # No-op
    
    def _apply_fc_layers(self, x: torch.Tensor) -> torch.Tensor:
        """
        Override base class method - not used for LaBraM.
        LaBraM has its own classification head.
        """
        return x  # No-op
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for reporting."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': f'LaBraMUserIdentificationWrapper({self.model_name})',
            'num_channels': self.num_channels,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'use_layer_norm': self.use_layer_norm,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'patch_size': self.patch_size,
            'architecture': 'Transformer (LaBraM)'
        }
