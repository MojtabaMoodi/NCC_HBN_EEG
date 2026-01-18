"""
Model loading utilities for CNN, LaBraM, and ResNet models.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
import importlib
import warnings


def load_cnn_model(
    model_type: str,
    checkpoint_path: str,
    num_classes: int = 3,
    device: str = "cuda",
    **model_kwargs
) -> nn.Module:
    """
    Load CNN model from checkpoint.
    
    Args:
        model_type: Model type (e.g., 'age_cnn', 'eeg_cnn')
        checkpoint_path: Path to checkpoint file
        num_classes: Number of classes
        device: Device to load model on
        **model_kwargs: Additional model initialization arguments
        
    Returns:
        Loaded model in eval mode
    """
    # Add CNN directory to path and import modules
    cnn_dir = Path(__file__).parent.parent / "CNN"
    cnn_dir_str = str(cnn_dir.resolve())
    if cnn_dir_str not in sys.path:
        sys.path.insert(0, cnn_dir_str)
    
    # Use importlib for better import resolution
    model_factory_module = importlib.import_module('models.model_factory', package=None)
    trainer_module = importlib.import_module('trainer', package=None)
    ModelFactory = model_factory_module.ModelFactory
    load_checkpoint = trainer_module.load_checkpoint
    
    # Create model
    model = ModelFactory.create_model(
        model_type,
        num_classes=num_classes,
        num_channels=60,  # Standard for EEG data
        **model_kwargs
    )
    
    # Load checkpoint
    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, model, device_obj, strict=False)
    
    model.eval()
    return model


def load_resnet_model(
    model_type: str,
    checkpoint_path: str,
    num_classes: int = 3,
    device: str = "cuda",
    **model_kwargs
) -> nn.Module:
    """
    Load ResNet model from checkpoint.
    
    Args:
        model_type: Model type (e.g., 'age_resnet', 'age_resnet34')
        checkpoint_path: Path to checkpoint file
        num_classes: Number of classes
        device: Device to load model on
        **model_kwargs: Additional model initialization arguments
        
    Returns:
        Loaded model in eval mode
    """
    # Add CNN directory to path and import modules
    cnn_dir = Path(__file__).parent.parent / "CNN"
    cnn_dir_str = str(cnn_dir.resolve())
    if cnn_dir_str not in sys.path:
        sys.path.insert(0, cnn_dir_str)
    
    # Use importlib for better import resolution
    model_factory_module = importlib.import_module('models.model_factory', package=None)
    trainer_module = importlib.import_module('trainer', package=None)
    ModelFactory = model_factory_module.ModelFactory
    load_checkpoint = trainer_module.load_checkpoint
    
    # Create model
    model = ModelFactory.create_model(
        model_type,
        num_classes=num_classes,
        num_channels=60,  # Standard for EEG data
        **model_kwargs
    )
    
    # Load checkpoint
    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, model, device_obj, strict=False)
    
    model.eval()
    return model


def get_labram_input_chans():
    """
    Get input_chans for LaBraM model using standard 10-20 channel names.
    
    Returns:
        List of channel indices for LaBraM input_chans parameter
    """
    # Standard 10-20 channel names (60 channels from EEG data)
    # These should match the channel order in HDF5 files
    standard_channel_names = [
        'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'F1', 'F2', 'F5', 'F6', 'F9', 'F10', 
        'AF3', 'AF4', 'AF7', 'AF8', 'AFZ', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 
        'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'P7', 'P3', 'PZ', 'P4', 'P8', 'P1', 'P2', 
        'P5', 'P6', 'PO3', 'PO4', 'PO7', 'PO8', 'POZ', 'OZ', 'O1', 'O2', 'C3', 'C4', 
        'C1', 'C2', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPZ'
    ]
    
    # Add LaBraM directory to path
    labram_dir = Path(__file__).parent.parent / "LaBraM"
    labram_dir_str = str(labram_dir.resolve())
    if labram_dir_str not in sys.path:
        sys.path.insert(0, labram_dir_str)
    
    try:
        import utils as labram_utils
        input_chans = labram_utils.get_input_chans(standard_channel_names)
        return input_chans
    except Exception as e:
        warnings.warn(
            f"Could not compute input_chans from channel names: {e}. "
            f"Using sequential indices as fallback."
        )
        # Fallback: sequential indices [0, 1, 2, ..., 60] where 0 is for CLS token
        return [0] + list(range(1, len(standard_channel_names) + 1))


def load_labram_model(
    model_name: str,
    checkpoint_path: str,
    num_classes: int = 3,
    device: str = "cuda",
    **model_kwargs
) -> nn.Module:
    """
    Load LaBraM model from checkpoint.
    
    Args:
        model_name: LaBraM model name (e.g., 'labram_base_patch200_200')
        checkpoint_path: Path to checkpoint file
        num_classes: Number of classes
        device: Device to load model on
        **model_kwargs: Additional model initialization arguments
        
    Returns:
        Loaded model in eval mode
    """
    # Add LaBraM directory to path
    labram_dir = Path(__file__).parent.parent / "LaBraM"
    labram_dir_str = str(labram_dir.resolve())
    if labram_dir_str not in sys.path:
        sys.path.insert(0, labram_dir_str)
    
    try:
        from timm.models import create_model
        # Import modeling_finetune dynamically to register models with timm
        importlib.import_module('modeling_finetune', package=None)
    except ImportError as e:
        raise ImportError(f"Failed to import LaBraM modules: {e}")
    
    # Create model
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=model_kwargs.get('drop_rate', 0.5),
        drop_path_rate=model_kwargs.get('drop_path_rate', 0.1),
        attn_drop_rate=model_kwargs.get('attn_drop_rate', 0.0),
        use_mean_pooling=model_kwargs.get('use_mean_pooling', True),
        init_scale=model_kwargs.get('init_scale', 0.001),
        use_rel_pos_bias=model_kwargs.get('use_rel_pos_bias', False),
        use_abs_pos_emb=model_kwargs.get('use_abs_pos_emb', True),
        init_values=model_kwargs.get('init_values', 0.1),
        qkv_bias=model_kwargs.get('qkv_bias', False),
        multi_output=model_kwargs.get('multi_output', False),
        EEG_size=model_kwargs.get('EEG_size', None),
    )
    
    # Load checkpoint
    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
        
        # Handle different checkpoint formats
        # LaBraM finetuned checkpoints typically have 'model' key
        # Pretrained checkpoints might be direct state_dict or have 'model' key
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # Assume it's a direct state_dict
            state_dict = checkpoint
        
        # Handle DataParallel prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        state_dict = new_state_dict
        
        # Remove relative_position_index keys (they're computed, not stored)
        state_dict = {k: v for k, v in state_dict.items() if 'relative_position_index' not in k}
        
        # Use LaBraM's load_state_dict which handles missing keys gracefully
        try:
            import utils as labram_utils
            labram_utils.load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index")
        except Exception as e:
            # Fallback to standard loading with strict=False
            warnings.warn(f"LaBraM load_state_dict failed, using standard loading: {e}")
            model.load_state_dict(state_dict, strict=False)
        
        model.to(device_obj)
    except Exception as e:
        warnings.warn(f"Could not load checkpoint {checkpoint_path}: {e}. Using random initialization.")
        model.to(device_obj)
    
    model.eval()
    return model


def load_model_from_config(model_config: Dict[str, Any], device: str = "cuda") -> nn.Module:
    """
    Load model from configuration dictionary.
    
    Expected config format:
    {
        'name': 'cnn' | 'labram' | 'resnet',
        'ckpt_path': str,
        'model_type': str,  # e.g., 'age_cnn', 'labram_base_patch200_200'
        'num_classes': int,
        'ctor_kwargs': dict,  # Optional
        'output_key': str,  # Optional, for dict outputs
    }
    
    Args:
        model_config: Model configuration dictionary
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    name = model_config['name'].lower()
    ckpt_path = model_config['ckpt_path']
    num_classes = model_config.get('num_classes', 3)
    model_type = model_config.get('model_type', None)
    ctor_kwargs = model_config.get('ctor_kwargs', {})
    
    if name == 'cnn':
        if model_type is None:
            raise ValueError("model_type required for CNN models (e.g., 'age_cnn')")
        return load_cnn_model(model_type, ckpt_path, num_classes, device, **ctor_kwargs)
    
    elif name == 'resnet':
        if model_type is None:
            raise ValueError("model_type required for ResNet models (e.g., 'age_resnet')")
        return load_resnet_model(model_type, ckpt_path, num_classes, device, **ctor_kwargs)
    
    elif name == 'labram':
        if model_type is None:
            model_type = 'labram_base_patch200_200'  # Default
        return load_labram_model(model_type, ckpt_path, num_classes, device, **ctor_kwargs)
    
    else:
        raise ValueError(f"Unknown model name: {name}. Expected 'cnn', 'resnet', or 'labram'")


def get_model_output(
    model: nn.Module, 
    x: torch.Tensor, 
    output_key: Optional[str] = None,
    model_name: Optional[str] = None,
    input_chans: Optional[list] = None
) -> torch.Tensor:
    """
    Get model output (logits or probabilities).
    
    Args:
        model: Model instance
        x: Input tensor, shape [B, 60, T] for CNN/ResNet or [B, 60, A, T] for LaBraM
        output_key: Optional key if model returns dict
        model_name: Optional model name to determine preprocessing (e.g., 'labram')
        input_chans: Optional input_chans for LaBraM models
        
    Returns:
        Output tensor (logits or probabilities)
    """
    with torch.no_grad():
        # Handle LaBraM-specific preprocessing
        # Check if this is a LaBraM model (by name or by type)
        is_labram = False
        if model_name and model_name.lower() == 'labram':
            is_labram = True
        elif model_name is None:
            # Try to detect from model type
            model_type_str = str(type(model)).lower()
            if 'neuraltransformer' in model_type_str or 'labram' in model_type_str:
                is_labram = True
        
        if is_labram:
            # LaBraM expects [B, 60, num_patches, patch_size] format
            # If input is [B, 60, T], reshape it
            if x.dim() == 3:
                # Debug: print input shape
                # print(f"LaBraM preprocessing: input shape {x.shape}, reshaping to 4D")
                batch_size, num_channels, seq_length = x.shape
                patch_size = 200  # Standard LaBraM patch size
                num_patches = seq_length // patch_size
                
                # Pad or truncate to match patch size
                target_length = num_patches * patch_size
                if seq_length < target_length:
                    x = torch.nn.functional.pad(x, (0, target_length - seq_length), mode='constant', value=0)
                elif seq_length > target_length:
                    x = x[:, :, :target_length]
                
                # Reshape: [B, 60, T] -> [B, 60, num_patches, patch_size]
                x = x.view(batch_size, num_channels, num_patches, patch_size)
            
            # Normalize by 100 (LaBraM standard preprocessing)
            x = x / 100.0
            
            # Call model with input_chans if provided
            if input_chans is not None:
                output = model(x, input_chans=input_chans)
            else:
                output = model(x)
        else:
            # CNN/ResNet: direct forward pass
            output = model(x)
    
    # Handle dict outputs
    if isinstance(output, dict):
        if output_key is None:
            # Try common keys
            for key in ['logits', 'output', 'pred', 'prediction', 'age']:
                if key in output:
                    output = output[key]
                    break
            else:
                raise ValueError(
                    f"Model returned dict but no output_key specified. "
                    f"Available keys: {list(output.keys())}"
                )
        else:
            output = output[output_key]
    
    return output
