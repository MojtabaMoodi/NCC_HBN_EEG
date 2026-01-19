"""
GPU Utility Functions for Centralized GPU Detection and Setup

This module provides centralized functions for GPU detection, device setup,
and model wrapping to ensure consistent GPU usage across the codebase.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def detect_available_gpus() -> int:
    """
    Detect the number of available GPUs.
    
    Returns:
        Number of available GPUs (0 if CUDA is not available)
    
    Raises:
        RuntimeError: If CUDA is requested but not available
    """
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def get_device(device_preference: str = 'auto') -> torch.device:
    """
    Get the appropriate device based on preference and availability.
    
    Args:
        device_preference: Device preference ('auto', 'cuda', 'cpu')
    
    Returns:
        torch.device object
    
    Raises:
        RuntimeError: If 'cuda' is requested but CUDA is not available
    """
    if device_preference == 'cpu':
        return torch.device('cpu')
    elif device_preference == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Use 'auto' or 'cpu' instead.")
        return torch.device('cuda')
    else:  # 'auto'
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def determine_num_gpus_to_use(requested_num_gpus: int, 
                               available_gpus: Optional[int] = None) -> int:
    """
    Determine how many GPUs to use based on request and availability.
    
    Args:
        requested_num_gpus: Number of GPUs requested from config
        available_gpus: Number of available GPUs (None = auto-detect)
    
    Returns:
        Number of GPUs to actually use
    
    Raises:
        ValueError: If requested_num_gpus is invalid (< 1)
        RuntimeError: If GPUs are requested but not available
    """
    if requested_num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {requested_num_gpus}")
    
    if available_gpus is None:
        available_gpus = detect_available_gpus()
    
    if requested_num_gpus > 1 and available_gpus == 0:
        raise RuntimeError(
            f"Requested {requested_num_gpus} GPUs but CUDA is not available. "
            "Set num_gpus=1 to use CPU, or ensure CUDA is properly configured."
        )
    
    if requested_num_gpus > available_gpus:
        raise RuntimeError(
            f"Requested {requested_num_gpus} GPUs but only {available_gpus} available. "
            f"Please set num_gpus={available_gpus} or lower in your config."
        )
    
    return requested_num_gpus


def setup_model_for_gpus(model: nn.Module,
                         num_gpus: int,
                         device: torch.device) -> nn.Module:
    """
    Setup model for multi-GPU training using DataParallel.
    
    Args:
        model: Model to wrap
        num_gpus: Number of GPUs to use
        device: Device to move model to
    
    Returns:
        Model wrapped with DataParallel if num_gpus > 1, otherwise original model
    
    Raises:
        RuntimeError: If CUDA is not available but GPUs are requested
        ValueError: If num_gpus is invalid
    """
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
    
    if num_gpus == 1:
        # Single GPU or CPU - no wrapping needed
        model = model.to(device)
        return model
    
    # Multi-GPU setup
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested {num_gpus} GPUs but CUDA is not available. "
            "Set num_gpus=1 to use CPU."
        )
    
    available_gpus = detect_available_gpus()
    if num_gpus > available_gpus:
        raise RuntimeError(
            f"Requested {num_gpus} GPUs but only {available_gpus} available. "
            f"Please set num_gpus={available_gpus} or lower."
        )
    
    # Wrap with DataParallel
    device_ids = list(range(num_gpus))
    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    
    return model


def get_underlying_model(model: nn.Module) -> nn.Module:
    """
    Get the underlying model by unwrapping DataParallel and torch.compile wrappers.
    
    Args:
        model: Model that may be wrapped with DataParallel or torch.compile
    
    Returns:
        Unwrapped base model
    """
    while True:
        # Unwrap torch.compile wrapper
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
            continue
        
        # Unwrap DataParallel wrapper
        if isinstance(model, nn.DataParallel):
            model = model.module
            continue
        
        # No more wrappers
        break
    
    return model


def print_gpu_info(num_gpus_requested: int, num_gpus_used: int, device: torch.device):
    """
    Print GPU configuration information.
    
    Args:
        num_gpus_requested: Number of GPUs requested from config
        num_gpus_used: Number of GPUs actually used
        device: Device being used
    """
    available_gpus = detect_available_gpus()
    
    print("=" * 60)
    print("GPU Configuration")
    print("=" * 60)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Available GPUs: {available_gpus}")
    print(f"Requested GPUs: {num_gpus_requested}")
    print(f"Using GPUs: {num_gpus_used}")
    print(f"Device: {device}")
    
    if torch.cuda.is_available() and num_gpus_used > 0:
        print(f"GPU Device IDs: {list(range(num_gpus_used))}")
        for i in range(num_gpus_used):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"  GPU {i}: {gpu_name}")
    
    print("=" * 60)

