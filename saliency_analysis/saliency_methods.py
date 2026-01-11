"""
Saliency Map Methods for EEG Channel Importance Analysis

This module implements various gradient-based saliency methods to identify
which EEG channels are most important for model predictions.

Methods implemented:
- Vanilla Gradients: Basic gradient of output w.r.t. input
- Integrated Gradients: More robust attribution method
- Channel-level aggregation: Aggregate importance across time dimension
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod


class SaliencyMethod(ABC):
    """Base class for saliency computation methods."""
    
    @staticmethod
    def _prepare_target_and_output(model: nn.Module, input_tensor: torch.Tensor,
                                   target: Optional[Union[int, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper method to prepare target and output for gradient computation.
        Handles multi-output models and target formatting.
        
        Args:
            model: Trained model
            input_tensor: Input tensor (batch_size, num_channels, timepoints)
            target: Target class index or tensor. If None, uses predicted class
        
        Returns:
            Tuple of (output_to_use, target_to_use) where:
            - output_to_use: Output tensor to compute gradients for
            - target_to_use: Target tensor properly formatted
        """
        # Forward pass
        output = model(input_tensor)
        
        # Handle multi-output models and determine target
        if isinstance(output, dict):
            # Multi-output model: use first available output
            # For now, prefer 'gender' then 'age', but can be extended
            output_to_use = output.get('gender', output.get('age', list(output.values())[0]))
            
            # Determine target
            if target is None:
                # Use predicted class
                target_to_use = output_to_use.argmax(dim=1)
            elif isinstance(target, dict):
                # Target is a dict, extract corresponding value
                target_to_use = target.get('gender', target.get('age', list(target.values())[0]))
            else:
                target_to_use = target
        else:
            # Single-output model
            output_to_use = output
            
            # Determine target
            if target is None:
                # Use predicted class
                target_to_use = output_to_use.argmax(dim=1)
            else:
                target_to_use = target
        
        # Ensure target is proper format and on correct device
        if isinstance(target_to_use, int):
            target_to_use = torch.tensor([target_to_use] * input_tensor.size(0), 
                                        device=input_tensor.device, dtype=torch.long)
        elif isinstance(target_to_use, torch.Tensor):
            # Ensure tensor is on correct device
            if target_to_use.device != input_tensor.device:
                target_to_use = target_to_use.to(input_tensor.device)
            # Ensure proper dimensions
            if target_to_use.dim() == 0:
                target_to_use = target_to_use.unsqueeze(0)
            # For classification tasks, ensure dtype is long (for indexing)
            # For regression, keep original dtype (float)
            # We'll determine this in _get_target_output based on output dimensions
        
        return output_to_use, target_to_use
    
    @staticmethod
    def _get_target_output(output_to_use: torch.Tensor, target_to_use: torch.Tensor) -> torch.Tensor:
        """
        Extract target output for gradient computation.
        
        Args:
            output_to_use: Output tensor
            target_to_use: Target tensor
        
        Returns:
            Target output tensor for gradient computation
        
        Note:
            - For classification: output shape is (batch_size, num_classes)
            - For regression: output shape is (batch_size, 1) or (batch_size,)
        """
        # Determine if this is classification or regression
        # Classification: (batch_size, num_classes) where num_classes > 1
        # Regression: (batch_size, 1) or (batch_size,)
        is_classification = (output_to_use.dim() > 1 and output_to_use.size(1) > 1)
        
        if is_classification:
            # Classification: select target class logit
            # output_to_use: (batch_size, num_classes)
            # target_to_use: (batch_size,) with class indices
            # Ensure target_to_use is long dtype for indexing
            if target_to_use.dtype != torch.long:
                target_to_use = target_to_use.long()
            
            # Validate target indices are within valid range
            num_classes = output_to_use.size(1)
            if (target_to_use < 0).any() or (target_to_use >= num_classes).any():
                invalid_indices = target_to_use[(target_to_use < 0) | (target_to_use >= num_classes)]
                min_target = target_to_use.min().item()
                max_target = target_to_use.max().item()
                raise ValueError(
                    f"Target indices out of range. "
                    f"Model has {num_classes} classes (valid indices: 0-{num_classes-1}), "
                    f"but targets contain values in range [{min_target}, {max_target}]. "
                    f"Invalid indices: {invalid_indices.tolist()}. "
                    f"This may indicate a mismatch between model output and target labels."
                )
            
            batch_indices = torch.arange(output_to_use.size(0), device=output_to_use.device)
            target_output = output_to_use[batch_indices, target_to_use]
        else:
            # Regression: use output directly
            # output_to_use: (batch_size,) or (batch_size, 1)
            # For batch_size=1: (1,) or (1, 1)
            # squeeze() ensures we get (1,) or scalar, both work for backward()
            if output_to_use.dim() == 2 and output_to_use.size(1) == 1:
                # Shape (batch_size, 1) -> (batch_size,)
                target_output = output_to_use.squeeze(1)
            else:
                # Shape (batch_size,) -> keep as is
                target_output = output_to_use
        
        return target_output
    
    @abstractmethod
    def compute(self, model: nn.Module, input_tensor: torch.Tensor, 
                target: Optional[Union[int, torch.Tensor]] = None) -> torch.Tensor:
        """
        Compute saliency map for given input.
        
        Args:
            model: Trained model (should be in eval mode)
            input_tensor: Input EEG data tensor of shape (batch_size, num_channels, timepoints)
            target: Target class index or tensor for gradient computation
                   If None, uses predicted class
        
        Returns:
            Saliency map tensor of same shape as input_tensor
        """
        pass


class VanillaGradients(SaliencyMethod):
    """
    Vanilla gradient-based saliency method.
    Computes gradient of output w.r.t. input features.
    """
    
    def __init__(self, abs_gradients: bool = True):
        """
        Args:
            abs_gradients: If True, return absolute values of gradients
        """
        self.abs_gradients = abs_gradients
    
    def compute(self, model: nn.Module, input_tensor: torch.Tensor,
                target: Optional[Union[int, torch.Tensor]] = None) -> torch.Tensor:
        """
        Compute vanilla gradient saliency.
        
        Args:
            model: Trained model
            input_tensor: Input tensor (batch_size, num_channels, timepoints)
            target: Target class index or tensor. If None, uses predicted class
        
        Returns:
            Saliency map (batch_size, num_channels, timepoints)
        """
        # Ensure input requires gradient
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        model.eval()
        
        # Prepare target and output using helper method
        output_to_use, target_to_use = SaliencyMethod._prepare_target_and_output(model, input_tensor, target)
        
        # Get target output for gradient computation
        target_output = SaliencyMethod._get_target_output(output_to_use, target_to_use)
        
        # Backward pass
        model.zero_grad()
        target_output.sum().backward()
        
        # Get gradients
        gradients = input_tensor.grad
        
        if self.abs_gradients:
            gradients = torch.abs(gradients)
        
        return gradients.detach()


class IntegratedGradients(SaliencyMethod):
    """
    Integrated Gradients method for more robust attribution.
    Integrates gradients along the path from baseline to input.
    
    Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017
    """
    
    def __init__(self, num_steps: int = 50, baseline: Optional[torch.Tensor] = None,
                 abs_gradients: bool = True):
        """
        Args:
            num_steps: Number of steps for integration
            baseline: Baseline input (usually zeros or mean). If None, uses zeros
            abs_gradients: If True, return absolute values
        """
        self.num_steps = num_steps
        self.baseline = baseline
        self.abs_gradients = abs_gradients
    
    def compute(self, model: nn.Module, input_tensor: torch.Tensor,
                target: Optional[Union[int, torch.Tensor]] = None) -> torch.Tensor:
        """
        Compute integrated gradients saliency.
        
        Args:
            model: Trained model
            input_tensor: Input tensor (batch_size, num_channels, timepoints)
            target: Target class index or tensor. If None, uses predicted class
        
        Returns:
            Saliency map (batch_size, num_channels, timepoints)
        """
        device = input_tensor.device
        batch_size = input_tensor.size(0)
        
        # Set baseline (default: zeros)
        if self.baseline is None:
            baseline = torch.zeros_like(input_tensor)
        else:
            baseline = self.baseline.to(device)
            if baseline.size(0) != batch_size:
                # Broadcast if needed
                baseline = baseline.expand(batch_size, -1, -1)
        
        # Determine target from original input first (if not provided)
        # This ensures we use a consistent target throughout the integration path
        if target is None:
            with torch.no_grad():
                original_output = model(input_tensor)
                if isinstance(original_output, dict):
                    original_output_to_use = original_output.get('gender', original_output.get('age', list(original_output.values())[0]))
                else:
                    original_output_to_use = original_output
                # Use predicted class from original input
                target = original_output_to_use.argmax(dim=1)
        
        # Create interpolated inputs
        alphas = torch.linspace(0, 1, self.num_steps, device=device)
        
        # Store gradients
        integrated_gradients = torch.zeros_like(input_tensor)
        
        model.eval()
        
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated = interpolated.clone().detach().requires_grad_(True)
            
            # Forward pass and prepare target/output using helper method
            # Use the fixed target determined from original input
            output_to_use, target_to_use = SaliencyMethod._prepare_target_and_output(model, interpolated, target)
            
            # Get target output for gradient computation
            target_output = SaliencyMethod._get_target_output(output_to_use, target_to_use)
            
            # Backward pass
            model.zero_grad()
            target_output.sum().backward()
            
            # Accumulate gradients
            gradients = interpolated.grad
            if self.abs_gradients:
                gradients = torch.abs(gradients)
            
            integrated_gradients += gradients
        
        # Average and multiply by (input - baseline)
        integrated_gradients = integrated_gradients / self.num_steps
        integrated_gradients = integrated_gradients * (input_tensor - baseline)
        
        return integrated_gradients.detach()


def aggregate_channel_importance(saliency_map: torch.Tensor, 
                                aggregation: str = 'mean') -> torch.Tensor:
    """
    Aggregate saliency values across time dimension to get channel-level importance.
    
    Args:
        saliency_map: Saliency map tensor (batch_size, num_channels, timepoints)
        aggregation: Aggregation method ('mean', 'sum', 'max', 'l2_norm')
    
    Returns:
        Channel importance tensor (batch_size, num_channels)
    """
    if aggregation == 'mean':
        return saliency_map.mean(dim=2)
    elif aggregation == 'sum':
        return saliency_map.sum(dim=2)
    elif aggregation == 'max':
        return saliency_map.max(dim=2)[0]
    elif aggregation == 'l2_norm':
        return torch.norm(saliency_map, p=2, dim=2)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")


def compute_batch_saliency(model: nn.Module,
                          data_loader: torch.utils.data.DataLoader,
                          saliency_method: SaliencyMethod,
                          target_key: str = '',
                          max_samples: Optional[int] = None,
                          device: Optional[torch.device] = None,
                          device_preference: str = 'auto',
                          prediction_type: str = 'classification') -> Dict[str, np.ndarray]:
    """
    Compute saliency maps for a batch of samples.
    
    Args:
        model: Trained model (may be wrapped with DataParallel)
        data_loader: DataLoader providing batches
        saliency_method: Saliency computation method
        target_key: Key for target labels ('gender', 'age', 'combined')
        max_samples: Maximum number of samples to process (None = all)
        device: Device to use (None = auto-detect from device_preference)
        device_preference: Device preference ('auto', 'cuda', 'cpu')
        prediction_type: Type of prediction ('classification' or 'regression')
    
    Returns:
        Dictionary containing:
        - 'saliency_maps': (num_samples, num_channels, timepoints) saliency maps
        - 'channel_importance': (num_samples, num_channels) channel-level importance
        - 'labels': (num_samples,) true labels
        - 'predictions': (num_samples,) model predictions
    
    Raises:
        ValueError: If target_key is empty
        RuntimeError: If CUDA is requested but not available
    """
    # Import here to avoid circular imports
    from CNN.gpu_utils import get_device
    from data_processing.constants import AGE_CLASS_1_MAX, AGE_CLASS_2_MAX
    
    # Validate target_key is not empty
    if not target_key:
        raise ValueError(
            "target_key cannot be empty. "
            "Must be one of: 'gender', 'age', 'combined'. "
            "Ensure analyzer is initialized with a valid target_type."
        )
    
    if device is None:
        device = get_device(device_preference)
    
    # Ensure model is on the correct device
    # Note: Model may already be wrapped with DataParallel, which is fine
    model.to(device)
    model.eval()
    
    all_saliency_maps = []
    all_channel_importance = []
    all_labels = []
    all_predictions = []
    all_task_types = []  # Track task types for per-task analysis
    
    num_samples = 0
    
    for batch_idx, batch in enumerate(data_loader):
        # Early exit: if we've already processed enough samples, stop loading more batches
        # This prevents unnecessary data loading and processing after reaching the limit
        if max_samples is not None and num_samples >= max_samples:
            break
        
        # Get input and labels
        # Validate batch structure
        if 'eeg_data' not in batch:
            raise KeyError("Batch missing 'eeg_data' key. Check dataset configuration.")
        
        inputs = batch['eeg_data'].to(device)
        
        # Get labels with explicit error handling
        if target_key not in batch:
            raise KeyError(
                f"Batch missing '{target_key}' key. "
                f"Available keys: {list(batch.keys())}. "
                f"Ensure dataset is configured with target_type='{target_key}'."
            )
        labels = batch[target_key].to(device)
        
        # Convert raw age values to class indices for age classification models
        # The dataset provides raw age values, but classification models need class indices (0, 1, 2)
        if target_key == 'age' and prediction_type == 'classification':
            # Convert raw age values to class indices using constants from config
            # Age classes: 0 (<AGE_CLASS_1_MAX), 1 (AGE_CLASS_1_MAX-AGE_CLASS_2_MAX), 2 (>AGE_CLASS_2_MAX)
            labels_float = labels.float()  # Ensure float for comparison
            age_classes = torch.zeros_like(labels, dtype=torch.long)
            age_classes[labels_float < AGE_CLASS_1_MAX] = 0
            age_classes[(labels_float >= AGE_CLASS_1_MAX) & (labels_float <= AGE_CLASS_2_MAX)] = 1
            age_classes[labels_float > AGE_CLASS_2_MAX] = 2
            labels = age_classes
        
        # Get task types (required for per-task analysis)
        # task_types is a list of strings ('active' or 'passive'), not a tensor
        if 'task_types' not in batch:
            raise KeyError(
                "Batch missing 'task_types' key. "
                "This should always be present from EEGDataset. "
                "Check dataset configuration."
            )
        task_types = batch['task_types']
        
        # Validate task_types: should be list of 'active' or 'passive' strings
        if not isinstance(task_types, list):
            raise TypeError(
                f"task_types should be a list, got {type(task_types)}. "
                "Check dataset collate function."
            )
        
        # Validate that all task_types are 'active' or 'passive'
        valid_task_types = {'active', 'passive'}
        invalid_types = [tt for tt in task_types if tt not in valid_task_types]
        if invalid_types:
            raise ValueError(
                f"Invalid task_types found: {set(invalid_types)}. "
                f"task_types must be 'active' or 'passive', got: {set(task_types)}. "
                "Check dataset configuration."
            )
        
        # Validate length matches batch size
        if len(task_types) != inputs.size(0):
            raise ValueError(
                f"task_types length ({len(task_types)}) does not match batch size ({inputs.size(0)}). "
                "Check dataset collate function."
            )
        
        # Determine batch size (may be smaller for last batch)
        batch_size = inputs.size(0)
        
        # Truncate batch if we're close to max_samples limit
        # This handles the case where the current batch would exceed max_samples
        if max_samples is not None:
            remaining = max_samples - num_samples
            if remaining < batch_size:
                # Only process the remaining samples needed to reach max_samples
                inputs = inputs[:remaining]
                labels = labels[:remaining]
                task_types = task_types[:remaining]
                batch_size = remaining
        
        # Compute saliency for each sample in batch
        # Note: We need gradients for saliency computation
        batch_saliency = []
        batch_channel_importance = []
        batch_predictions = []
        
        for i in range(batch_size):
            sample_input = inputs[i:i+1]  # Keep batch dimension: (1, num_channels, timepoints)
            # Extract label for this sample
            # For classification: labels[i:i+1] gives shape (1,) - correct
            # For regression: labels[i:i+1] gives shape (1,) - correct (float32)
            sample_label = labels[i:i+1]
            
            # Compute saliency (this requires gradients)
            # sample_label shape: (1,) - will be handled correctly by _prepare_target_and_output
            saliency = saliency_method.compute(model, sample_input, target=sample_label)
            
            # Aggregate to channel level
            channel_imp = aggregate_channel_importance(saliency, aggregation='mean')
            
            # Get prediction (no gradients needed here)
            with torch.no_grad():
                output = model(sample_input)
                if isinstance(output, dict):
                    # For dict outputs, target_key must be present
                    # Do not silently fall back to other keys - this could lead to incorrect predictions
                    if target_key not in output:
                        raise KeyError(
                            f"Model output dict missing required key '{target_key}'. "
                            f"Available keys: {list(output.keys())}. "
                            f"Model output structure does not match target_type='{target_key}'. "
                            f"Ensure model type matches the target_type used in analyzer initialization."
                        )
                    pred = output[target_key]
                else:
                    # For single tensor outputs (e.g., gender_cnn, age_cnn, combined_cnn), use directly
                    pred = output
                
                # Extract prediction value
                # For classification: pred shape is (1, num_classes) -> use argmax
                # For regression: pred shape is (1, 1) -> use value directly
                if pred.dim() > 1:
                    if pred.size(1) == 1:
                        # Regression: shape (1, 1) -> extract value directly
                        pred_class = pred.squeeze().item()
                    else:
                        # Classification: shape (1, num_classes) -> use argmax
                        pred_class = pred.argmax(dim=1).item()
                else:
                    # Single dimension: shape (1,) -> extract value directly
                    pred_class = pred.item()
            
            batch_saliency.append(saliency.cpu())
            batch_channel_importance.append(channel_imp.cpu())
            batch_predictions.append(pred_class)
        
        # Concatenate batch results
        # Note: batch_saliency, batch_channel_importance, and batch_predictions
        # already contain only the processed samples (handled by batch_size loop)
        all_saliency_maps.extend(batch_saliency)
        all_channel_importance.extend(batch_channel_importance)
        all_predictions.extend(batch_predictions)
        
        # Ensure labels and task_types match the processed batch size
        # (they were already truncated above if needed)
        all_labels.extend(labels.cpu().numpy()[:batch_size])
        all_task_types.extend(task_types[:batch_size])
        
        num_samples += batch_size
        
        if batch_idx % 10 == 0:
            print(f"Processed {num_samples} samples...")
    
    # Handle empty result case (no samples processed)
    if num_samples == 0:
        raise ValueError(
            "No samples were processed. "
            "Check that the data loader is not empty and max_samples is not 0."
        )
    
    # Convert to numpy arrays
    result = {
        'saliency_maps': torch.cat(all_saliency_maps, dim=0).numpy(),
        'channel_importance': torch.cat(all_channel_importance, dim=0).numpy(),
        'labels': np.array(all_labels),
        'predictions': np.array(all_predictions),
        'task_types': np.array(all_task_types)  # Include task types for per-task analysis
    }
    
    return result
