# GPU Setup Changes Summary

## Overview
This document summarizes the changes made to enable proper GPU utilization across the codebase, supporting 4 GPUs when available, otherwise falling back to 1 GPU. All GPU-related code now follows best practices with explicit error handling and no silent fallbacks.

## Key Changes

### 1. New GPU Utility Module (`CNN/gpu_utils.py`)
Created a centralized GPU utility module with the following functions:
- `detect_available_gpus()`: Detects number of available GPUs
- `get_device(device_preference)`: Gets appropriate device with explicit error handling
- `determine_num_gpus_to_use(requested_num_gpus, available_gpus)`: Determines GPUs to use with validation
- `setup_model_for_gpus(model, num_gpus, device)`: Sets up model for multi-GPU training
- `get_underlying_model(model)`: Unwraps DataParallel and torch.compile wrappers
- `print_gpu_info(...)`: Prints GPU configuration information

**Key Features:**
- Explicit error handling (no silent fallbacks)
- Raises RuntimeError if GPU configuration is invalid
- Raises ValueError if num_gpus is invalid
- Centralized logic for DRY principle

### 2. Updated Trainer (`CNN/trainer.py`)
- Uses `gpu_utils` for all GPU-related operations
- Accepts `device_preference` parameter ('auto', 'cuda', 'cpu')
- Explicit GPU detection and setup with error handling
- Prints GPU information on initialization
- Default `num_gpus` changed from 2 to 1

### 3. Updated Evaluator (`CNN/evaluator.py`)
- Uses `gpu_utils` for model unwrapping
- Accepts `device_preference` parameter
- Properly handles DataParallel-wrapped models
- Explicit device setup with error handling

### 4. Updated Experiment (`CNN/experiment.py`)
- Passes `device_preference` from SystemConfig to trainer and evaluator
- Uses `num_gpus` from SystemConfig (defaults to 1)
- Explicit configuration passing

### 5. Updated Main Script (`CNN/main.py`)
- Auto-detects GPUs: uses 4 if available, else uses available GPUs, else 1 (CPU mode)
- `--num_gpus` argument defaults to None (auto-detect)
- Creates SystemConfig with proper GPU settings
- Removed hardcoded GPU fallbacks

### 6. Updated Config (`CNN/config.py`)
- `SystemConfig.num_gpus` default changed from 2 to 1
- Updated docstring to clarify auto-detection behavior

### 7. Updated Saliency Analysis (`saliency_analysis/`)
- `analyzer.py`: Uses `gpu_utils` for model unwrapping and device setup
- `saliency_methods.py`: Updated `compute_batch_saliency` to accept `device_preference`
- `main.py`: Passes `device_preference` to analyzer
- Properly handles DataParallel-wrapped models

## GPU Detection Logic

### Auto-Detection (when `num_gpus` not specified)
1. Detect available GPUs
2. If >= 4 GPUs available → use 4 GPUs
3. Else if >= 1 GPU available → use available GPUs
4. Else → use 1 GPU (CPU mode)

### Explicit Configuration (when `num_gpus` specified)
1. Validate requested number (must be >= 1)
2. Check if CUDA is available (if num_gpus > 1)
3. Check if requested GPUs <= available GPUs
4. Raise explicit errors if configuration is invalid

## Error Handling

All GPU-related functions now raise explicit errors:
- `RuntimeError`: If CUDA is requested but not available
- `RuntimeError`: If more GPUs are requested than available
- `ValueError`: If num_gpus < 1

No silent fallbacks or assumptions are made.

## Saliency Map Logic

The saliency computation logic has been reviewed and is correct:
- Properly handles DataParallel-wrapped models
- Processes samples correctly with gradient computation
- Device handling is explicit and correct
- Model unwrapping works correctly for checkpoint loading

## Usage Examples

### Using 4 GPUs (auto-detect)
```python
# In main.py or experiment script
system_config = SystemConfig(num_gpus=None)  # Will auto-detect
# Or via command line:
# python CNN/main.py --num_gpus 4
```

### Using Explicit Number of GPUs
```python
system_config = SystemConfig(num_gpus=4)  # Will use 4 if available, else raise error
```

### Using Single GPU
```python
system_config = SystemConfig(num_gpus=1)  # Will use 1 GPU or CPU
```

## Testing Recommendations

1. Test with 4 GPUs available: Should use all 4
2. Test with 2 GPUs available: Should use 2 (if requested) or raise error if 4 requested
3. Test with 1 GPU available: Should use 1 (if requested) or raise error if more requested
4. Test with no GPUs: Should use CPU (if num_gpus=1) or raise error (if num_gpus>1)
5. Test saliency analysis with DataParallel models: Should work correctly

## Backward Compatibility

- Default `num_gpus` changed from 2 to 1 (safer default)
- Auto-detection now prefers 4 GPUs if available (better utilization)
- All existing code should work, but may use different number of GPUs by default

## Code Quality Improvements

1. **DRY Principle**: Centralized GPU logic in `gpu_utils.py`
2. **Explicit Error Handling**: No silent fallbacks
3. **Configuration-Driven**: Uses config files, not hardcoded values
4. **Best Practices**: Follows PyTorch best practices for multi-GPU training
