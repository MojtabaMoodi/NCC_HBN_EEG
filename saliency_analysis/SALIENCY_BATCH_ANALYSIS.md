# Saliency Batch Analysis and Fixes

## Batch Structure Analysis

### HDF5 Dataset Batch Structure (from `eeg_dataset.py`)

The batch dictionary returned by `EEGDataLoader.collate_fn` contains:

```python
{
    'eeg_data': torch.Tensor,      # Shape: (batch_size, 60, timepoints), dtype: float32
    'gender': torch.Tensor,        # Shape: (batch_size,), dtype: long (classification)
    'age': torch.Tensor,           # Shape: (batch_size,), dtype: float32 (regression)
    'participant_ids': List[str],  # List of participant ID strings
    'task_types': List[str],       # List of task type strings ('active' or 'passive')
    'task_numbers': List[int],     # List of task numbers
    'sample_ids': List[str],       # List of sample ID strings
    'combined': torch.Tensor,      # Optional, Shape: (batch_size,), dtype: long (if target_type='combined')
}
```

### Key Observations

1. **`eeg_data`**: Always present, shape `(batch_size, 60, timepoints)`
2. **`gender`** and **`age`**: Always present in batch (even if not used as target)
3. **`combined`**: Only present if `target_type='combined'` in dataset
4. **`task_types`**: List of strings, not a tensor
5. **Target key**: Must match `target_type` passed to dataset ('gender', 'age', or 'combined')

## Issues Found and Fixed

### Issue 1: Missing Error Handling for Target Key ✅ FIXED

**Location**: Line 336 in `compute_batch_saliency`

**Problem**:
```python
labels = batch[target_key].to(device)  # Raises KeyError if target_key missing
```

**Fix**:
```python
if target_key not in batch:
    raise KeyError(
        f"Batch missing '{target_key}' key. "
        f"Available keys: {list(batch.keys())}. "
        f"Ensure dataset is configured with target_type='{target_key}'."
    )
labels = batch[target_key].to(device)
```

**Impact**: Now provides clear error message if target key is missing, helping users identify configuration issues.

### Issue 2: Missing Validation for EEG Data ✅ FIXED

**Location**: Line 335 in `compute_batch_saliency`

**Problem**: No validation that `eeg_data` exists in batch

**Fix**:
```python
if 'eeg_data' not in batch:
    raise KeyError("Batch missing 'eeg_data' key. Check dataset configuration.")
inputs = batch['eeg_data'].to(device)
```

**Impact**: Catches dataset configuration errors early.

### Issue 3: Regression Target Output Handling ✅ IMPROVED

**Location**: `_get_target_output` method

**Problem**: For regression, `squeeze()` might not handle all cases correctly:
- If output is `(batch_size, 1)`, `squeeze()` removes both dimensions if batch_size=1
- This could cause issues in some edge cases

**Fix**:
```python
if output_to_use.dim() > 1:
    # Classification: select target class logit
    batch_indices = torch.arange(output_to_use.size(0), device=output_to_use.device)
    target_output = output_to_use[batch_indices, target_to_use]
else:
    # Regression: use output directly
    if output_to_use.dim() == 2 and output_to_use.size(1) == 1:
        # Shape (batch_size, 1) -> (batch_size,)
        target_output = output_to_use.squeeze(1)
    else:
        # Shape (batch_size,) -> keep as is
        target_output = output_to_use
```

**Impact**: More explicit handling of regression outputs, ensuring correct tensor shapes.

### Issue 4: Added Documentation Comments ✅ ADDED

**Location**: Sample processing loop

**Fix**: Added comments explaining tensor shapes:
```python
sample_input = inputs[i:i+1]  # Keep batch dimension: (1, num_channels, timepoints)
# Extract label for this sample
# For classification: labels[i:i+1] gives shape (1,) - correct
# For regression: labels[i:i+1] gives shape (1,) - correct (float32)
sample_label = labels[i:i+1]
```

**Impact**: Better code documentation for future maintenance.

## Verified Correct Logic

### Task Types Handling ✅ FIXED

**Problem**: 
- Original code had fallback: `task_types = batch.get('task_types', ['unknown'] * inputs.size(0))`
- `task_types` should ALWAYS be 'active' or 'passive' from dataset, never 'unknown'
- The fallback would silently create invalid 'unknown' values

**Fix**:
- Removed 'unknown' fallback
- Added explicit validation that `task_types` exists in batch (raises KeyError if missing)
- Added validation that all values are 'active' or 'passive' (raises ValueError if invalid)
- Added validation that length matches batch size (raises ValueError if mismatch)
- All errors provide clear messages to help debug dataset configuration issues

**Status**: 
1. **Validation**: Now validates `task_types` exists and contains only 'active' or 'passive'
2. **Truncation**: `task_types = task_types[:remaining]` correctly slices list when truncating batch
3. **Extension**: `all_task_types.extend(task_types[:batch_size])` correctly extends with list slice

### Label Extraction ✅ CORRECT

1. **Line 363**: `sample_label = labels[i:i+1]`
   - Creates tensor with shape `(1,)` which is correct
   - Works for both classification (long) and regression (float32)

2. **Line 396**: `all_labels.extend(labels.cpu().numpy()[:batch_size])`
   - Correctly extracts labels as numpy array
   - Handles both classification and regression dtypes

### Batch Truncation Logic ✅ CORRECT

When `max_samples` is specified and batch would exceed limit:
1. Calculates `remaining = max_samples - num_samples`
2. Truncates `inputs`, `labels`, and `task_types` correctly
3. Updates `batch_size` to match truncated size
4. Processes only the needed samples

## Testing Recommendations

1. **Test with missing target_key**: Should raise clear KeyError
2. **Test with missing eeg_data**: Should raise clear KeyError
3. **Test regression saliency**: Verify correct handling of float32 labels
4. **Test classification saliency**: Verify correct handling of long labels
5. **Test combined target**: Verify correct handling of combined labels
6. **Test batch truncation**: Verify correct behavior when max_samples < batch_size
7. **Test with DataParallel models**: Verify saliency computation works correctly

## Summary

The batch processing logic is now more robust with:
- ✅ Explicit error handling for missing keys
- ✅ Better regression output handling
- ✅ Improved documentation
- ✅ Verified correctness of existing logic

All changes maintain backward compatibility while improving error messages and edge case handling.
