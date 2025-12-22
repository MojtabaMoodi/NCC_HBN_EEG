# ResNet Implementation Details

## Residual Connection Location

The **residual connection (skip connection)** is implemented in the `ResNetBlock` class in `/CNN/resnet/resnet_model.py`.

### Location: Lines 77-114

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    identity = x  # Line 87: Save input for skip connection
    
    # First conv + BN + ReLU
    out = self.conv1(x)
    if self.use_batch_norm:
        out = self.bn1(out)
    out = F.relu(out)
    
    # Second conv + BN
    out = self.conv2(out)
    if self.use_batch_norm:
        out = self.bn2(out)
    
    # Skip connection (Lines 104-108)
    if self.skip_connection is not None:
        identity = self.skip_connection(identity)  # 1x1 conv to match dimensions
        if self.use_batch_norm:
            identity = self.skip_bn(identity)
    
    # RESIDUAL CONNECTION: Add skip connection (Line 111)
    out = out + identity  # ← THIS IS THE RESIDUAL CONNECTION
    out = F.relu(out)     # Final activation
    
    return out
```

### Key Components:

1. **Identity Path** (Line 87): `identity = x` - saves the input
2. **Skip Connection** (Lines 68-75): Optional 1x1 convolution + batch norm to match dimensions when:
   - `stride != 1` (downsampling)
   - `in_channels != out_channels` (channel mismatch)
3. **Residual Addition** (Line 111): `out = out + identity` - **THE CORE RESIDUAL CONNECTION**
4. **Final Activation** (Line 112): ReLU after addition

## Implementation Verification

### ✅ Correct Implementation

The residual connection is **correctly implemented**:

1. **Identity Preservation**: Input is saved before processing (line 87)
2. **Dimension Matching**: Skip connection handles dimension mismatches via 1x1 conv (lines 68-75)
3. **Residual Addition**: `out = out + identity` correctly adds skip connection (line 111)
4. **Post-Addition Activation**: ReLU applied after addition (line 112)

### Architecture Flow:

```
Input (x)
  ↓
[Save identity = x]
  ↓
Conv1 → BN → ReLU
  ↓
Conv2 → BN
  ↓
[Add identity] ← RESIDUAL CONNECTION
  ↓
ReLU
  ↓
Output
```

### When Skip Connection is Used:

- **Stride > 1**: When downsampling (temporal dimension reduction)
- **Channel Mismatch**: When `in_channels != out_channels` (e.g., 64 → 128)

### When Skip Connection is NOT Used:

- **Stride = 1** AND **Same Channels**: Direct identity addition (no 1x1 conv needed)

## ResNet Type Selection

### New Feature: `--resnet_type` Argument

You can now specify ResNet architecture type (18, 34, or 50) when running experiments:

```bash
# ResNet18 (default)
python run_resnet_age.py --mode 2s --epochs 100

# ResNet34
python run_resnet_age.py --mode 2s --epochs 100 --resnet_type 34

# ResNet50
python run_resnet_age.py --mode 2s --epochs 100 --resnet_type 50
```

### Architecture Differences:

| ResNet Type | Blocks | Block Type | Base Channels | Final Channels | Total Layers |
|-------------|--------|------------|---------------|----------------|--------------|
| ResNet18    | 8      | BasicBlock | 64            | 512           | 42           |
| ResNet34    | 16     | BasicBlock | 64            | 512           | 74           |
| ResNet50    | 16     | BottleneckBlock | 64        | 2048          | 108          |

### Model Type Mapping:

- `--resnet_type 18` → `age_resnet` (ResNet18)
- `--resnet_type 34` → `age_resnet34` (ResNet34)
- `--resnet_type 50` → `age_resnet50` (ResNet50)

## Code Organization (DRY Principle)

### Shared Utilities (`utils.py`)

Common functions are centralized to avoid duplication:

- `create_data_config_for_segment_length()`: Shared data configuration
- `get_resnet_model_type()`: Maps ResNet type to model configuration

### Usage:

```python
from utils import create_data_config_for_segment_length, get_resnet_model_type

# Create data config (used by both main.py and run_resnet_age.py)
data_config = create_data_config_for_segment_length('2s', batch_size=256, num_gpus=2)

# Get ResNet model type and config
model_type, config_dict = get_resnet_model_type(34, task='age')
# Returns: ('age_resnet34', {'num_classes': 3})
```

## Best Practices Followed

1. **DRY (Don't Repeat Yourself)**: Shared utilities in `utils.py`
2. **Single Responsibility**: Each class/function has one clear purpose
3. **Type Safety**: Proper type hints and validation
4. **Error Handling**: Clear error messages for invalid inputs
5. **Documentation**: Comprehensive docstrings
6. **Modularity**: Easy to extend with new ResNet types

## File Structure

```
CNN/
├── utils.py                    # Shared utilities (DRY)
├── run_resnet_age.py          # ResNet age experiment script
├── resnet/
│   ├── resnet_model.py        # ResNet implementation
│   │   ├── ResNetBlock        # Residual block (lines 18-114)
│   │   ├── EEGResNet          # Base ResNet class
│   │   ├── EEGResNet18        # ResNet18
│   │   ├── EEGResNet34        # ResNet34
│   │   ├── EEGResNet50        # ResNet50
│   │   ├── EEGAgeResNet       # ResNet18 for age
│   │   ├── EEGAgeResNet34     # ResNet34 for age
│   │   └── EEGAgeResNet50     # ResNet50 for age
│   └── __init__.py
└── models/
    └── model_factory.py       # Model factory with ResNet registration
```

## Example Usage

```bash
# Run ResNet34 age classification with 2s segments for 100 epochs
python run_resnet_age.py --mode 2s --resnet_type 34 --epochs 100 --learning_rate 0.0001

# Run ResNet50 age classification with 1s segments
python run_resnet_age.py --mode 1s --resnet_type 50 --epochs 50
```

## Prerequisites

**Note**: The ResNet module is required. If the ResNet module is not available, the ModelFactory will raise an `ImportError` when attempting to import ResNet models.

## Summary

- ✅ **Residual Connection**: Correctly implemented at line 111 in `ResNetBlock.forward()`
- ✅ **ResNet Type Selection**: Added `--resnet_type` argument (18, 34, 50)
- ✅ **Architecture**: ResNet18 (8 blocks), ResNet34 (16 blocks), ResNet50 (16 BottleneckBlocks)
- ✅ **DRY Principle**: Shared utilities in `utils.py`
- ✅ **Best Practices**: Modular, documented, type-safe code
- ✅ **Required Module**: ResNet is required (raises ImportError if missing)

