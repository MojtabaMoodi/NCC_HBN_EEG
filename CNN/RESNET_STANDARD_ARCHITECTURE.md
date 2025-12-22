# Standard ResNet Architecture Implementation

**Note**: The ResNet module is required. If the ResNet module is not available, the ModelFactory will raise an `ImportError` when attempting to import ResNet models.

## ✅ Implementation Status

The ResNet implementation now follows the **standard ResNet architecture** as specified:

### ResNet-18
- ✅ **8 BasicBlocks** arranged as **[2,2,2,2]** across 4 stages
- ✅ Stage channels: 64, 128, 256, 512
- ✅ Final feature size: **512 channels**

### ResNet-34
- ✅ **16 BasicBlocks** arranged as **[3,4,6,3]** across 4 stages
- ✅ Stage channels: 64, 128, 256, 512
- ✅ Final feature size: **512 channels**

### ResNet-50
- ✅ **16 BottleneckBlocks** arranged as **[3,4,6,3]** across 4 stages
- ✅ Uses **BottleneckBlock** (3 conv layers: 1x1, 3x3, 1x1) instead of BasicBlock
- ✅ Stage channels: 256, 512, 1024, 2048 (with expansion=4)
- ✅ Final feature size: **2048 channels**

## Architecture Details

### BasicBlock (Used in ResNet-18 and ResNet-34)
```
Input
  ↓
Conv2d (3x3) → BN → ReLU
  ↓
Conv2d (3x3) → BN
  ↓
[Add Skip Connection]
  ↓
ReLU
  ↓
Output
```

**Structure**: 2 convolutional layers per block

### BottleneckBlock (Used in ResNet-50)
```
Input
  ↓
Conv2d (1x1) → BN → ReLU  [Reduce channels]
  ↓
Conv2d (3x3) → BN → ReLU  [Extract features]
  ↓
Conv2d (1x1) → BN         [Expand channels]
  ↓
[Add Skip Connection]
  ↓
ReLU
  ↓
Output
```

**Structure**: 3 convolutional layers per block
**Expansion**: Output channels = base_channels × 4

## Stage-by-Stage Breakdown

### ResNet-18: [2,2,2,2] = 8 BasicBlocks

| Stage | Base Channels | Blocks | First Block Stride | Final Channels |
|-------|---------------|--------|-------------------|----------------|
| 1     | 64            | 2      | 1                 | 64             |
| 2     | 128           | 2      | 2                 | 128            |
| 3     | 256           | 2      | 2                 | 256            |
| 4     | 512           | 2      | 2                 | **512**        |

**Total**: 8 BasicBlocks, Final: 512 channels

### ResNet-34: [3,4,6,3] = 16 BasicBlocks

| Stage | Base Channels | Blocks | First Block Stride | Final Channels |
|-------|---------------|--------|-------------------|----------------|
| 1     | 64            | 3      | 1                 | 64             |
| 2     | 128           | 4      | 2                 | 128            |
| 3     | 256           | 6      | 2                 | 256            |
| 4     | 512           | 3      | 2                 | **512**        |

**Total**: 16 BasicBlocks, Final: 512 channels

### ResNet-50: [3,4,6,3] = 16 BottleneckBlocks

| Stage | Base Channels | Blocks | Expansion | Final Channels |
|-------|---------------|--------|-----------|----------------|
| 1     | 64            | 3      | ×4        | **256**        |
| 2     | 128           | 4      | ×4        | **512**        |
| 3     | 256           | 6      | ×4        | **1024**       |
| 4     | 512           | 3      | ×4        | **2048**       |

**Total**: 16 BottleneckBlocks, Final: **2048 channels**

## Key Differences

### ResNet-18 vs ResNet-34
- **Same block type**: Both use BasicBlock (2 conv layers)
- **Same final channels**: Both produce 512 channels
- **Difference**: ResNet-34 has **8 more blocks** (16 vs 8)
- **More depth**: ResNet-34 can learn more complex features

### ResNet-34 vs ResNet-50
- **Different block types**: 
  - ResNet-34: BasicBlock (2 conv layers)
  - ResNet-50: BottleneckBlock (3 conv layers)
- **Different final channels**:
  - ResNet-34: 512 channels
  - ResNet-50: **2048 channels** (4× larger)
- **Same block count**: Both have 16 blocks
- **More capacity**: ResNet-50 has much more parameters

## FC Layer Sizes

| Architecture | FC1 Input | FC1 Output | FC2 Output |
|-------------|-----------|------------|------------|
| ResNet-18   | 512       | 128        | num_classes|
| ResNet-34   | 512       | 128        | num_classes|
| ResNet-50   | **2048**  | 128        | num_classes|

## Implementation Location

### Block Definitions
- **BasicBlock**: Lines 18-114 in `resnet_model.py`
- **BottleneckBlock**: Lines 116-195 in `resnet_model.py`

### Architecture Classes
- **EEGResNet18**: Lines 430-446 (uses `layers=[2,2,2,2]`, `block_type=BasicBlock`)
- **EEGResNet34**: Lines 448-464 (uses `layers=[3,4,6,3]`, `block_type=BasicBlock`)
- **EEGResNet50**: Lines 469-495 (uses `layers=[3,4,6,3]`, `block_type=BottleneckBlock`)

### Stage Building
- **`_build_conv_layers()`**: Lines 286-330
  - Creates stages based on `layers` parameter
  - First block in each stage (except stage 0) uses stride=2 for downsampling
  - Channels double between stages

## Verification

You can verify the architecture:

```python
from resnet.resnet_model import EEGResNet18, EEGResNet34, EEGResNet50

model18 = EEGResNet18()
model34 = EEGResNet34()
model50 = EEGResNet50()

print(f"ResNet18: {model18.layers} = {sum(model18.layers)} blocks, "
      f"{model18.block_type.__name__}, final_channels={model18.final_channels}")
# Output: ResNet18: [2, 2, 2, 2] = 8 blocks, BasicBlock, final_channels=512

print(f"ResNet34: {model34.layers} = {sum(model34.layers)} blocks, "
      f"{model34.block_type.__name__}, final_channels={model34.final_channels}")
# Output: ResNet34: [3, 4, 6, 3] = 16 blocks, BasicBlock, final_channels=512

print(f"ResNet50: {model50.layers} = {sum(model50.layers)} blocks, "
      f"{model50.block_type.__name__}, final_channels={model50.final_channels}")
# Output: ResNet50: [3, 4, 6, 3] = 16 blocks, BottleneckBlock, final_channels=2048
```

## Summary

✅ **ResNet-18**: 8 BasicBlocks in [2,2,2,2] stages → 512 channels  
✅ **ResNet-34**: 16 BasicBlocks in [3,4,6,3] stages → 512 channels  
✅ **ResNet-50**: 16 BottleneckBlocks in [3,4,6,3] stages → 2048 channels  

The implementation now correctly follows the standard ResNet architecture!

