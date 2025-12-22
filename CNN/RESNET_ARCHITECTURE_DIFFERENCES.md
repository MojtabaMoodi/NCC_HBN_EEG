# ResNet Architecture Differences: ResNet18, ResNet34, and ResNet50

**Note**: The ResNet module is required. If the ResNet module is not available, the ModelFactory will raise an `ImportError` when attempting to import ResNet models.

## Summary

The ResNet architectures **ARE different** in implementation. The key differences are:

| Architecture | Blocks | Block Type | Final Channels | FC1 Input | Parameters |
|--------------|--------|------------|----------------|-----------|------------|
| ResNet18     | 8      | BasicBlock | 512            | 512       | Fewest     |
| ResNet34     | 16     | BasicBlock | 512            | 512       | Medium     |
| ResNet50     | 16     | BottleneckBlock | 2048       | 2048      | Most       |

## Detailed Architecture Breakdown

### ResNet18 (8 BasicBlocks: [2,2,2,2])
```
Initial Conv: 1 -> 64 channels
Stage 1: 64 channels, 2 blocks (stride=1)
Stage 2: 128 channels, 2 blocks (first block stride=2)
Stage 3: 256 channels, 2 blocks (first block stride=2)
Stage 4: 512 channels, 2 blocks (first block stride=2)

Final: 512 channels
FC1: 512 -> 128
FC2: 128 -> num_classes
```

### ResNet34 (16 BasicBlocks: [3,4,6,3])
```
Initial Conv: 1 -> 64 channels
Stage 1: 64 channels, 3 blocks (stride=1)
Stage 2: 128 channels, 4 blocks (first block stride=2)
Stage 3: 256 channels, 6 blocks (first block stride=2)
Stage 4: 512 channels, 3 blocks (first block stride=2)

Final: 512 channels
FC1: 512 -> 128
FC2: 128 -> num_classes
```

**Difference from ResNet18**: 
- 8 more blocks (16 vs 8)
- Same final channel count (512)
- More depth allows learning more complex features

### ResNet50 (16 BottleneckBlocks: [3,4,6,3])
```
Initial Conv: 1 -> 64 channels
Stage 1: 256 channels (64×4), 3 BottleneckBlocks (first block stride=2)
Stage 2: 512 channels (128×4), 4 BottleneckBlocks (first block stride=2)
Stage 3: 1024 channels (256×4), 6 BottleneckBlocks (first block stride=2)
Stage 4: 2048 channels (512×4), 3 BottleneckBlocks (first block stride=2)

Final: 2048 channels
FC1: 2048 -> 128
FC2: 128 -> num_classes
```

**Difference from ResNet34**: 
- Same number of blocks (16)
- Uses BottleneckBlock (3 conv layers) instead of BasicBlock (2 conv layers)
- **2048 final channels** (vs 512 for ResNet34) due to expansion=4
- Much larger FC1 input (2048 vs 512)

## Key Differences

### 1. Number of Blocks
- **ResNet18**: 8 BasicBlocks ([2,2,2,2])
- **ResNet34**: 16 BasicBlocks ([3,4,6,3]) - double ResNet18
- **ResNet50**: 16 BottleneckBlocks ([3,4,6,3]) - same count as ResNet34, but different block type

### 2. Block Type
- **ResNet18**: BasicBlock (2 conv layers per block)
- **ResNet34**: BasicBlock (2 conv layers per block)
- **ResNet50**: BottleneckBlock (3 conv layers per block with expansion=4)

### 3. Final Feature Channels
- **ResNet18**: 512 channels
- **ResNet34**: 512 channels (same as ResNet18)
- **ResNet50**: **2048 channels** (4x ResNet18/34 due to expansion)

### 4. Fully Connected Layer Input
- **ResNet18**: FC1 receives 512 features
- **ResNet34**: FC1 receives 512 features (same as ResNet18)
- **ResNet50**: FC1 receives **2048 features** (4x ResNet18/34)

### 4. Model Capacity
- **ResNet18**: Smallest, fastest
- **ResNet34**: Medium capacity (more blocks but same feature size)
- **ResNet50**: **Largest capacity** (more blocks + larger features)

## Why They Look Similar

The class definitions look similar because they inherit from base classes:

```python
class EEGAgeResNet34(EEGResNet34):  # Inherits from EEGResNet34
    # Just sets num_classes=3

class EEGAgeResNet50(EEGResNet50):  # Inherits from EEGResNet50
    # Just sets num_classes=3
```

The actual differences come from the parent classes:
- `EEGResNet34` sets `num_blocks=6`
- `EEGResNet50` sets `num_blocks=8`

The `_build_conv_layers()` method in `EEGResNet` uses `num_blocks` to create different architectures.

## Code Location

The differences are defined in:
- **Line 439**: `EEGResNet18.__init__()` sets `layers=[2,2,2,2]` (8 BasicBlocks)
- **Line 466**: `EEGResNet34.__init__()` sets `layers=[3,4,6,3]` (16 BasicBlocks)
- **Line 495**: `EEGResNet50.__init__()` sets `layers=[3,4,6,3]` with `BottleneckBlock` (16 BottleneckBlocks)

The stage building logic (`_build_conv_layers()`) uses `layers` parameter to determine:
- How many blocks per stage
- Final channel count (which affects FC1 input size)
- Block type (BasicBlock vs BottleneckBlock) determines channel expansion

## Verification

You can verify the differences by checking the model:

```python
from resnet.resnet_model import EEGResNet18, EEGResNet34, EEGResNet50

model18 = EEGResNet18()
model34 = EEGResNet34()
model50 = EEGResNet50()

print(f"ResNet18 layers: {model18.layers} = {sum(model18.layers)} blocks")
# Output: ResNet18 layers: [2, 2, 2, 2] = 8 blocks

print(f"ResNet34 layers: {model34.layers} = {sum(model34.layers)} blocks")
# Output: ResNet34 layers: [3, 4, 6, 3] = 16 blocks

print(f"ResNet50 layers: {model50.layers} = {sum(model50.layers)} blocks")
# Output: ResNet50 layers: [3, 4, 6, 3] = 16 blocks

print(f"ResNet18 final channels: {model18.final_channels}")  # 512
print(f"ResNet34 final channels: {model34.final_channels}")  # 512
print(f"ResNet50 final channels: {model50.final_channels}")  # 2048

print(f"ResNet18 block type: {model18.block_type.__name__}")  # BasicBlock
print(f"ResNet34 block type: {model34.block_type.__name__}")  # BasicBlock
print(f"ResNet50 block type: {model50.block_type.__name__}")  # BottleneckBlock
```

## Conclusion

**ResNet18, ResNet34, and ResNet50 ARE different**:
- **ResNet18 vs ResNet34**: ResNet34 has double the blocks (16 vs 8), but same final channels (512)
- **ResNet34 vs ResNet50**: Same number of blocks (16), but ResNet50 uses BottleneckBlocks with expansion=4, resulting in 4x final channels (2048 vs 512)
- **ResNet50** has significantly more parameters and capacity due to larger channel dimensions
- The differences are in the `layers` parameter and `block_type` passed to the base `EEGResNet` class

