# ResNet Layer Count Analysis

This document provides a detailed breakdown of the number of layers in ResNet18, ResNet34, and ResNet50 architectures as implemented in this codebase.

**Note**: The ResNet module is required. If the ResNet module is not available, the ModelFactory will raise an `ImportError` when attempting to import ResNet models.

## Layer Counting Methodology

We count **trainable layers** (convolutional and fully connected layers) and **normalization layers** (batch normalization). We exclude:
- Activation functions (ReLU) - these are operations, not layers
- Dropout layers - these are regularization techniques, not learnable layers
- Pooling operations (adaptive average pooling) - these are operations, not layers

## Architecture Overview

All ResNet architectures share:
- **Initial Convolution**: 1 conv layer + 1 batch norm layer
- **Fully Connected Layers**: 2 FC layers (fc1: final_channels → 128, fc2: 128 → num_classes)

The difference is in the **ResNet blocks**:
- **ResNet18 & ResNet34**: Use `BasicBlock` (2 conv layers per block)
- **ResNet50**: Uses `BottleneckBlock` (3 conv layers per block)

---

## ResNet18 Layer Count

**Configuration**: `layers=[2, 2, 2, 2]` → 8 BasicBlocks total

### BasicBlock Structure:
Each BasicBlock contains:
- `conv1` + `bn1` = 2 layers
- `conv2` + `bn2` = 2 layers
- Skip connection (when needed): `skip_connection` conv + `skip_bn` = 2 layers
- **Total per block**: 4 layers (no skip) or 6 layers (with skip)

### Skip Connection Distribution:
- **Stage 1** (64 channels): 2 blocks, no skip connections needed (channels match)
- **Stage 2** (128 channels): 2 blocks, first block has skip connection
- **Stage 3** (256 channels): 2 blocks, first block has skip connection
- **Stage 4** (512 channels): 2 blocks, first block has skip connection

### Layer Breakdown:
1. **Initial Conv**: 1 conv + 1 BN = **2 layers**
2. **ResNet Blocks**:
   - Stage 1: 2 blocks × 4 layers = 8 layers
   - Stage 2: 1 block × 6 layers + 1 block × 4 layers = 10 layers
   - Stage 3: 1 block × 6 layers + 1 block × 4 layers = 10 layers
   - Stage 4: 1 block × 6 layers + 1 block × 4 layers = 10 layers
   - **Total blocks**: 8 + 10 + 10 + 10 = **38 layers**
3. **FC Layers**: fc1 + fc2 = **2 layers**

### **Total ResNet18 Layers: 2 + 38 + 2 = 42 layers**

---

## ResNet34 Layer Count

**Configuration**: `layers=[3, 4, 6, 3]` → 16 BasicBlocks total

### Skip Connection Distribution:
- **Stage 1** (64 channels): 3 blocks, no skip connections needed
- **Stage 2** (128 channels): 4 blocks, first block has skip connection
- **Stage 3** (256 channels): 6 blocks, first block has skip connection
- **Stage 4** (512 channels): 3 blocks, first block has skip connection

### Layer Breakdown:
1. **Initial Conv**: 1 conv + 1 BN = **2 layers**
2. **ResNet Blocks**:
   - Stage 1: 3 blocks × 4 layers = 12 layers
   - Stage 2: 1 block × 6 layers + 3 blocks × 4 layers = 18 layers
   - Stage 3: 1 block × 6 layers + 5 blocks × 4 layers = 26 layers
   - Stage 4: 1 block × 6 layers + 2 blocks × 4 layers = 14 layers
   - **Total blocks**: 12 + 18 + 26 + 14 = **70 layers**
3. **FC Layers**: fc1 + fc2 = **2 layers**

### **Total ResNet34 Layers: 2 + 70 + 2 = 74 layers**

---

## ResNet50 Layer Count

**Configuration**: `layers=[3, 4, 6, 3]` → 16 BottleneckBlocks total

### BottleneckBlock Structure:
Each BottleneckBlock contains:
- `conv1` (1×1) + `bn1` = 2 layers
- `conv2` (3×3) + `bn2` = 2 layers
- `conv3` (1×1) + `bn3` = 2 layers
- Skip connection (when needed): `skip_connection` conv + `skip_bn` = 2 layers
- **Total per block**: 6 layers (no skip) or 8 layers (with skip)

### Skip Connection Distribution:
- **Stage 1** (256 channels after expansion): 3 blocks, first block has skip connection (channels change from 64 to 256)
- **Stage 2** (512 channels after expansion): 4 blocks, first block has skip connection
- **Stage 3** (1024 channels after expansion): 6 blocks, first block has skip connection
- **Stage 4** (2048 channels after expansion): 3 blocks, first block has skip connection

### Layer Breakdown:
1. **Initial Conv**: 1 conv + 1 BN = **2 layers**
2. **ResNet Blocks**:
   - Stage 1: 1 block × 8 layers + 2 blocks × 6 layers = 20 layers
   - Stage 2: 1 block × 8 layers + 3 blocks × 6 layers = 26 layers
   - Stage 3: 1 block × 8 layers + 5 blocks × 6 layers = 38 layers
   - Stage 4: 1 block × 8 layers + 2 blocks × 6 layers = 20 layers
   - **Total blocks**: 20 + 26 + 38 + 20 = **104 layers**
3. **FC Layers**: fc1 + fc2 = **2 layers**

### **Total ResNet50 Layers: 2 + 104 + 2 = 108 layers**

---

## Summary Table

| Architecture | Blocks | Block Type | Conv Layers per Block | Total Layers |
|-------------|--------|------------|---------------------|--------------|
| **ResNet18** | 8 | BasicBlock | 2 | **42 layers** |
| **ResNet34** | 16 | BasicBlock | 2 | **74 layers** |
| **ResNet50** | 16 | BottleneckBlock | 3 | **108 layers** |

### Breakdown by Component:

| Component | ResNet18 | ResNet34 | ResNet50 |
|-----------|----------|----------|----------|
| Initial Conv + BN | 2 | 2 | 2 |
| ResNet Blocks | 38 | 70 | 104 |
| FC Layers | 2 | 2 | 2 |
| **Total** | **42** | **74** | **108** |

---

## Notes

1. **Batch Normalization**: All architectures use batch normalization, which adds one BN layer per conv layer.

2. **Skip Connections**: Skip connections add 2 layers (1 conv + 1 BN) when dimensions don't match. This occurs:
   - In ResNet18/34: At the first block of stages 2, 3, and 4
   - In ResNet50: At the first block of all 4 stages (due to channel expansion)

3. **Channel Expansion**: ResNet50 uses BottleneckBlocks with `expansion=4`, meaning output channels = base_channels × 4. This is why skip connections are needed in stage 1 of ResNet50.

4. **Fully Connected Layers**: All architectures use the same FC structure:
   - `fc1`: final_channels → 128
   - `fc2`: 128 → num_classes (2 for gender, 3 for age)

5. **Layer Count vs. Parameter Count**: The number of layers doesn't directly correlate with parameter count. ResNet50 has significantly more parameters than ResNet34 despite having the same number of blocks, due to the larger channel dimensions in BottleneckBlocks.

