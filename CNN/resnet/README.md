# ResNet Models for EEG Classification

This directory contains ResNet-like CNN architectures adapted for EEG time-series classification tasks.

**Note**: The ResNet module is required. If the ResNet module is not available, the ModelFactory will raise an `ImportError` when attempting to import ResNet models.

## Overview

ResNet (Residual Network) models use skip connections (residual connections) to enable training of deeper networks. These models are adapted for EEG data with the following characteristics:

- **Input**: EEG data with shape `(batch, 60 channels, timepoints)`
  - 1s segments: 200 timepoints
  - 2s segments: 400 timepoints  
  - 4s segments: 800 timepoints
- **Architecture**: ResNet blocks with batch normalization and skip connections
- **Output**: Classification or regression outputs depending on the task

## Available Models

### Base Models

- **`EEGResNet`**: Base ResNet model (configurable number of blocks)
- **`EEGResNet18`**: ResNet-18 architecture (8 BasicBlocks arranged as [2,2,2,2], 512 final channels)
- **`EEGResNet34`**: ResNet-34 architecture (16 BasicBlocks arranged as [3,4,6,3], 512 final channels)
- **`EEGResNet50`**: ResNet-50 architecture (16 BottleneckBlocks arranged as [3,4,6,3], 2048 final channels)

### Task-Specific Models

- **`EEGGenderResNet`**: Gender classification (2 classes: Female/Male)
- **`EEGAgeResNet`**: Age classification (3 classes: <8.5, 8.5-12.5, >12.5 years)
- **`EEGAgeRegressionResNet`**: Age regression (continuous output, normalized to [0, 1])
- **`CombinedResNet`**: Combined age+gender classification (6 classes)
- **`MultiOutputResNet`**: Multi-output model with separate heads for gender and age

## Architecture Details

### ResNet Block Structure

Each ResNet block consists of:
1. **First Convolution**: 3x3 conv with optional stride for downsampling
2. **Batch Normalization**: Applied after each convolution
3. **ReLU Activation**: Non-linearity
4. **Second Convolution**: 3x3 conv
5. **Batch Normalization**: Applied after second convolution
6. **Skip Connection**: Adds input to output (with 1x1 conv if dimensions don't match)
7. **Final ReLU**: Applied after skip connection

### Model Architecture

```
Input (batch, 60, timepoints)
  ↓
Add channel dimension → (batch, 1, 60, timepoints)
  ↓
Initial Conv: (1, 60, 7) → (64, 1, timepoints/2)
  ↓
BatchNorm + ReLU
  ↓
ResNet Block 1: 64 → 64 channels
  ↓
ResNet Block 2: 64 → 128 channels (stride=2)
  ↓
ResNet Block 3: 128 → 128 channels
  ↓
ResNet Block 4: 128 → 256 channels (stride=2)
  ↓
... (more blocks for ResNet34/50)
  ↓
Global Average Pooling → (batch, final_channels)
  ↓
FC1: final_channels → 128
  ↓
Dropout
  ↓
FC2: 128 → num_classes
```

## Usage

### Using ModelFactory

```python
from models.model_factory import ModelFactory

# Create a ResNet-18 model for gender classification
model = ModelFactory.create_model(
    'gender_resnet',
    num_channels=60,
    num_classes=2,
    dropout_rate=0.5,
    use_batch_norm=True
)

# Create a ResNet-34 model for age classification
model = ModelFactory.create_model(
    'age_resnet',
    num_channels=60,
    num_classes=3,
    dropout_rate=0.5,
    use_batch_norm=True
)

# Create a custom ResNet model
model = ModelFactory.create_model(
    'resnet',
    num_channels=60,
    num_classes=2,
    num_blocks=6,
    base_channels=64,
    dropout_rate=0.5,
    use_batch_norm=True
)
```

### Direct Import

```python
from resnet.resnet_model import EEGResNet18, EEGGenderResNet

# Create model directly
model = EEGResNet18(
    num_channels=60,
    num_classes=2,
    dropout_rate=0.5,
    use_batch_norm=True
)

# Or use task-specific model
model = EEGGenderResNet(num_channels=60)
```

### Using in Experiments

ResNet models can be used in the experiment framework by specifying the model type:

```python
from config import ExperimentConfig, DataConfig, ModelConfig, TrainingConfig

experiment = ExperimentConfig(
    name="gender_resnet_1s",
    model_type="gender_resnet",  # Use ResNet instead of 'gender_cnn'
    target_type="gender",
    data_config=DataConfig(...),
    model_config=ModelConfig(num_channels=60),
    training_config=TrainingConfig(...)
)
```

## Model Parameters

### Common Parameters

- **`num_channels`**: Number of EEG channels (default: 60)
- **`num_classes`**: Number of output classes (default: 2)
- **`dropout_rate`**: Dropout rate for FC layers (default: 0.5)
- **`use_layer_norm`**: Whether to use layer normalization on input (default: False)
- **`use_batch_norm`**: Whether to use batch normalization in blocks (default: True)

### EEGResNet-Specific Parameters

- **`layers`**: List of blocks per stage (e.g., [2,2,2,2] for ResNet18, [3,4,6,3] for ResNet34/50)
- **`block_type`**: Block type (BasicBlock or BottleneckBlock)
- **`base_channels`**: Base number of channels (default: 64)

## Key Features

1. **Skip Connections**: Enable gradient flow through deep networks
2. **Batch Normalization**: Stabilizes training and allows higher learning rates
3. **Adaptive Architecture**: Automatically handles different input sizes (1s, 2s, 4s segments)
4. **Task-Specific Heads**: Pre-configured models for common tasks
5. **Multi-Output Support**: Separate heads for multi-task learning

## Comparison with Standard CNN

| Feature | Standard CNN | ResNet |
|---------|--------------|--------|
| Skip Connections | ❌ | ✅ |
| Batch Normalization | Optional | Built-in |
| Depth | Fixed (8 layers) | Configurable (8-16 blocks) |
| Gradient Flow | Can degrade | Improved via skip connections |
| Training Stability | Good | Excellent |
| Architecture Variants | Single | ResNet18, ResNet34, ResNet50 |

## Notes

- ResNet models use batch normalization by default, which typically works better than layer normalization for CNNs
- The initial convolution processes all 60 EEG channels simultaneously using a `(60, 7)` kernel
- Adaptive average pooling ensures the model works with different segment lengths (1s, 2s, 4s)
- ResNet blocks use stride=2 for downsampling every 2 blocks to reduce computational cost

## References

- Original ResNet paper: "Deep Residual Learning for Image Recognition" (He et al., 2016)
- Adapted for EEG time-series classification with 2D convolutions over (channels, time) dimensions

