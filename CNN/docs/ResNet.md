# ResNet Models for EEG Classification

ResNet (Residual Network) models adapted for EEG time-series classification. They use skip connections to enable training of deeper networks.

**Note:** The ResNet module is required. If it is not available, ModelFactory raises `ImportError` when importing ResNet models.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Layer counts](#layer-counts)
4. [Implementation details](#implementation-details)
5. [Usage](#usage)
6. [File structure](#file-structure)

---

## Overview

- **Input:** EEG data shape `(batch, 60 channels, timepoints)` — 1s: 200, 2s: 400, 4s: 800.
- **Output:** Classification (gender 2 classes, age 3 classes) or regression.
- **Features:** Skip connections, batch normalization, adaptive pooling.

### Available models

| Model | Block type | Blocks | Final channels | Use case |
|-------|------------|--------|----------------|----------|
| EEGResNet18 | BasicBlock | 8 [2,2,2,2] | 512 | Gender/age (light) |
| EEGResNet34 | BasicBlock | 16 [3,4,6,3] | 512 | Gender/age (deeper) |
| EEGResNet50 | BottleneckBlock | 16 [3,4,6,3] | 2048 | Gender/age (largest) |

Task-specific: `EEGGenderResNet`, `EEGAgeResNet`, `EEGAgeRegressionResNet`, `CombinedResNet`, `MultiOutputResNet`.

---

## Architecture

### BasicBlock (ResNet18, ResNet34)

- 2 conv layers per block (3×3).
- Skip: identity or 1×1 conv + BN when stride ≠ 1 or channels change.

### BottleneckBlock (ResNet50)

- 3 conv layers per block: 1×1 (reduce), 3×3 (features), 1×1 (expand).
- Expansion: output channels = base × 4.

### Stage layout

| Architecture | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Final ch |
|-------------|---------|---------|---------|---------|----------|
| ResNet18 | 64, 2 blk | 128, 2 blk | 256, 2 blk | 512, 2 blk | 512 |
| ResNet34 | 64, 3 blk | 128, 4 blk | 256, 6 blk | 512, 3 blk | 512 |
| ResNet50 | 256, 3 blk | 512, 4 blk | 1024, 6 blk | 2048, 3 blk | 2048 |

All use: initial conv+BN → ResNet blocks → global average pool → FC1 (final_ch → 128) → dropout → FC2 (128 → num_classes).

### Differences summary

- **ResNet18 vs 34:** Same BasicBlock and 512 channels; 34 has 16 blocks vs 8.
- **ResNet34 vs 50:** Same block count (16); 50 uses BottleneckBlock and 2048 channels (4×).

---

## Layer counts

Trainable layers counted: conv, FC, and batch norm (ReLU, dropout, pooling excluded).

| Architecture | Initial | ResNet blocks | FC | **Total** |
|-------------|---------|---------------|-----|-----------|
| ResNet18 | 2 | 38 | 2 | **42** |
| ResNet34 | 2 | 70 | 2 | **74** |
| ResNet50 | 2 | 104 | 2 | **108** |

---

## Implementation details

### Residual connection

In `resnet/resnet_model.py`, the residual is applied in the block’s `forward`:

- Save identity before conv path.
- Optional 1×1 conv + BN on identity when stride ≠ 1 or `in_channels != out_channels`.
- Add: `out = out + identity`, then ReLU.

### ResNet type selection

Scripts accept `--resnet_type` (18, 34, or 50). ModelFactory mapping:

- `--resnet_type 18` → `age_resnet` / `gender_resnet`
- `--resnet_type 34` → `age_resnet34` / `gender_resnet34`
- `--resnet_type 50` → `age_resnet50` / `gender_resnet50`

### Shared utilities (DRY)

- `utils.create_data_config_for_segment_length()` — data config for segment length.
- `utils.get_resnet_model_type()` — map ResNet type to model type and config.

---

## Usage

### Command-line scripts

Run with the `-m` form so that `config`, `experiment`, and `utils` resolve. Either from **project root (EEG)** or from **CNN/**:

**From project root (EEG):**
```bash
python -m CNN.resnet.run_resnet_age [OPTIONS]
python -m CNN.resnet.run_resnet_gender [OPTIONS]
```

**From CNN/:**
```bash
cd CNN
python -m resnet.run_resnet_age [OPTIONS]
python -m resnet.run_resnet_gender [OPTIONS]
```

**Age** = 3 classes; **Gender** = 2 classes.

**Common options:**

- `--mode`: Segment length (`1s`, `2s`, `4s`), default `1s`
- `--resnet_type`: `18`, `34`, or `50`, default `18`
- `--epochs`, `--learning_rate`, `--batch_size` (default when None: 512 for 1s, 192 for 2s, 128 for 4s), `--num_gpus`, `--random_seed`
- `--results_dir`, `--reports_dir`

**Examples:**

```bash
python -m resnet.run_resnet_age --mode 2s --resnet_type 34 --epochs 100
python -m resnet.run_resnet_gender --mode 4s --resnet_type 50
```

### Via ModelFactory

```python
from CNN.models.model_factory import ModelFactory

model = ModelFactory.create_model(
    'age_resnet34',
    num_channels=60,
    num_classes=3,
    dropout_rate=0.5,
)
```

### Direct import

```python
from resnet.resnet_model import EEGResNet18, EEGGenderResNet

model = EEGResNet18(num_channels=60, num_classes=2, dropout_rate=0.5)
# or
model = EEGGenderResNet(num_channels=60)
```

---

## File structure

```
CNN/
├── docs/
│   ├── ResNet.md           # This file
│   ├── Troubleshooting.md  # Class collapse, balance_method, class weights
│   └── GPU_SETUP_CHANGES.md # GPU detection, multi-GPU, gpu_utils
├── resnet/
│   ├── README.md           # Short pointer to docs/ResNet.md
│   ├── resnet_model.py     # Blocks, EEGResNet, EEGResNet18/34/50, task variants
│   ├── run_resnet_age.py
│   └── run_resnet_gender.py
├── utils.py                # create_data_config_for_segment_length, get_resnet_model_type
└── models/
    └── model_factory.py    # ResNet registration
```

---

## Verification

```python
from resnet.resnet_model import EEGResNet18, EEGResNet34, EEGResNet50

for name, model_cls in [("18", EEGResNet18), ("34", EEGResNet34), ("50", EEGResNet50)]:
    m = model_cls()
    print(f"ResNet{name}: layers={m.layers}, blocks={sum(m.layers)}, "
          f"block={m.block_type.__name__}, final_channels={m.final_channels}")
```
