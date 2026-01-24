# How to Draw Topography Plots

This guide explains how to create topography plots (scalp maps) showing EEG channel importance.

**Important:** MNE is now **required** for topography plotting. Make sure to activate the conda environment:
```bash
conda activate eeg_env
```

The topography plots use MNE's `standard_1020` montage for accurate 10-20 system electrode positions, ensuring all electrodes are placed correctly on the scalp with proper labels.

## Quick Start

### Method 1: Automatic (Recommended)

Topography plots are **automatically generated** when you run saliency analysis:

```bash
python saliency_analysis/main.py \
    --checkpoint <checkpoint_path> \
    --target_type gender \
    --segment_length 4s \
    --method integrated_gradients \
    --output_dir saliency_results/gender_analysis
```

**Output files:**
- `saliency_results/gender_analysis/topography_integrated_gradients.png` - Overall topography
- `saliency_results/gender_analysis/active/topography_integrated_gradients.png` - Active tasks
- `saliency_results/gender_analysis/passive/topography_integrated_gradients.png` - Passive tasks

### Method 2: From Saved Results

If you've already run saliency analysis, you can create topography plots from saved results:

```python
import numpy as np
from saliency_analysis.visualization import plot_topography
from utils.eeg_constants import STANDARD_CHANNEL_NAMES

# Load saved saliency results
data = np.load("saliency_results/gender_baseline_4s/saliency_results_integrated_gradients.npz")
channel_importance = data['channel_importance']  # Shape: (num_samples, num_channels)

# Create topography plot (automatically averages across samples)
plot_topography(
    channel_importance,
    channel_names=STANDARD_CHANNEL_NAMES,
    title="Channel Importance Topography - Gender Classification",
    save_path="my_topography.png",
    cmap='viridis'
)
```

### Method 3: From Custom Data

Create topography plots from any channel importance values:

```python
import numpy as np
from saliency_analysis.visualization import plot_topography
from utils.eeg_constants import STANDARD_CHANNEL_NAMES

# Your channel importance values (60 channels)
channel_importance = np.array([...])  # Shape: (60,)

# Create topography plot
plot_topography(
    channel_importance,
    channel_names=STANDARD_CHANNEL_NAMES,
    title="My Custom Topography",
    save_path="custom_topography.png"
)
```

## Function Parameters

```python
plot_topography(
    channel_importance,      # Required: 1D array (num_channels,) or 2D (num_samples, num_channels)
    channel_names=None,      # Optional: List of channel names (uses STANDARD_CHANNEL_NAMES if None)
    title="...",             # Optional: Plot title
    save_path=None,          # Optional: Path to save figure (None = don't save)
    figsize=(10, 8),         # Optional: Figure size
    cmap='viridis',          # Optional: Colormap name ('viridis', 'Spectral_r', 'hot', etc.)
    vmin=None,               # Optional: Min value for colormap (None = auto)
    vmax=None,               # Optional: Max value for colormap (None = auto)
    show_colorbar=True,      # Optional: Show colorbar
    sensors=True,            # Optional: Show sensor locations
    contours=6,              # Optional: Number of contour lines
    outlines='head',         # Optional: Head outline style ('head', 'skirt', 'head+skirt')
    extrapolate='auto',      # Optional: Extrapolation method for full scalp coverage
    border='mean',           # Optional: Border style for full coverage
    res=128                  # Optional: Resolution of interpolation grid (higher = smoother)
)
```

**Note:** All parameters use MNE's `standard_1020` montage for accurate 10-20 system electrode positions. Electrode labels are automatically added at their correct positions on the scalp.

## Examples

### Example 1: Basic Usage

```python
from saliency_analysis.visualization import plot_topography
from utils.eeg_constants import STANDARD_CHANNEL_NAMES
import numpy as np

# Create example data
channel_importance = np.random.rand(60) * 0.5

# Plot
plot_topography(channel_importance, save_path="topography.png")
```

### Example 2: Custom Colormap and Title

```python
plot_topography(
    channel_importance,
    channel_names=STANDARD_CHANNEL_NAMES,
    title="Gender Classification - Channel Importance",
    save_path="gender_topography.png",
    cmap='hot',  # Try: 'viridis', 'plasma', 'inferno', 'hot', 'coolwarm'
    figsize=(12, 10)
)
```

### Example 3: Multiple Samples (Auto-Averaging)

```python
# If you have multiple samples, the function automatically averages them
channel_importance = np.random.rand(100, 60)  # 100 samples, 60 channels

plot_topography(
    channel_importance,  # 2D array - will be averaged automatically
    title="Averaged across 100 samples",
    save_path="averaged_topography.png"
)
```

### Example 4: From Saliency Analysis Results

```python
import numpy as np
from saliency_analysis.visualization import plot_topography
from utils.eeg_constants import STANDARD_CHANNEL_NAMES

# Load results from saliency analysis
results = np.load("saliency_results/gender_baseline_4s/saliency_results_integrated_gradients.npz")
channel_importance = results['channel_importance']

# Create topography
plot_topography(
    channel_importance,
    channel_names=STANDARD_CHANNEL_NAMES,
    title="Gender Classification - Integrated Gradients",
    save_path="gender_topography.png"
)
```

## Requirements

### MNE (Required)

**MNE is now REQUIRED** for topography plotting. The code will raise an error if MNE is not available.

**Installation:**
```bash
conda activate eeg_env
# MNE should already be installed in eeg_env
# If not: pip install mne
```

**With MNE:**
- High-quality scalp plots with accurate 10-20 system electrode positions
- Smooth interpolation between channels
- Full scalp coverage (front to back, left to right)
- Electrode labels positioned correctly on the scalp
- Professional appearance matching standard EEG visualization practices

## Troubleshooting

### Issue: "MNE is required for topography plotting"

**Solution:** MNE is now required. Activate the conda environment:
```bash
conda activate eeg_env
```

If MNE is not installed in the environment:
```bash
conda activate eeg_env
pip install mne
```

### Issue: "Channel names don't match"

**Solution:** Make sure you're using the correct channel names. The function uses `STANDARD_CHANNEL_NAMES` by default (60 channels).

### Issue: "Wrong number of channels"

**Solution:** Ensure your `channel_importance` array has exactly 60 values (or matches your number of channels).

## Example Script

Run the script to create topography plots:

```bash
conda activate eeg_env
python saliency_analysis/create_topography.py
```

This script provides an interactive interface to:
1. Create topography plots from saved saliency results (prompts for file path)
2. Create topography plots from custom channel importance data

The script uses MNE's `standard_1020` montage for accurate 10-20 system electrode positions and includes electrode labels on the scalp.

## See Also

- `create_topography.py` - Interactive script for creating topography plots with MNE support
- `visualization.py` - Full function documentation
- `README.md` - General saliency analysis documentation
