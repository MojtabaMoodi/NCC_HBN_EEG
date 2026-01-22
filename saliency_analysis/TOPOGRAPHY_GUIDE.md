# How to Draw Topography Plots

This guide explains how to create topography plots (scalp maps) showing EEG channel importance.

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
    cmap='viridis',          # Optional: Colormap name
    vmin=None,               # Optional: Min value for colormap (None = auto)
    vmax=None,               # Optional: Max value for colormap (None = auto)
    show_colorbar=True,      # Optional: Show colorbar
    sensors=True,            # Optional: Show sensor locations (MNE only)
    contours=6               # Optional: Number of contour lines (MNE only)
)
```

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

### MNE (Recommended - High Quality)

For best quality topography plots, install MNE:

```bash
pip install mne
```

**With MNE:**
- High-quality scalp plots with accurate channel positions
- Smooth interpolation between channels
- Professional appearance

### Without MNE (Fallback)

If MNE is not installed, the code automatically uses a matplotlib-based fallback:
- Simplified circular scalp plot
- Approximate channel positions
- Still functional but less polished

## Troubleshooting

### Issue: "MNE not available"

**Solution:** Install MNE or use the matplotlib fallback (automatic):
```bash
pip install mne
```

### Issue: "Channel names don't match"

**Solution:** Make sure you're using the correct channel names. The function uses `STANDARD_CHANNEL_NAMES` by default (60 channels).

### Issue: "Wrong number of channels"

**Solution:** Ensure your `channel_importance` array has exactly 60 values (or matches your number of channels).

## See Also

- `example_topography.py` - Complete examples with different use cases
- `visualization.py` - Full function documentation
- `README.md` - General saliency analysis documentation
