#!/usr/bin/env python3
"""
Example script for creating topography plots using MNE with accurate 10-20 system electrode positions.

This script demonstrates how to create topography plots from saliency results
using the plot_topography function with MNE support. The plots include:
- Accurate electrode positions based on the 10-20 system
- Electrode labels on the scalp
- Full scalp coverage with continuous heatmaps
- Professional-quality visualization

Requirements:
- MNE must be installed
- Conda environment must be activated: conda activate eeg_env
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from saliency_analysis.visualization import plot_topography

# Import STANDARD_CHANNEL_NAMES with fallback handling
try:
    from utils.eeg_constants import STANDARD_CHANNEL_NAMES
except ImportError:
    try:
        # Try importing from visualization module which has it as fallback
        from saliency_analysis.visualization import STANDARD_CHANNEL_NAMES
    except ImportError:
        # Final fallback: define locally
        STANDARD_CHANNEL_NAMES = [
            'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'F1', 'F2', 'F5', 'F6', 'F9', 'F10', 
            'AF3', 'AF4', 'AF7', 'AF8', 'AFZ', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 
            'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'P7', 'P3', 'PZ', 'P4', 'P8', 'P1', 'P2', 
            'P5', 'P6', 'PO3', 'PO4', 'PO7', 'PO8', 'POZ', 'OZ', 'O1', 'O2', 'C3', 'C4', 
            'C1', 'C2', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPZ'
        ]


def example_1_from_saved_results():
    """
    Example 1: Create topography plot from saved saliency results.
    """
    print("=" * 80)
    print("Example 1: Topography Plot from Saved Results")
    print("=" * 80)

    # Get path from user input
    default_path = "saliency_results/gender_baseline_4s/saliency_results_vanilla_gradients.npz"
    print(f"\nDefault path: {default_path}")
    results_path = input("Enter path to saliency results file (or press Enter for default): ").strip()
    
    if not results_path:
        results_path = default_path
    
    # Validate path exists
    results_path_obj = Path(results_path)
    if not results_path_obj.exists():
        print(f"❌ Error: Results file not found: {results_path}")
        print("   Run saliency analysis first, or provide a valid path to your results file.")
        return
    
    try:
        data = np.load(results_path)
        channel_importance = data['channel_importance']
        
        print(f"\nLoaded results from: {results_path}")
        print(f"Channel importance shape: {channel_importance.shape}")
        
        # Get output filename from user
        default_output = "topography_from_results.png"
        output_path = input(f"\nEnter output filename (or press Enter for '{default_output}'): ").strip()
        if not output_path:
            output_path = default_output
        
        # Use the existing plot_topography function which handles MNE properly
        # The function now includes explicit head outline visualization and full scalp coverage
        plot_topography(
            channel_importance,
            channel_names=STANDARD_CHANNEL_NAMES,
            title="Channel Importance Topography - Gender Classification",
            save_path=output_path,
            cmap='Spectral_r',  # Use Spectral_r colormap like the original
            figsize=(10, 8),
            outlines='head',  # Show head outline clearly
            sensors=True,  # Show sensor locations
            extrapolate='auto',  # Auto extrapolation for full scalp coverage
            border='mean',  # Extend to head boundary
            res=128  # High resolution for smooth heatmap
        )
        print(f"✅ Topography plot saved successfully to: {output_path}")
        
    except KeyError as e:
        print(f"❌ Error: Required key not found in results file: {e}")
        print("   Expected key: 'channel_importance'")
        print("   Available keys:", list(data.keys()) if 'data' in locals() else "N/A")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def example_2_custom_data():
    """
    Example 2: Create topography plot from custom channel importance data.
    """
    print("\n" + "=" * 80)
    print("Example 2: Topography Plot from Custom Data")
    print("=" * 80)

    np.random.seed(42)
    num_channels = len(STANDARD_CHANNEL_NAMES)
    channel_importance = np.random.rand(num_channels) * 0.5

    # Boost frontal and central indices
    frontal_indices = [0, 1, 2, 3, 4]  # FP1, FP2, F7, F3, FZ
    channel_importance[frontal_indices] += 0.5
    
    print(f"\nCreated custom channel importance for {num_channels} channels")
    print(f"Top 5 most important channels:")
    top_5_indices = np.argsort(channel_importance)[::-1][:5]
    for rank, idx in enumerate(top_5_indices, 1):
        print(f"  {rank}. {STANDARD_CHANNEL_NAMES[idx]}: {channel_importance[idx]:.4f}")

    # Get output filename from user
    default_output = "topography_custom.png"
    output_path = input(f"\nEnter output filename (or press Enter for '{default_output}'): ").strip()
    if not output_path:
        output_path = default_output

    # Use the existing plot_topography function
    # The function now includes explicit head outline visualization and full scalp coverage
    plot_topography(
        channel_importance,
        channel_names=STANDARD_CHANNEL_NAMES,
        title="Custom Channel Importance Topography",
        save_path=output_path,
        cmap='Spectral_r',  # Use Spectral_r colormap like the original
        figsize=(10, 8),
        outlines='head',  # Show head outline clearly
        sensors=True,  # Show sensor locations
        extrapolate='auto',  # Auto extrapolation for full scalp coverage
        border='mean',  # Extend to head boundary
        res=128  # High resolution for smooth heatmap
    )
    print(f"✅ Topography plot saved successfully to: {output_path}")


def main():
    print("1. From saved saliency results")
    print("2. From custom channel importance data")
    choice = input("\nEnter choice (1-2): ").strip()

    if choice == '1':
        example_1_from_saved_results()
    else:
        example_2_custom_data()


if __name__ == '__main__':
    main()
