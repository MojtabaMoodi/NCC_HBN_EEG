#!/usr/bin/env python3
"""
Example script showing how to draw topography plots for EEG channel importance.

This script demonstrates:
1. How topography plots are automatically generated during saliency analysis
2. How to create topography plots from saved saliency results
3. How to create custom topography plots from channel importance data
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import visualization functions
try:
    from saliency_analysis.visualization import plot_topography
except ImportError:
    # If importing as a module fails, try direct import
    from visualization import plot_topography

# Import channel names
try:
    from utils.eeg_constants import STANDARD_CHANNEL_NAMES
except ImportError:
    # Fallback: define locally if import fails
    STANDARD_CHANNEL_NAMES = [
        'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'F1', 'F2', 'F5', 'F6', 'F9', 'F10', 
        'AF3', 'AF4', 'AF7', 'AF8', 'AFZ', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 
        'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'P7', 'P3', 'PZ', 'P4', 'P8', 'P1', 'P2', 
        'P5', 'P6', 'PO3', 'PO4', 'PO7', 'PO8', 'POZ', 'OZ', 'O1', 'O2', 'C3', 'C4', 
        'C1', 'C2', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPZ'
    ]


def example_1_from_saved_results():
    """
    Example 1: Load saved saliency results and create topography plot.
    
    This is useful when you've already run saliency analysis and want to
    create additional topography plots from the saved results.
    """
    print("=" * 80)
    print("Example 1: Creating topography from saved saliency results")
    print("=" * 80)
    
    # Path to saved saliency results (adjust to your actual path)
    results_path = "saliency_results/gender_baseline_4s/saliency_results_vanilla_gradients.npz"
    
    try:
        # Load saved results
        data = np.load(results_path)
        channel_importance = data['channel_importance']  # Shape: (num_samples, num_channels)
        
        print(f"Loaded results from: {results_path}")
        print(f"Channel importance shape: {channel_importance.shape}")
        print(f"Number of samples: {channel_importance.shape[0]}")
        print(f"Number of channels: {channel_importance.shape[1]}")
        
        # Create topography plot (averages across all samples automatically)
        plot_topography(
            channel_importance,
            channel_names=STANDARD_CHANNEL_NAMES,
            title="Channel Importance Topography - Gender Classification (4s)",
            save_path="example_topography_from_saved.png",
            cmap='viridis'
        )
        print("✅ Topography plot saved to: example_topography_from_saved.png")
        
    except FileNotFoundError:
        print(f"⚠️  Results file not found: {results_path}")
        print("   Run saliency analysis first, or adjust the path to your results file.")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_2_custom_data():
    """
    Example 2: Create topography plot from custom channel importance data.
    
    This is useful for creating topography plots from any channel importance values,
    not just from saliency analysis.
    """
    print("\n" + "=" * 80)
    print("Example 2: Creating topography from custom channel importance data")
    print("=" * 80)
    
    # Create example channel importance data (60 channels)
    # In practice, this would come from your analysis
    np.random.seed(42)
    num_channels = len(STANDARD_CHANNEL_NAMES)
    
    # Simulate channel importance (higher values = more important)
    # You can replace this with your actual channel importance values
    channel_importance = np.random.rand(num_channels) * 0.5
    
    # Make some channels more important (e.g., frontal and central regions)
    frontal_indices = [0, 1, 2, 3, 4, 5, 6]  # FP1, FP2, F7, F3, FZ, F4, F8
    central_indices = [46, 47, 48, 49]  # C3, C4, C1, C2
    
    for idx in frontal_indices + central_indices:
        channel_importance[idx] += 0.3
    
    print(f"Created example channel importance for {num_channels} channels")
    print(f"Top 5 most important channels:")
    top_5_indices = np.argsort(channel_importance)[::-1][:5]
    for rank, idx in enumerate(top_5_indices, 1):
        print(f"  {rank}. {STANDARD_CHANNEL_NAMES[idx]}: {channel_importance[idx]:.4f}")
    
    # Create topography plot
    plot_topography(
        channel_importance,
        channel_names=STANDARD_CHANNEL_NAMES,
        title="Example Channel Importance Topography",
        save_path="example_topography_custom.png",
        cmap='viridis'
    )
    print("✅ Topography plot saved to: example_topography_custom.png")


def example_3_multiple_samples():
    """
    Example 3: Create topography plot from multiple samples (averages automatically).
    
    The plot_topography function automatically averages across samples if you
    provide a 2D array (num_samples, num_channels).
    """
    print("\n" + "=" * 80)
    print("Example 3: Creating topography from multiple samples (auto-averaging)")
    print("=" * 80)
    
    num_channels = len(STANDARD_CHANNEL_NAMES)
    num_samples = 100
    
    # Create example data for multiple samples
    np.random.seed(42)
    channel_importance = np.random.rand(num_samples, num_channels) * 0.5
    
    # Add some consistent patterns across samples
    frontal_indices = [0, 1, 2, 3, 4, 5, 6]
    for idx in frontal_indices:
        channel_importance[:, idx] += 0.3
    
    print(f"Created channel importance for {num_samples} samples, {num_channels} channels")
    print(f"Shape: {channel_importance.shape}")
    print("Note: plot_topography will automatically average across samples")
    
    # Create topography plot (will average across samples)
    plot_topography(
        channel_importance,  # 2D array - will be averaged automatically
        channel_names=STANDARD_CHANNEL_NAMES,
        title=f"Channel Importance Topography (averaged across {num_samples} samples)",
        save_path="example_topography_multiple_samples.png",
        cmap='viridis'
    )
    print("✅ Topography plot saved to: example_topography_multiple_samples.png")


def example_4_during_saliency_analysis():
    """
    Example 4: Topography plots are automatically generated during saliency analysis.
    
    When you run saliency analysis, topography plots are automatically created.
    This example shows what happens behind the scenes.
    """
    print("\n" + "=" * 80)
    print("Example 4: Topography plots during saliency analysis")
    print("=" * 80)
    
    print("""
    When you run saliency analysis using the main script:
    
    python saliency_analysis/main.py \\
        --checkpoint <checkpoint_path> \\
        --target_type gender \\
        --segment_length 4s \\
        --method vanilla_gradients \\
        --output_dir saliency_results/gender_analysis
    
    The topography plots are automatically generated in the output directory:
    
    Output files:
    - topography_vanilla_gradients.png          (overall topography)
    - active/topography_vanilla_gradients.png   (active tasks only)
    - passive/topography_vanilla_gradients.png (passive tasks only)
    
    You don't need to do anything extra - it's all automatic!
    """)


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("EEG Topography Plotting Examples")
    print("=" * 80)
    print("\nThis script demonstrates different ways to create topography plots.")
    print("Choose an example to run:\n")
    print("1. From saved saliency results")
    print("2. From custom channel importance data")
    print("3. From multiple samples (auto-averaging)")
    print("4. Show how it works during saliency analysis")
    print("5. Run all examples")
    print("\n" + "=" * 80)
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        example_1_from_saved_results()
    elif choice == '2':
        example_2_custom_data()
    elif choice == '3':
        example_3_multiple_samples()
    elif choice == '4':
        example_4_during_saliency_analysis()
    elif choice == '5':
        example_2_custom_data()
        example_3_multiple_samples()
        example_4_during_saliency_analysis()
        print("\n" + "=" * 80)
        print("Note: Example 1 requires existing saliency results.")
        print("      Run it separately if you have saved results.")
        print("=" * 80)
    else:
        print("Invalid choice. Please run the script again and choose 1-5.")


if __name__ == '__main__':
    main()
