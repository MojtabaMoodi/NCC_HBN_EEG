"""
Visualization utilities for saliency map analysis.

Provides functions to visualize channel importance, create heatmaps,
and generate summary statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import sys
from pathlib import Path

# Try to import STANDARD_CHANNEL_NAMES from utils
# Handle different import contexts (direct import, module import, etc.)
try:
    # First try: direct import (when run as script or when utils is in path)
    from utils.eeg_constants import STANDARD_CHANNEL_NAMES
except ImportError:
    try:
        # Second try: add parent directory and import
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        from utils.eeg_constants import STANDARD_CHANNEL_NAMES
    except ImportError:
        # Fallback: define locally if import fails
        # Standard 10-20 channel names (60 channels from EEG data)
        # These should match the channel order in HDF5 files
        STANDARD_CHANNEL_NAMES = [
            'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'F1', 'F2', 'F5', 'F6', 'F9', 'F10', 
            'AF3', 'AF4', 'AF7', 'AF8', 'AFZ', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 
            'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'P7', 'P3', 'PZ', 'P4', 'P8', 'P1', 'P2', 
            'P5', 'P6', 'PO3', 'PO4', 'PO7', 'PO8', 'POZ', 'OZ', 'O1', 'O2', 'C3', 'C4', 
            'C1', 'C2', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPZ'
        ]

# MNE is REQUIRED for topography plotting
# Make sure to activate conda environment: conda activate eeg_env
try:
    import mne
    from mne.channels import make_standard_montage
    from mne.channels.layout import _find_topomap_coords
    try:
        from mne.viz import add_colorbar as mne_add_colorbar
        MNE_COLORBAR_AVAILABLE = True
    except ImportError:
        MNE_COLORBAR_AVAILABLE = False
    MNE_AVAILABLE = True
except ImportError as e:
    MNE_AVAILABLE = False
    MNE_COLORBAR_AVAILABLE = False
    raise ImportError(
        "MNE is required for topography plotting. "
        "Please install MNE and activate the conda environment: "
        "conda activate eeg_env\n"
        f"Original error: {e}"
    )

# Constants for plotting
# Font sizes
FONT_SIZE_TITLE = 14
FONT_SIZE_AXIS_LABEL = 12
FONT_SIZE_SUBTITLE = 12
FONT_SIZE_SMALL = 10
FONT_SIZE_TINY = 8
FONT_SIZE_ELECTRODE_LABEL = 7

# Figure settings
FIGURE_DPI = 300
FIGURE_BBOX = 'tight'

# Colorbar settings
COLORBAR_PAD = 0.1
COLORBAR_SHRINK = 0.8
COLORBAR_LABEL = 'Channel Importance'

# Topography-specific constants
TOPOGRAPHY_COLORBAR_PAD = COLORBAR_PAD
TOPOGRAPHY_COLORBAR_SHRINK = COLORBAR_SHRINK
TOPOGRAPHY_COLORBAR_LABEL = COLORBAR_LABEL
TOPOGRAPHY_LABEL_FONTSIZE = FONT_SIZE_ELECTRODE_LABEL
TOPOGRAPHY_TITLE_FONTSIZE = FONT_SIZE_TITLE
TOPOGRAPHY_TITLE_PAD = 20
TOPOGRAPHY_DPI = FIGURE_DPI

# Interpolation settings for matplotlib fallback
INTERPOLATION_GRID_RESOLUTION = 300
INTERPOLATION_GRID_RANGE = 1.1
INTERPOLATION_NORMALIZATION_FACTOR = 0.95


def _save_figure(save_path: Optional[str], figure_name: str = "plot") -> None:
    """
    Save figure with consistent settings.
    
    Args:
        save_path: Path to save figure (None = don't save)
        figure_name: Name of the figure type for logging
    """
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches=FIGURE_BBOX)
        print(f"Saved {figure_name} to {save_path}")


def plot_channel_importance_heatmap(channel_importance: np.ndarray,
                                   channel_names: Optional[List[str]] = None,
                                   title: str = "Channel Importance Heatmap",
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (12, 8),
                                   cmap: str = 'viridis'):
    """
    Plot heatmap of channel importance across samples.
    
    Args:
        channel_importance: Array of shape (num_samples, num_channels)
        channel_names: List of channel names (if None, uses indices)
        title: Plot title
        save_path: Path to save figure (None = don't save)
        figsize: Figure size
        cmap: Colormap name
    """
    num_samples, num_channels = channel_importance.shape
    
    # Create channel names if not provided
    if channel_names is None:
        channel_names = [f'Channel {i+1}' for i in range(num_channels)]
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame(channel_importance.T, columns=[f'Sample {i+1}' for i in range(num_samples)],
                     index=channel_names)
    
    # Create figure
    plt.figure(figsize=figsize)
    # Limit x-axis labels for readability (show labels only if <= 50 samples)
    show_x_labels = num_samples <= 50
    sns.heatmap(df, cmap=cmap, cbar_kws={'label': 'Importance'}, 
                xticklabels=show_x_labels,
                yticklabels=True)
    plt.title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold')
    plt.xlabel('Sample Index', fontsize=FONT_SIZE_AXIS_LABEL)
    plt.ylabel('EEG Channel', fontsize=FONT_SIZE_AXIS_LABEL)
    plt.tight_layout()
    _save_figure(save_path, "heatmap")
    
    return plt.gcf()


def plot_average_channel_importance(channel_importance: np.ndarray,
                                   channel_names: Optional[List[str]] = None,
                                   title: str = "Average Channel Importance",
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (14, 6),
                                   top_k: Optional[int] = None,
                                   color: str = 'steelblue'):
    """
    Plot bar chart of average channel importance.
    
    Args:
        channel_importance: Array of shape (num_samples, num_channels)
        channel_names: List of channel names
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        top_k: Show only top K channels (None = show all)
        color: Bar color
    """
    # Compute average importance per channel
    avg_importance = channel_importance.mean(axis=0)
    
    num_channels = len(avg_importance)
    if channel_names is None:
        channel_names = [f'Channel {i+1}' for i in range(num_channels)]
    
    # Sort by importance
    sorted_indices = np.argsort(avg_importance)[::-1]
    sorted_importance = avg_importance[sorted_indices]
    sorted_names = [channel_names[i] for i in sorted_indices]
    
    # Select top K if specified
    if top_k is not None:
        sorted_importance = sorted_importance[:top_k]
        sorted_names = sorted_names[:top_k]
    
    # Create figure
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(sorted_names)), sorted_importance, color=color, alpha=0.7)
    plt.xlabel('EEG Channel', fontsize=FONT_SIZE_AXIS_LABEL)
    plt.ylabel('Average Importance', fontsize=FONT_SIZE_AXIS_LABEL)
    plt.title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold')
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, sorted_importance)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=FONT_SIZE_TINY)
    
    _save_figure(save_path, "bar chart")
    
    return plt.gcf()


def plot_channel_importance_distribution(channel_importance: np.ndarray,
                                        channel_names: Optional[List[str]] = None,
                                        title: str = "Channel Importance Distribution",
                                        save_path: Optional[str] = None,
                                        figsize: Tuple[int, int] = (14, 8),
                                        top_k: int = 10):
    """
    Plot distribution of importance values for top K channels.
    
    Args:
        channel_importance: Array of shape (num_samples, num_channels)
        channel_names: List of channel names
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        top_k: Number of top channels to show
    """
    # Compute average importance
    avg_importance = channel_importance.mean(axis=0)
    
    num_channels = len(avg_importance)
    if channel_names is None:
        channel_names = [f'Channel {i+1}' for i in range(num_channels)]
    
    # Get top K channels
    top_k_indices = np.argsort(avg_importance)[::-1][:top_k]
    
    # Prepare data for plotting
    plot_data = []
    for idx in top_k_indices:
        channel_name = channel_names[idx]
        importance_values = channel_importance[:, idx]
        plot_data.extend([(channel_name, val) for val in importance_values])
    
    df = pd.DataFrame(plot_data, columns=['Channel', 'Importance'])
    
    # Create figure
    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x='Channel', y='Importance', palette='Set2')
    plt.title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold')
    plt.xlabel('EEG Channel', fontsize=FONT_SIZE_AXIS_LABEL)
    plt.ylabel('Importance Value', fontsize=FONT_SIZE_AXIS_LABEL)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save_figure(save_path, "distribution plot")
    
    return plt.gcf()


def plot_saliency_map_sample(saliency_map: np.ndarray,
                            channel_names: Optional[List[str]] = None,
                            title: str = "Saliency Map",
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (16, 8),
                            cmap: Optional[str] = None,
                            use_absolute: bool = True):
    """
    Plot saliency map for a single sample (channels x timepoints).
    
    Args:
        saliency_map: Saliency map array of shape (num_channels, timepoints)
        channel_names: List of channel names
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        cmap: Colormap name. If None, automatically chosen based on use_absolute
        use_absolute: If True, show absolute values (standard for aggregated maps).
                     If False, show signed values (useful for individual samples).
                     Default: True (following best practices for aggregated maps)
    """
    num_channels, timepoints = saliency_map.shape
    
    if channel_names is None:
        channel_names = [f'Channel {i+1}' for i in range(num_channels)]
    
    # Determine if we have negative values
    has_negative = saliency_map.min() < 0
    
    # Auto-select colormap if not specified
    if cmap is None:
        if use_absolute or not has_negative:
            # Absolute values or positive-only: use sequential colormap
            # This is standard for aggregated maps showing importance magnitude
            cmap = 'viridis'  # Sequential, perceptually uniform
        else:
            # Signed values: use diverging colormap
            cmap = 'RdBu_r'  # Diverging: red=positive, blue=negative
    
    # Apply absolute if requested (standard for aggregated maps)
    if use_absolute and has_negative:
        saliency_map = np.abs(saliency_map)
        title_suffix = " (absolute values)"
        if title_suffix not in title:
            title += title_suffix
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Normalize for better visualization
    if cmap == 'viridis' or saliency_map.min() >= 0:
        # Sequential colormap or positive-only values (absolute values)
        vmax = saliency_map.max()
        vmin = 0
        colorbar_label = 'Saliency Magnitude' if use_absolute else 'Saliency Value'
    else:
        # Diverging colormap (signed values)
        vmax = np.abs(saliency_map).max()
        vmin = -vmax
        colorbar_label = 'Saliency Value (red=positive, blue=negative)'
    
    im = plt.imshow(saliency_map, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest')
    plt.colorbar(im, label=colorbar_label)
    plt.xlabel('Time Point', fontsize=FONT_SIZE_AXIS_LABEL)
    plt.ylabel('EEG Channel', fontsize=FONT_SIZE_AXIS_LABEL)
    plt.title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold')
    plt.yticks(range(num_channels), channel_names)
    plt.tight_layout()
    _save_figure(save_path, "saliency map")
    
    return plt.gcf()


def plot_comparison_by_class(channel_importance: np.ndarray,
                            labels: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            title: str = "Channel Importance by Class",
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (16, 8),
                            top_k: int = 10):
    """
    Compare channel importance across different classes.
    
    Args:
        channel_importance: Array of shape (num_samples, num_channels)
        labels: Array of shape (num_samples,) with class labels
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        top_k: Number of top channels to show
    """
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    
    if class_names is None:
        class_names_full = [f'Class {int(cls)}' for cls in unique_classes]
    else:
        # Map class values to their names from the provided class_names list
        # This handles cases where filtered data has fewer classes than class_names
        class_names_full = class_names
    
    # Map unique classes in this data subset to their names
    # If class_names was provided, use it; otherwise use generated names
    mapped_class_names = []
    for cls in unique_classes:
        cls_int = int(cls)
        if class_names is None:
            mapped_class_names.append(f'Class {cls_int}')
        else:
            # Use the class value as index into class_names
            # Handle cases where class value might be out of range
            if 0 <= cls_int < len(class_names_full):
                mapped_class_names.append(class_names_full[cls_int])
            else:
                mapped_class_names.append(f'Class {cls_int}')
    
    # Compute average importance per class
    # Use class value as key to avoid index mismatches
    class_importance = {}
    for cls in unique_classes:
        mask = labels == cls
        class_importance[cls] = channel_importance[mask].mean(axis=0)
    
    # Get top K channels based on overall average
    overall_avg = channel_importance.mean(axis=0)
    top_k_indices = np.argsort(overall_avg)[::-1][:top_k]
    
    # Prepare data for plotting
    plot_data = []
    for cls_idx, cls in enumerate(unique_classes):
        cls_name = mapped_class_names[cls_idx]
        for ch_idx in top_k_indices:
            plot_data.append({
                'Class': cls_name,
                'Channel': f'Channel {ch_idx+1}',
                'Importance': class_importance[cls][ch_idx]
            })
    
    df = pd.DataFrame(plot_data)
    
    # Create figure
    fig, axes = plt.subplots(1, num_classes, figsize=figsize, sharey=True)
    if num_classes == 1:
        axes = [axes]
    
    for ax, cls_idx, cls_name in zip(axes, range(num_classes), mapped_class_names):
        cls_data = df[df['Class'] == cls_name]
        channels = cls_data['Channel'].values
        importance = cls_data['Importance'].values
        
        ax.barh(range(len(channels)), importance, color=f'C{cls_idx}', alpha=0.7)
        ax.set_yticks(range(len(channels)))
        ax.set_yticklabels(channels)
        ax.set_xlabel('Average Importance', fontsize=FONT_SIZE_SMALL)
        ax.set_title(f'{cls_name}', fontsize=FONT_SIZE_SUBTITLE, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()  # Top channel at top
    
    plt.suptitle(title, fontsize=FONT_SIZE_TITLE, fontweight='bold')
    plt.tight_layout()
    _save_figure(save_path, "comparison plot")
    
    return fig


def save_channel_ranking(channel_importance: np.ndarray,
                        channel_names: Optional[List[str]] = None,
                        save_path: str = "channel_ranking.csv",
                        top_k: Optional[int] = None):
    """
    Save channel ranking to CSV file.
    
    Args:
        channel_importance: Array of shape (num_samples, num_channels)
        channel_names: List of channel names
        save_path: Path to save CSV
        top_k: Save only top K channels (None = all)
    """
    avg_importance = channel_importance.mean(axis=0)
    std_importance = channel_importance.std(axis=0)
    
    num_channels = len(avg_importance)
    if channel_names is None:
        channel_names = [f'Channel {i+1}' for i in range(num_channels)]
    
    # Sort by importance
    sorted_indices = np.argsort(avg_importance)[::-1]
    
    # Create DataFrame
    data = {
        'Rank': range(1, num_channels + 1),
        'Channel': [channel_names[i] for i in sorted_indices],
        'Mean_Importance': avg_importance[sorted_indices],
        'Std_Importance': std_importance[sorted_indices],
        'Min_Importance': channel_importance[:, sorted_indices].min(axis=0),
        'Max_Importance': channel_importance[:, sorted_indices].max(axis=0)
    }
    
    df = pd.DataFrame(data)
    
    if top_k is not None:
        df = df.head(top_k)
    
    df.to_csv(save_path, index=False)
    print(f"Saved channel ranking to {save_path}")
    
    return df


def plot_method_comparison(channel_importance_vg: np.ndarray,
                          channel_importance_ig: np.ndarray,
                          channel_names: Optional[List[str]] = None,
                          title: str = "Saliency Method Comparison",
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (16, 10),
                          top_k: int = 20):
    """
    Compare channel importance between Vanilla Gradients and Integrated Gradients.
    
    Args:
        channel_importance_vg: Array of shape (num_samples, num_channels) for Vanilla Gradients
        channel_importance_ig: Array of shape (num_samples, num_channels) for Integrated Gradients
        channel_names: List of channel names
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        top_k: Number of top channels to show
    """
    num_channels = channel_importance_vg.shape[1]
    if channel_names is None:
        channel_names = [f'Channel {i+1}' for i in range(num_channels)]
    
    # Compute average importance for each method
    avg_vg = channel_importance_vg.mean(axis=0)
    avg_ig = channel_importance_ig.mean(axis=0)
    
    # Get top K channels based on average of both methods
    combined_avg = (avg_vg + avg_ig) / 2
    top_k_indices = np.argsort(combined_avg)[::-1][:top_k]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Side-by-side bar comparison (top K channels)
    ax1 = axes[0, 0]
    x = np.arange(len(top_k_indices))
    width = 0.35
    ax1.bar(x - width/2, avg_vg[top_k_indices], width, label='Vanilla Gradients', alpha=0.8, color='steelblue')
    ax1.bar(x + width/2, avg_ig[top_k_indices], width, label='Integrated Gradients', alpha=0.8, color='coral')
    ax1.set_xlabel('Channel', fontsize=FONT_SIZE_SMALL)
    ax1.set_ylabel('Average Importance', fontsize=FONT_SIZE_SMALL)
    ax1.set_title(f'Top {top_k} Channels Comparison', fontsize=FONT_SIZE_SUBTITLE, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([channel_names[i] for i in top_k_indices], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Scatter plot: VG vs IG (all channels)
    ax2 = axes[0, 1]
    ax2.scatter(avg_vg, avg_ig, alpha=0.6, s=50)
    # Add diagonal line
    max_val = max(avg_vg.max(), avg_ig.max())
    min_val = min(avg_vg.min(), avg_ig.min())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
    ax2.set_xlabel('Vanilla Gradients Importance', fontsize=FONT_SIZE_SMALL)
    ax2.set_ylabel('Integrated Gradients Importance', fontsize=FONT_SIZE_SMALL)
    ax2.set_title('Method Correlation (All Channels)', fontsize=FONT_SIZE_SUBTITLE, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Compute correlation
    correlation = np.corrcoef(avg_vg, avg_ig)[0, 1]
    ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=ax2.transAxes, fontsize=FONT_SIZE_SMALL,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Difference plot (IG - VG)
    ax3 = axes[1, 0]
    diff = avg_ig - avg_vg
    sorted_indices = np.argsort(np.abs(diff))[::-1][:top_k]
    colors = ['green' if diff[i] > 0 else 'red' for i in sorted_indices]
    ax3.barh(range(len(sorted_indices)), diff[sorted_indices], color=colors, alpha=0.7)
    ax3.set_yticks(range(len(sorted_indices)))
    ax3.set_yticklabels([channel_names[i] for i in sorted_indices])
    ax3.set_xlabel('Difference (IG - VG)', fontsize=FONT_SIZE_SMALL)
    ax3.set_title(f'Top {top_k} Channels by Absolute Difference', fontsize=FONT_SIZE_SUBTITLE, fontweight='bold')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Ranking comparison
    ax4 = axes[1, 1]
    ranking_vg = np.argsort(avg_vg)[::-1]
    ranking_ig = np.argsort(avg_ig)[::-1]
    # Create ranking arrays
    rank_vg = np.zeros(num_channels, dtype=int)
    rank_ig = np.zeros(num_channels, dtype=int)
    for i, idx in enumerate(ranking_vg):
        rank_vg[idx] = i + 1
    for i, idx in enumerate(ranking_ig):
        rank_ig[idx] = i + 1
    
    # Show top K channels with their rankings
    top_k_for_ranking = min(top_k, num_channels)
    top_channels_combined = np.argsort(combined_avg)[::-1][:top_k_for_ranking]
    
    x_pos = np.arange(len(top_channels_combined))
    ax4.scatter(rank_vg[top_channels_combined], rank_ig[top_channels_combined], 
               s=100, alpha=0.6, c=range(len(top_channels_combined)), cmap='viridis')
    for i, ch_idx in enumerate(top_channels_combined):
        ax4.annotate(channel_names[ch_idx], 
                    (rank_vg[ch_idx], rank_ig[ch_idx]),
                    fontsize=FONT_SIZE_TINY, alpha=0.7)
    ax4.plot([1, top_k_for_ranking], [1, top_k_for_ranking], 'r--', alpha=0.5, label='Same ranking')
    ax4.set_xlabel('Vanilla Gradients Rank', fontsize=FONT_SIZE_SMALL)
    ax4.set_ylabel('Integrated Gradients Rank', fontsize=FONT_SIZE_SMALL)
    ax4.set_title('Ranking Comparison (Lower = Better)', fontsize=FONT_SIZE_SUBTITLE, fontweight='bold')
    ax4.invert_xaxis()
    ax4.invert_yaxis()
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=FONT_SIZE_TITLE, fontweight='bold')
    plt.tight_layout()
    _save_figure(save_path, "method comparison plot")
    
    return fig


def save_method_comparison_report(channel_importance_vg: np.ndarray,
                                 channel_importance_ig: np.ndarray,
                                 channel_names: Optional[List[str]] = None,
                                 save_path: str = "method_comparison_report.csv"):
    """
    Save detailed comparison report between methods.
    
    Args:
        channel_importance_vg: Array of shape (num_samples, num_channels) for Vanilla Gradients
        channel_importance_ig: Array of shape (num_samples, num_channels) for Integrated Gradients
        channel_names: List of channel names
        save_path: Path to save CSV report
    """
    num_channels = channel_importance_vg.shape[1]
    if channel_names is None:
        channel_names = [f'Channel {i+1}' for i in range(num_channels)]
    
    # Compute statistics
    avg_vg = channel_importance_vg.mean(axis=0)
    avg_ig = channel_importance_ig.mean(axis=0)
    std_vg = channel_importance_vg.std(axis=0)
    std_ig = channel_importance_ig.std(axis=0)
    diff = avg_ig - avg_vg
    abs_diff = np.abs(diff)
    
    # Compute rankings
    ranking_vg = np.argsort(avg_vg)[::-1]
    ranking_ig = np.argsort(avg_ig)[::-1]
    rank_vg = np.zeros(num_channels, dtype=int)
    rank_ig = np.zeros(num_channels, dtype=int)
    for i, idx in enumerate(ranking_vg):
        rank_vg[idx] = i + 1
    for i, idx in enumerate(ranking_ig):
        rank_ig[idx] = i + 1
    rank_diff = rank_ig - rank_vg
    
    # Create DataFrame
    data = {
        'Channel': channel_names,
        'VG_Mean': avg_vg,
        'VG_Std': std_vg,
        'VG_Rank': rank_vg,
        'IG_Mean': avg_ig,
        'IG_Std': std_ig,
        'IG_Rank': rank_ig,
        'Difference': diff,
        'Abs_Difference': abs_diff,
        'Rank_Difference': rank_diff
    }
    
    df = pd.DataFrame(data)
    # Sort by absolute difference (descending)
    df = df.sort_values('Abs_Difference', ascending=False)
    
    df.to_csv(save_path, index=False)
    print(f"Saved method comparison report to {save_path}")
    
    # Print summary statistics
    correlation = np.corrcoef(avg_vg, avg_ig)[0, 1]
    print(f"\nMethod Comparison Summary:")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  Mean absolute difference: {abs_diff.mean():.6f}")
    print(f"  Max absolute difference: {abs_diff.max():.6f}")
    print(f"  Mean rank difference: {np.abs(rank_diff).mean():.2f}")
    
    return df


def plot_topography(channel_importance: np.ndarray,
                   channel_names: Optional[List[str]] = None,
                   title: str = "Channel Importance Topography",
                   save_path: Optional[str] = None,
                   figsize: Tuple[int, int] = (10, 8),
                   cmap: str = 'viridis',
                   vmin: Optional[float] = None,
                   vmax: Optional[float] = None,
                   show_colorbar: bool = True,
                   sensors: bool = True,
                   contours: int = 6,
                   outlines: str = 'head',
                   head_pos: Optional[dict] = None,
                   res: int = 128,
                   extrapolate: str = 'auto',
                   interp: str = 'cubic',
                   sphere: Optional[Union[float, tuple]] = None,
                   border: Union[str, float] = 'mean'):
    """
    Plot channel importance as a topography map on the scalp.
    
    This function creates a 2D scalp plot showing the spatial distribution
    of channel importance values. Uses MNE if available for high-quality
    topography, otherwise falls back to a simplified matplotlib approach.
    
    Args:
        channel_importance: Array of shape (num_channels,) with importance values
                          OR shape (num_samples, num_channels) for average across samples
        channel_names: List of channel names (if None, uses STANDARD_CHANNEL_NAMES)
        title: Plot title
        save_path: Path to save figure (None = don't save)
        figsize: Figure size
        cmap: Colormap name
        vmin: Minimum value for colormap (None = auto)
        vmax: Maximum value for colormap (None = auto)
        show_colorbar: Whether to show colorbar
        sensors: Whether to show sensor locations (MNE only)
        contours: Number of contour lines (MNE only)
        outlines: Head outline style ('head', 'skirt', 'head+skirt', or dict for custom) (MNE only)
        head_pos: Dictionary for head outline position (MNE only)
        res: Resolution of interpolation grid (higher = smoother, default: 128) (MNE only)
        extrapolate: Extrapolation method ('auto' for full scalp, 'box', 'head', or 'local') (MNE only)
        interp: Interpolation method ('cubic', 'linear', 'nearest') (MNE only)
        sphere: Sphere size for head model (None = auto, or tuple of (x, y, z, radius)) (MNE only)
        border: Border style ('mean' for full coverage, or float for padding) (MNE only)
    
    Returns:
        matplotlib figure object
    """
    # Handle input shape: if 2D, average across samples
    if channel_importance.ndim == 2:
        channel_importance = channel_importance.mean(axis=0)
    elif channel_importance.ndim != 1:
        raise ValueError(f"channel_importance must be 1D or 2D, got shape {channel_importance.shape}")
    
    num_channels = len(channel_importance)
    
    # Use standard channel names if not provided
    if channel_names is None:
        if num_channels == len(STANDARD_CHANNEL_NAMES):
            channel_names = STANDARD_CHANNEL_NAMES.copy()
        else:
            channel_names = [f'Channel {i+1}' for i in range(num_channels)]
            print(f"Warning: Using generic channel names. Expected {len(STANDARD_CHANNEL_NAMES)} channels, got {num_channels}.")
    
    # Use absolute values for topography (standard practice)
    channel_importance = np.abs(channel_importance)
    
    # Set colormap limits if not provided
    if vmin is None:
        vmin = channel_importance.min()
    if vmax is None:
        vmax = channel_importance.max()
    
    # MNE is REQUIRED for topography plotting
    if not MNE_AVAILABLE:
        raise RuntimeError(
            "MNE is required for topography plotting. "
            "Please install MNE and activate the conda environment: conda activate eeg_env"
        )
    
    # Use MNE for high-quality topography with accurate 10-20 system positions
    return _plot_topography_mne(
        channel_importance, channel_names, title, save_path,
        figsize, cmap, vmin, vmax, show_colorbar, sensors, contours,
        outlines, head_pos, res, extrapolate, interp, sphere, border
    )


def _add_colorbar_to_topography(im, ax: plt.Axes, show_colorbar: bool) -> None:
    """
    Add colorbar to topography plot.
    
    Args:
        im: Image/contour object from MNE plot
        ax: Matplotlib axes
        show_colorbar: Whether to show colorbar
    """
    if not show_colorbar:
        return
    
    try:
        if MNE_COLORBAR_AVAILABLE:
            mne_add_colorbar(im, ax=ax, pad=TOPOGRAPHY_COLORBAR_PAD, shrink=TOPOGRAPHY_COLORBAR_SHRINK)
        else:
            raise ImportError("MNE colorbar not available")
    except (ImportError, AttributeError, TypeError):
        # Fallback: use matplotlib's colorbar
        cbar = plt.colorbar(im, ax=ax, pad=TOPOGRAPHY_COLORBAR_PAD, shrink=TOPOGRAPHY_COLORBAR_SHRINK)
        cbar.set_label(TOPOGRAPHY_COLORBAR_LABEL, fontsize=FONT_SIZE_SMALL)


def _add_electrode_labels(ax: plt.Axes, info: mne.Info) -> None:
    """
    Add electrode labels to topography plot at their exact positions.
    
    Args:
        ax: Matplotlib axes
        info: MNE Info object with channel information
    """
    try:
        # Get 2D positions for all channels in info
        pos_2d = _find_topomap_coords(info, picks='eeg')
        
        # Get the channel names that are actually plotted
        plotted_ch_names = info.ch_names
        
        # Add text labels for each electrode at their exact positions
        for ch_name, (x, y) in zip(plotted_ch_names, pos_2d):
            # Place label directly at the electrode position (x, y)
            # Use a white background with transparency for visibility over the heatmap
            ax.text(x, y, ch_name, 
                   ha='center', va='center',
                   fontsize=TOPOGRAPHY_LABEL_FONTSIZE, fontweight='bold',
                   color='black', zorder=10,
                   bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                            edgecolor='black', alpha=0.85, linewidth=0.5))
    except Exception as e:
        print(f"Warning: Could not add electrode labels: {e}")


def _match_channels_to_montage(channel_names: List[str], montage) -> Tuple[List[str], List[int]]:
    """
    Match channel names to MNE montage channels (case-insensitive).
    
    Args:
        channel_names: List of channel names to match
        montage: MNE montage object
        
    Returns:
        Tuple of (matched_channel_names, matched_indices)
    """
    montage_ch_names_lower = [ch.lower() for ch in montage.ch_names]
    matched_chs = []
    matched_indices = []
    
    for i, ch in enumerate(channel_names):
        ch_lower = ch.lower()
        if ch_lower in montage_ch_names_lower:
            # Find the actual montage channel name (preserving case)
            montage_idx = montage_ch_names_lower.index(ch_lower)
            matched_chs.append(montage.ch_names[montage_idx])
            matched_indices.append(i)
    
    return matched_chs, matched_indices


def _plot_topography_mne(channel_importance: np.ndarray,
                        channel_names: List[str],
                        title: str,
                        save_path: Optional[str],
                        figsize: Tuple[int, int],
                        cmap: str,
                        vmin: float,
                        vmax: float,
                        show_colorbar: bool,
                        sensors: bool,
                        contours: int,
                        outlines: str = 'head',
                        head_pos: Optional[dict] = None,
                        res: int = 128,
                        extrapolate: str = 'auto',
                        interp: str = 'cubic',
                        sphere: Optional[Union[float, tuple]] = None,
                        border: Union[str, float] = 'mean'):
    """
    Plot topography using MNE (high-quality, standard approach).
    """
    # Create a fake Raw object with the channel data
    # MNE needs channel positions from a montage
    montage = make_standard_montage('standard_1020')
    
    # Create info object with channel names
    info = mne.create_info(ch_names=channel_names, sfreq=200, ch_types='eeg')
    
    # Set montage (this assigns 3D positions to channels)
    # Use match_case=False and on_missing='warn' to handle channel name variations
    try:
        info.set_montage(montage, match_case=False, on_missing='warn')
    except Exception as e:
        print(f"Warning: Could not set montage for all channels: {e}")
        print("Trying to match available channels...")
        # Try to match available channels in the montage using helper function
        matched_chs, matched_indices = _match_channels_to_montage(channel_names, montage)
        
        if len(matched_chs) < len(channel_names):
            print(f"Warning: Only {len(matched_chs)}/{len(channel_names)} channels found in montage.")
            print(f"Missing channels: {set(channel_names) - set([ch.lower() for ch in matched_chs])}")
            # Create a subset info with only available channels
            channel_importance = channel_importance[matched_indices]
            channel_names = matched_chs
            info = mne.create_info(ch_names=channel_names, sfreq=200, ch_types='eeg')
            info.set_montage(montage, match_case=False, on_missing='warn')
        else:
            # All channels found, but montage setting failed - try again with matched names
            info = mne.create_info(ch_names=matched_chs, sfreq=200, ch_types='eeg')
            info.set_montage(montage, match_case=False, on_missing='warn')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot topography with head outline
    # Handle matplotlib compatibility issues (e.g., QuadContourSet.collections)
    # Set default head position if not provided to ensure head outline is visible
    if head_pos is None:
        head_pos = dict(head_radius=0.5, head_width=0.2, head_length=0.2)
    
    # Set default sphere if not provided (standard head radius)
    if sphere is None:
        sphere = (0.0, 0.0, 0.0, 0.095)  # Standard head radius in meters
    
    # Build plot_topomap arguments
    # MNE API: plot_topomap(data, pos, *, ...) where pos can be Info object or array
    # Use 'auto' extrapolation to ensure full scalp coverage
    # Note: MNE uses vlim=(vmin, vmax) instead of separate vmin/vmax
    # Build plot_kwargs - only include valid MNE parameters
    # Note: MNE's plot_topomap doesn't accept 'head_pos' - it's handled via 'outlines' parameter
    plot_kwargs = {
        'axes': ax,
        'cmap': cmap,
        'vlim': (vmin, vmax),  # MNE uses vlim tuple, not separate vmin/vmax
        'show': False,
        'sensors': sensors,
        'contours': contours,
        'outlines': outlines,  # 'head', 'skirt', 'head+skirt', or dict for custom outlines
        'res': res,
        'extrapolate': extrapolate,  # 'auto' automatically determines best extrapolation for full scalp
        'image_interp': interp,  # Note: parameter name is 'image_interp', not 'interp'
        'sphere': sphere,
        'border': border  # 'mean' extends to head boundary for full coverage
    }
    
    # Note: 'head_pos' is not a valid parameter for plot_topomap
    # If custom head position is needed, it should be passed via 'outlines' as a dict
    
    try:
        # Correct API: plot_topomap(data, pos, ...) where pos is Info object
        # Note: MNE's plot_topomap doesn't accept colorbar/cbar parameter
        im, _ = mne.viz.plot_topomap(channel_importance, info, **plot_kwargs)
        
        # Add colorbar manually if requested (MNE doesn't have a direct colorbar parameter)
        _add_colorbar_to_topography(im, ax, show_colorbar)
    except AttributeError as e:
        if 'collections' in str(e) or 'QuadContourSet' in str(e):
            # Matplotlib version compatibility issue - try without contours
            print(f"Warning: Matplotlib compatibility issue detected ({e}). "
                  "Trying without contour lines...")
            try:
                # Retry with contours disabled
                plot_kwargs['contours'] = 0
                im, _ = mne.viz.plot_topomap(channel_importance, info, **plot_kwargs)
                # Add colorbar if requested
                _add_colorbar_to_topography(im, ax, show_colorbar)
            except Exception as e2:
                raise RuntimeError(
                    f"MNE topography plotting failed due to matplotlib compatibility: {e2}. "
                    "Consider updating MNE or matplotlib, or the code will fall back to matplotlib-based plotting."
                ) from e2
        else:
            raise
    
    # Add electrode labels directly at their electrode positions on the scalp
    _add_electrode_labels(ax, info)
    
    ax.set_title(title, fontsize=TOPOGRAPHY_TITLE_FONTSIZE, fontweight='bold', pad=TOPOGRAPHY_TITLE_PAD)
    plt.tight_layout()
    
    _save_figure(save_path, "topography plot")
    
    return fig


def _plot_topography_matplotlib(channel_importance: np.ndarray,
                                channel_names: List[str],
                                title: str,
                                save_path: Optional[str],
                                figsize: Tuple[int, int],
                                cmap: str,
                                vmin: float,
                                vmax: float,
                                show_colorbar: bool):
    """
    Plot topography using matplotlib with MNE positions.
    
    This function uses MNE's standard_1020 montage to get accurate 10-20 system positions.
    MNE is REQUIRED - this function will not work without it.
    """
    if not MNE_AVAILABLE:
        raise RuntimeError(
            "MNE is required for topography plotting. "
            "Please install MNE and activate the conda environment: conda activate eeg_env"
        )
    
    try:
        from scipy.interpolate import griddata
        SCIPY_AVAILABLE = True
    except ImportError:
        SCIPY_AVAILABLE = False
        print("Warning: scipy not available. Using scatter plot instead of continuous heatmap.")
    
    # Get actual channel positions from MNE standard_1020 montage
    # This ensures accurate positioning covering full scalp according to 10-20 system
    use_mne_positions = False
    
    try:
        montage = make_standard_montage('standard_1020')
        # Try to match channels and get their 2D positions
        valid_channels = []
        valid_indices = []
        valid_channel_importance = []
        
        # Use helper function to match channels
        valid_channels, valid_indices = _match_channels_to_montage(channel_names, montage)
        valid_channel_importance = [channel_importance[i] for i in valid_indices]
        
        if len(valid_channels) < len(channel_names) * 0.8:  # Less than 80% match
            raise ValueError(
                f"Only {len(valid_channels)}/{len(channel_names)} channels matched in MNE standard_1020 montage. "
                f"Missing channels: {set(channel_names) - set([ch.lower() for ch in valid_channels])}. "
                "Please check channel names match the 10-20 system."
            )
        
        # Use MNE positions for matched channels
        info_temp = mne.create_info(ch_names=valid_channels, sfreq=200, ch_types='eeg')
        info_temp.set_montage(montage, match_case=False, on_missing='warn')
        pos_2d = _find_topomap_coords(info_temp, picks='eeg')
        
        # Extract x, y coordinates
        x_coords = pos_2d[:, 0]
        y_coords = pos_2d[:, 1]
        
        # Normalize to fit in unit circle (preserve aspect ratio)
        max_dist = np.sqrt(x_coords**2 + y_coords**2).max()
        if max_dist > 0:
            x_coords = x_coords / max_dist * INTERPOLATION_NORMALIZATION_FACTOR
            y_coords = y_coords / max_dist * INTERPOLATION_NORMALIZATION_FACTOR
        
        points = np.column_stack([x_coords, y_coords])
        values = np.array(valid_channel_importance)
        use_mne_positions = True
        print(f"Using MNE standard_1020 positions for {len(valid_channels)}/{len(channel_names)} channels.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to extract MNE positions: {e}. "
            "MNE with standard_1020 montage is required for accurate 10-20 system electrode positions. "
            "Please ensure MNE is installed and the conda environment is activated: conda activate eeg_env"
        ) from e
    
    # MNE positions should always be used - if we get here, something went wrong
    if not use_mne_positions:
        raise RuntimeError(
            "Failed to obtain MNE positions. This should not happen. "
            "Please ensure MNE is properly installed and the conda environment is activated: conda activate eeg_env"
        )
    
    # Use MNE positions - already converted above
    fig, ax = plt.subplots(figsize=figsize)
    
    if SCIPY_AVAILABLE:
        # Create a fine grid for continuous heatmap covering full circular scalp
        # Create circular grid that covers full scalp (symmetric around origin)
        x_grid = np.linspace(-INTERPOLATION_GRID_RANGE, INTERPOLATION_GRID_RANGE, INTERPOLATION_GRID_RESOLUTION)
        y_grid = np.linspace(-INTERPOLATION_GRID_RANGE, INTERPOLATION_GRID_RANGE, INTERPOLATION_GRID_RESOLUTION)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        
        # Calculate distance from center for each grid point
        R_grid = np.sqrt(X_grid**2 + Y_grid**2)
        
        # Interpolate values onto the grid (cubic interpolation for smoothness)
        try:
            Z_grid = griddata(points, values, grid_points, method='cubic', fill_value=np.nan)
            Z_grid = Z_grid.reshape(X_grid.shape)
            
            # For areas outside sensor coverage but inside head, use nearest neighbor extrapolation
            # This ensures full scalp coverage across entire circular scalp
            mask_outside = R_grid > 1.0
            mask_inside = R_grid <= 1.0
            mask_no_data = np.isnan(Z_grid) & mask_inside
            
            if mask_no_data.any():
                # Fill gaps inside head with nearest neighbor interpolation to ensure full coverage
                try:
                    # Get valid data points (not NaN)
                    valid_mask = ~np.isnan(Z_grid.ravel())
                    valid_points = grid_points[valid_mask]
                    valid_values = Z_grid.ravel()[valid_mask]
                    
                    if len(valid_points) > 0:
                        # Fill gaps with nearest neighbor
                        Z_grid_filled = griddata(
                            valid_points, valid_values, 
                            grid_points[mask_no_data.ravel()], 
                            method='nearest', 
                            fill_value=np.nan
                        )
                        Z_grid_flat = Z_grid.ravel()
                        Z_grid_flat[mask_no_data.ravel()] = Z_grid_filled
                        Z_grid = Z_grid_flat.reshape(X_grid.shape)
                except Exception as e:
                    print(f"Warning: Gap filling failed ({e}), some areas may not be covered.")
            
            # Mask values outside the head (radius > 1.0) - full circular scalp
            Z_grid[mask_outside] = np.nan
            
            # Ensure we have data covering the full circular area
            # If there are still large gaps, use a fallback approach
            if np.isnan(Z_grid[mask_inside]).sum() > mask_inside.sum() * 0.3:
                # Too many gaps - try filling with a smoother approach
                from scipy.ndimage import gaussian_filter
                try:
                    # Create a mask for valid data
                    valid_data = ~np.isnan(Z_grid)
                    if valid_data.sum() > 0:
                        # Smooth and extrapolate
                        Z_filled = Z_grid.copy()
                        Z_filled[~valid_data & mask_inside] = np.nanmean(Z_grid[valid_data])
                        Z_filled = gaussian_filter(Z_filled, sigma=2)
                        Z_filled[mask_outside] = np.nan
                        Z_grid = Z_filled
                except Exception:
                    pass  # If smoothing fails, use original
            
        except Exception as e:
            # Fallback to linear interpolation if cubic fails
            print(f"Warning: Cubic interpolation failed ({e}), using linear interpolation.")
            Z_grid = griddata(points, values, grid_points, method='linear', fill_value=np.nan)
            Z_grid = Z_grid.reshape(X_grid.shape)
            R_grid = np.sqrt(X_grid**2 + Y_grid**2)
            Z_grid[R_grid > 1.0] = np.nan
        
        # Create continuous heatmap covering full circular scalp
        # Use many levels for smooth gradient
        im = ax.contourf(X_grid, Y_grid, Z_grid, levels=100, cmap=cmap, 
                         vmin=vmin, vmax=vmax, extend='both', alpha=0.9)
        
        # Add channel locations as points
        scatter = ax.scatter(x_coords, y_coords, c=values, cmap=cmap, 
                            s=100, edgecolors='black', linewidths=1.5,
                            vmin=vmin, vmax=vmax, zorder=5)
    else:
        # Fallback: use scatter plot if scipy is not available
        scatter = ax.scatter(x_coords, y_coords, c=values, cmap=cmap, 
                            s=200, edgecolors='black', linewidths=1.5,
                            vmin=vmin, vmax=vmax, zorder=5, alpha=0.8)
        im = scatter  # For colorbar compatibility
    
    # Add channel labels
    for x, y, name in zip(x_coords, y_coords, channel_names):
        # Position label slightly outside the head
        label_radius = np.sqrt(x**2 + y**2) + 0.15
        label_x = (x / np.sqrt(x**2 + y**2 + 1e-10)) * label_radius
        label_y = (y / np.sqrt(x**2 + y**2 + 1e-10)) * label_radius
        ax.text(label_x, label_y, name, ha='center', va='center', 
               fontsize=FONT_SIZE_ELECTRODE_LABEL, fontweight='bold', zorder=6)
    
    # Draw head outline (circle)
    theta_head = np.linspace(0, 2 * np.pi, 100)
    x_head = np.cos(theta_head)
    y_head = np.sin(theta_head)
    ax.plot(x_head, y_head, 'k-', linewidth=3, zorder=4)
    
    # Draw nose (top of circle)
    nose_y = 1.05
    nose_x = 0
    ax.plot([nose_x - 0.05, nose_x, nose_x + 0.05], 
            [nose_y, nose_y + 0.1, nose_y], 'k-', linewidth=3, zorder=4)
    
    # Draw ears (left and right sides)
    # Left ear
    ear_theta_left = np.linspace(np.pi/2 - 0.15, np.pi/2 + 0.15, 20)
    ear_x_left = np.cos(ear_theta_left) * 1.02
    ear_y_left = np.sin(ear_theta_left) * 1.02
    ax.plot(ear_x_left, ear_y_left, 'k-', linewidth=2, zorder=4)
    
    # Right ear
    ear_theta_right = np.linspace(-np.pi/2 - 0.15, -np.pi/2 + 0.15, 20)
    ear_x_right = np.cos(ear_theta_right) * 1.02
    ear_y_right = np.sin(ear_theta_right) * 1.02
    ax.plot(ear_x_right, ear_y_right, 'k-', linewidth=2, zorder=4)
    
    # Set equal aspect and limits
    ax.set_aspect('equal')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis('off')
    
    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, pad=COLORBAR_PAD, shrink=COLORBAR_SHRINK)
        cbar.set_label(COLORBAR_LABEL, fontsize=FONT_SIZE_SMALL)
    
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=TOPOGRAPHY_TITLE_PAD)
    
    plt.tight_layout()
    
    _save_figure(save_path, "topography plot")
    
    return fig


def _get_approximate_channel_positions(channel_names: List[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Get approximate polar coordinates (angles, radii) for channel names based on 10-20 system.
    
    This function provides approximate positions when MNE is not available.
    For accurate positions, MNE's standard_1020 montage should be used.
    
    Returns:
        Tuple of (angles, radii) in radians, or None if mapping fails
    """
    # 10-20 system mapping based on standard electrode placement
    # Coordinate system: Standard mathematical polar coordinates
    # Angles: 0 = right (x>0, y=0), π/2 = top/front (x=0, y>0), π = left (x<0, y=0), 3π/2 = bottom/back (x=0, y<0)
    # Radii: 0 = center, 1 = edge
    # 
    # Standard 10-20 system layout:
    # - Front to back: FP (front) → F → C → P → O (back)
    # - Left to right: odd numbers (left, negative x) → Z (midline, x=0) → even numbers (right, positive x)
    # - Number indicates distance from midline: 1,2 (close), 3,4 (medium), 5,6,7,8 (lateral)
    
    # 10-20 system positions using a more systematic approach
    # Key principle: Use Cartesian-like positioning then convert to polar
    # Front-to-back axis: y coordinate (positive = front, negative = back)
    # Left-to-right axis: x coordinate (negative = left, positive = right)
    
    # Helper function to convert (x, y) to (angle, radius)
    def xy_to_polar(x, y):
        angle = np.arctan2(y, x)  # Standard: atan2(y, x) gives angle from positive x-axis
        radius = np.sqrt(x**2 + y**2)
        return angle, radius
    
    # Define positions in (x, y) coordinates first, then convert
    # x: negative = left, positive = right, 0 = midline
    # y: positive = front, negative = back, 0 = center
    # Values normalized to roughly fit in unit circle
    
    position_map_xy = {
        # Frontal Pole (very front)
        'FP1': (-0.3, 0.95), 'FP2': (0.3, 0.95),
        
        # Anterior Frontal
        'AF3': (-0.25, 0.9), 'AF4': (0.25, 0.9),
        'AF7': (-0.4, 0.85), 'AF8': (0.4, 0.85),
        'AFZ': (0.0, 0.9),
        
        # Frontal
        'F1': (-0.2, 0.85), 'F2': (0.2, 0.85),
        'F3': (-0.35, 0.8), 'F4': (0.35, 0.8),
        'FZ': (0.0, 0.85),
        'F5': (-0.45, 0.75), 'F6': (0.45, 0.75),
        'F7': (-0.55, 0.7), 'F8': (0.55, 0.7),
        'F9': (-0.6, 0.65), 'F10': (0.6, 0.65),
        
        # Frontal-Central
        'FC1': (-0.2, 0.7), 'FC2': (0.2, 0.7),
        'FC3': (-0.35, 0.65), 'FC4': (0.35, 0.65),
        'FC5': (-0.5, 0.6), 'FC6': (0.5, 0.6),
        
        # Frontal-Temporal / Temporal
        'FT7': (-0.7, 0.5), 'FT8': (0.7, 0.5),
        'T7': (-0.75, 0.4), 'T8': (0.75, 0.4),
        'T9': (-0.8, 0.3), 'T10': (0.8, 0.3),
        
        # Central
        'C1': (-0.2, 0.5), 'C2': (0.2, 0.5),
        'C3': (-0.4, 0.45), 'C4': (0.4, 0.45),
        'C5': (-0.55, 0.4), 'C6': (0.55, 0.4),
        'CZ': (0.0, 0.5),
        
        # Central-Parietal (transitioning to back)
        'CP1': (-0.2, 0.3), 'CP2': (0.2, 0.3),
        'CP3': (-0.35, 0.25), 'CP4': (0.35, 0.25),
        'CP5': (-0.5, 0.2), 'CP6': (0.5, 0.2),
        'CPZ': (0.0, 0.3),
        
        # Parietal (back of head)
        'P1': (-0.2, 0.1), 'P2': (0.2, 0.1),
        'P3': (-0.35, 0.05), 'P4': (0.35, 0.05),
        'P5': (-0.5, 0.0), 'P6': (0.5, 0.0),
        'P7': (-0.6, -0.05), 'P8': (0.6, -0.05),
        'PZ': (0.0, 0.1),
        
        # Parietal-Occipital
        'PO3': (-0.3, -0.1), 'PO4': (0.3, -0.1),
        'PO7': (-0.45, -0.15), 'PO8': (0.45, -0.15),
        'POZ': (0.0, -0.1),
        
        # Occipital (very back)
        'O1': (-0.25, -0.2), 'O2': (0.25, -0.2),
        'OZ': (0.0, -0.2),
    }
    
    # Convert (x, y) positions to (angle, radius)
    position_map = {}
    for ch_name, (x, y) in position_map_xy.items():
        angle, radius = xy_to_polar(x, y)
        position_map[ch_name] = (angle, radius)
    
    angles = []
    radii = []
    
    for ch_name in channel_names:
        if ch_name in position_map:
            angle, radius = position_map[ch_name]
            angles.append(angle)
            radii.append(radius)
        else:
            # If channel not found, return None to use circular layout
            return None
    
    return np.array(angles), np.array(radii)


