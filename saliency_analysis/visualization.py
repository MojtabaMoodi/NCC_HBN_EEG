"""
Visualization utilities for saliency map analysis.

Provides functions to visualize channel importance, create heatmaps,
and generate summary statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
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

# Try to import MNE for topography plotting
try:
    import mne
    from mne.channels import make_standard_montage
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("Warning: MNE not available. Topography plots will use a simplified matplotlib-based approach.")


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
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('EEG Channel', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    
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
    plt.xlabel('EEG Channel', fontsize=12)
    plt.ylabel('Average Importance', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, sorted_importance)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved bar chart to {save_path}")
    
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
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('EEG Channel', fontsize=12)
    plt.ylabel('Importance Value', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved distribution plot to {save_path}")
    
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
    plt.xlabel('Time Point', fontsize=12)
    plt.ylabel('EEG Channel', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.yticks(range(num_channels), channel_names)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved saliency map to {save_path}")
    
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
        ax.set_xlabel('Average Importance', fontsize=10)
        ax.set_title(f'{cls_name}', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()  # Top channel at top
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
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
    ax1.set_xlabel('Channel', fontsize=11)
    ax1.set_ylabel('Average Importance', fontsize=11)
    ax1.set_title(f'Top {top_k} Channels Comparison', fontsize=12, fontweight='bold')
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
    ax2.set_xlabel('Vanilla Gradients Importance', fontsize=11)
    ax2.set_ylabel('Integrated Gradients Importance', fontsize=11)
    ax2.set_title('Method Correlation (All Channels)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Compute correlation
    correlation = np.corrcoef(avg_vg, avg_ig)[0, 1]
    ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Difference plot (IG - VG)
    ax3 = axes[1, 0]
    diff = avg_ig - avg_vg
    sorted_indices = np.argsort(np.abs(diff))[::-1][:top_k]
    colors = ['green' if diff[i] > 0 else 'red' for i in sorted_indices]
    ax3.barh(range(len(sorted_indices)), diff[sorted_indices], color=colors, alpha=0.7)
    ax3.set_yticks(range(len(sorted_indices)))
    ax3.set_yticklabels([channel_names[i] for i in sorted_indices])
    ax3.set_xlabel('Difference (IG - VG)', fontsize=11)
    ax3.set_title(f'Top {top_k} Channels by Absolute Difference', fontsize=12, fontweight='bold')
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
                    fontsize=8, alpha=0.7)
    ax4.plot([1, top_k_for_ranking], [1, top_k_for_ranking], 'r--', alpha=0.5, label='Same ranking')
    ax4.set_xlabel('Vanilla Gradients Rank', fontsize=11)
    ax4.set_ylabel('Integrated Gradients Rank', fontsize=11)
    ax4.set_title('Ranking Comparison (Lower = Better)', fontsize=12, fontweight='bold')
    ax4.invert_xaxis()
    ax4.invert_yaxis()
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved method comparison plot to {save_path}")
    
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
                   contours: int = 6):
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
    
    # Try to use MNE for high-quality topography
    if MNE_AVAILABLE:
        try:
            return _plot_topography_mne(
                channel_importance, channel_names, title, save_path,
                figsize, cmap, vmin, vmax, show_colorbar, sensors, contours
            )
        except Exception as e:
            print(f"Warning: MNE topography plotting failed ({e}). Falling back to matplotlib approach.")
            # Fall through to matplotlib approach
    
    # Fallback: matplotlib-based approach
    return _plot_topography_matplotlib(
        channel_importance, channel_names, title, save_path,
        figsize, cmap, vmin, vmax, show_colorbar
    )


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
                        contours: int):
    """
    Plot topography using MNE (high-quality, standard approach).
    """
    # Create a fake Raw object with the channel data
    # MNE needs channel positions from a montage
    montage = make_standard_montage('standard_1020')
    
    # Create info object with channel names
    info = mne.create_info(ch_names=channel_names, sfreq=200, ch_types='eeg')
    
    # Set montage (this assigns 3D positions to channels)
    try:
        info.set_montage(montage, match_case=False, on_missing='warn')
    except Exception as e:
        print(f"Warning: Could not set montage for all channels: {e}")
        print("Trying to match available channels...")
        # Try to match available channels in the montage
        available_chs = [ch for ch in channel_names if ch in montage.ch_names]
        if len(available_chs) < len(channel_names):
            print(f"Warning: Only {len(available_chs)}/{len(channel_names)} channels found in montage.")
            # Create a subset info with only available channels
            subset_indices = [i for i, ch in enumerate(channel_names) if ch in available_chs]
            channel_importance = channel_importance[subset_indices]
            channel_names = available_chs
            info = mne.create_info(ch_names=channel_names, sfreq=200, ch_types='eeg')
            info.set_montage(montage, match_case=False, on_missing='warn')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot topography
    im, _ = mne.viz.plot_topomap(
        channel_importance,
        info,
        axes=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        show=False,
        sensors=sensors,
        contours=contours,
        colorbar=show_colorbar
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved topography plot to {save_path}")
    
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
    Plot topography using matplotlib (simplified fallback when MNE is not available).
    
    This creates a circular scalp plot with approximate channel positions.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    # Define approximate positions for standard 10-20 channels
    # This is a simplified mapping - MNE provides more accurate positions
    channel_positions = _get_approximate_channel_positions(channel_names)
    
    if channel_positions is None:
        # If we can't map channels, create a simple circular layout
        print("Warning: Could not map channel names to positions. Using circular layout.")
        angles = np.linspace(0, 2 * np.pi, len(channel_names), endpoint=False)
        radii = np.ones(len(channel_names)) * 0.8
    else:
        angles, radii = channel_positions
    
    # Normalize importance values for visualization
    normalized_importance = (channel_importance - vmin) / (vmax - vmin + 1e-10)
    
    # Create scatter plot with color-coded importance
    scatter = ax.scatter(angles, radii, c=normalized_importance, 
                        cmap=cmap, s=200, alpha=0.8, edgecolors='black', linewidths=1.5)
    
    # Add channel labels
    for angle, radius, name, importance in zip(angles, radii, channel_names, channel_importance):
        # Position label slightly outside the circle
        label_radius = radius + 0.15
        ax.text(angle, label_radius, name, ha='center', va='center', 
               fontsize=8, fontweight='bold')
    
    # Draw head outline (circle) - for polar plot, we draw it as a circle at radius 1
    theta_head = np.linspace(0, 2 * np.pi, 100)
    ax.plot(theta_head, np.ones_like(theta_head), 'k-', linewidth=2)
    
    # Draw nose (top of circle) - at angle 0 (front)
    ax.plot([0, 0], [0.95, 1.05], 'k-', linewidth=2)
    
    # Set limits
    ax.set_ylim(0, 1.3)
    ax.set_theta_zero_location('N')  # Nose at top
    ax.set_theta_direction(-1)  # Clockwise
    
    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Channel Importance', fontsize=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')  # Hide polar axis lines
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved topography plot to {save_path}")
    
    return fig


def _get_approximate_channel_positions(channel_names: List[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Get approximate polar coordinates (angles, radii) for channel names.
    
    Returns:
        Tuple of (angles, radii) in radians, or None if mapping fails
    """
    # Simplified mapping based on standard 10-20 system
    # Angles: 0 = front (nose), increasing clockwise
    # Radii: 0 = center, 1 = edge
    
    position_map = {
        # Frontal
        'FP1': (np.pi * 0.25, 0.95), 'FP2': (np.pi * 0.75, 0.95),
        'F1': (np.pi * 0.3, 0.85), 'F2': (np.pi * 0.7, 0.85),
        'F3': (np.pi * 0.2, 0.8), 'F4': (np.pi * 0.8, 0.8),
        'FZ': (0, 0.85), 'F5': (np.pi * 0.15, 0.75), 'F6': (np.pi * 0.85, 0.75),
        'F7': (np.pi * 0.1, 0.7), 'F8': (np.pi * 0.9, 0.7),
        'F9': (np.pi * 0.05, 0.65), 'F10': (np.pi * 0.95, 0.65),
        
        # Anterior Frontal
        'AF3': (np.pi * 0.25, 0.9), 'AF4': (np.pi * 0.75, 0.9),
        'AF7': (np.pi * 0.15, 0.8), 'AF8': (np.pi * 0.85, 0.8),
        'AFZ': (0, 0.9),
        
        # Frontal-Central
        'FC1': (np.pi * 0.3, 0.7), 'FC2': (np.pi * 0.7, 0.7),
        'FC3': (np.pi * 0.2, 0.65), 'FC4': (np.pi * 0.8, 0.65),
        'FC5': (np.pi * 0.15, 0.6), 'FC6': (np.pi * 0.85, 0.6),
        
        # Temporal
        'FT7': (np.pi * 0.1, 0.5), 'FT8': (np.pi * 0.9, 0.5),
        'T7': (np.pi * 0.05, 0.4), 'T8': (np.pi * 0.95, 0.4),
        'T9': (np.pi * 0.02, 0.3), 'T10': (np.pi * 0.98, 0.3),
        
        # Central
        'C1': (np.pi * 0.3, 0.5), 'C2': (np.pi * 0.7, 0.5),
        'C3': (np.pi * 0.2, 0.45), 'C4': (np.pi * 0.8, 0.45),
        'C5': (np.pi * 0.15, 0.4), 'C6': (np.pi * 0.85, 0.4),
        'CZ': (0, 0.5),
        
        # Central-Parietal
        'CP1': (np.pi * 0.3, 0.3), 'CP2': (np.pi * 0.7, 0.3),
        'CP3': (np.pi * 0.2, 0.35), 'CP4': (np.pi * 0.8, 0.35),
        'CP5': (np.pi * 0.15, 0.4), 'CP6': (np.pi * 0.85, 0.4),
        'CPZ': (0, 0.3),
        
        # Parietal
        'P1': (np.pi * 0.3, 0.2), 'P2': (np.pi * 0.7, 0.2),
        'P3': (np.pi * 0.2, 0.25), 'P4': (np.pi * 0.8, 0.25),
        'P5': (np.pi * 0.15, 0.3), 'P6': (np.pi * 0.85, 0.3),
        'P7': (np.pi * 0.1, 0.35), 'P8': (np.pi * 0.9, 0.35),
        'PZ': (0, 0.2),
        
        # Parietal-Occipital
        'PO3': (np.pi * 0.25, 0.15), 'PO4': (np.pi * 0.75, 0.15),
        'PO7': (np.pi * 0.15, 0.2), 'PO8': (np.pi * 0.85, 0.2),
        'POZ': (0, 0.15),
        
        # Occipital
        'O1': (np.pi * 0.25, 0.05), 'O2': (np.pi * 0.75, 0.05),
        'OZ': (0, 0.05),
    }
    
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


