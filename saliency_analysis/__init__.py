"""
Saliency Analysis Module for EEG Channel Importance

This module provides tools for analyzing which EEG channels are most important
for model predictions using gradient-based saliency methods.
"""

from .saliency_methods import (
    SaliencyMethod,
    VanillaGradients,
    IntegratedGradients,
    aggregate_channel_importance,
    compute_batch_saliency
)

from .analyzer import SaliencyAnalyzer

from .visualization import (
    plot_channel_importance_heatmap,
    plot_average_channel_importance,
    plot_channel_importance_distribution,
    plot_saliency_map_sample,
    plot_comparison_by_class,
    save_channel_ranking,
    plot_method_comparison,
    save_method_comparison_report
)

__all__ = [
    # Saliency methods
    'SaliencyMethod',
    'VanillaGradients',
    'IntegratedGradients',
    'aggregate_channel_importance',
    'compute_batch_saliency',
    
    # Analyzer
    'SaliencyAnalyzer',
    
    # Visualization
    'plot_channel_importance_heatmap',
    'plot_average_channel_importance',
    'plot_channel_importance_distribution',
    'plot_saliency_map_sample',
    'plot_comparison_by_class',
    'save_channel_ranking',
    'plot_method_comparison',
    'save_method_comparison_report',
]

__version__ = '1.0.0'


