"""
Main Saliency Analyzer Class

Integrates saliency computation, analysis, and visualization for EEG channel importance.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# Add parent directory and CNN directory to path for imports
# Parent directory allows: from CNN.trainer import ...
# CNN directory allows relative imports within CNN module (from models import ...)
parent_dir = Path(__file__).parent.parent
cnn_dir = parent_dir / 'CNN'
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
if str(cnn_dir) not in sys.path:
    sys.path.insert(0, str(cnn_dir))

from saliency_methods import (
    SaliencyMethod, VanillaGradients, IntegratedGradients,
    compute_batch_saliency, aggregate_channel_importance
)
from visualization import (
    plot_channel_importance_heatmap,
    plot_average_channel_importance,
    plot_channel_importance_distribution,
    plot_saliency_map_sample,
    plot_comparison_by_class,
    save_channel_ranking
)

from CNN.trainer import load_checkpoint as trainer_load_checkpoint
from CNN.gpu_utils import get_underlying_model as gpu_get_underlying_model, get_device


class SaliencyAnalyzer:
    """
    Main class for performing saliency analysis on EEG models.
    """
    
    def __init__(self,
                 model: nn.Module,
                 target_type: str = 'gender',
                 device: Optional[torch.device] = None,
                 num_channels: int = 60,
                 device_preference: str = 'auto',
                 prediction_type: str = 'classification'):
        """
        Initialize saliency analyzer.
        
        Args:
            model: Trained model instance (may be wrapped with DataParallel)
            target_type: Type of prediction ('gender', 'age', 'combined')
            device: Device to use (None = auto-detect from device_preference)
            num_channels: Number of EEG channels
            device_preference: Device preference ('auto', 'cuda', 'cpu')
            prediction_type: Type of prediction ('classification' or 'regression')
        
        Raises:
            RuntimeError: If CUDA is requested but not available
        """
        self.model = model
        self.target_type = target_type
        self.num_channels = num_channels
        self.prediction_type = prediction_type
        
        # Setup device (explicit, no fallbacks)
        if device is None:
            self.device = get_device(device_preference)
        else:
            self.device = device
        
        # Move model to device and set to eval mode
        # Note: Model may already be wrapped with DataParallel, which is fine
        self.model.to(self.device)
        self.model.eval()
        
        # Results storage
        self.results = {}
    
    def _get_underlying_model(self):
        """Get underlying model, unwrapping DataParallel and torch.compile wrappers."""
        return gpu_get_underlying_model(self.model)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        underlying_model = self._get_underlying_model()
        checkpoint = trainer_load_checkpoint(checkpoint_path, underlying_model, self.device)
        self.model.eval()
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
        return checkpoint
    
    def compute_saliency(self,
                        data_loader: DataLoader,
                        method: str,
                        max_samples: Optional[int] = None,
                        **method_kwargs) -> Dict[str, np.ndarray]:
        """
        Compute saliency maps for samples in data loader.
        
        Args:
            data_loader: DataLoader providing batches
            method: Saliency method ('vanilla_gradients' or 'integrated_gradients')
            max_samples: Maximum number of samples to process
            **method_kwargs: Additional arguments for saliency method
        
        Returns:
            Dictionary with saliency results
        """
        # Create saliency method with appropriate kwargs
        if method == 'vanilla_gradients':
            # VanillaGradients only accepts abs_gradients
            filtered_kwargs = {k: v for k, v in method_kwargs.items() if k == 'abs_gradients'}
            saliency_method = VanillaGradients(**filtered_kwargs)
        elif method == 'integrated_gradients':
            # IntegratedGradients accepts num_steps, baseline, abs_gradients
            filtered_kwargs = {k: v for k, v in method_kwargs.items() 
                             if k in ('num_steps', 'baseline', 'abs_gradients')}
            saliency_method = IntegratedGradients(**filtered_kwargs)
        else:
            raise ValueError(f"Unknown saliency method: {method}")
        
        print(f"Computing saliency using {method}...")
        print(f"Target type: {self.target_type}")
        
        # Compute saliency
        # Note: We need to determine prediction_type from the model or pass it as a parameter
        # For now, we'll infer it from target_type and model output shape
        # This should be passed from the caller (main.py) which knows the model type
        results = compute_batch_saliency(
            model=self.model,
            data_loader=data_loader,
            saliency_method=saliency_method,
            target_key=self.target_type,
            max_samples=max_samples,
            device=self.device,
            device_preference='auto',  # Already set in __init__, but pass for consistency
            prediction_type=getattr(self, 'prediction_type', 'classification')
        )
        
        self.results[method] = results
        print(f"✅ Computed saliency for {len(results['labels'])} samples")
        
        # Print task type distribution if available
        if 'task_types' in results:
            task_types = results['task_types']
            unique, counts = np.unique(task_types, return_counts=True)
            print(f"Task type distribution:")
            for task_type, count in zip(unique, counts):
                print(f"  {task_type}: {count} samples")
        
        return results
    
    def analyze_channel_importance(self,
                                  results: Optional[Dict[str, np.ndarray]] = None,
                                  method: str = '') -> Dict[str, np.ndarray]:
        """
        Analyze channel importance from saliency results.
        
        Note: Channel importance is already aggregated (mean) during saliency computation.
        This method analyzes the pre-aggregated channel importance values.
        
        Args:
            results: Saliency results (if None, uses stored results)
            method: Method name to use from stored results
        
        Returns:
            Dictionary with analysis results
        """
        if results is None:
            if method not in self.results:
                raise ValueError(f"No results found for method: {method}. "
                               "Run compute_saliency() first.")
            results = self.results[method]
        
        channel_importance = results['channel_importance']
        labels = results['labels']
        predictions = results['predictions']
        
        # Compute statistics
        avg_importance = channel_importance.mean(axis=0)
        std_importance = channel_importance.std(axis=0)
        
        # Rank channels
        # For Integrated Gradients, use absolute value for ranking since negative values
        # indicate channels that reduce predictions (but are still influential)
        # For Vanilla Gradients, values are already non-negative
        if method == 'integrated_gradients':
            # Rank by absolute value to find most influential channels (positive or negative)
            ranked_indices = np.argsort(np.abs(avg_importance))[::-1]
        else:
            # Vanilla gradients are already non-negative
            ranked_indices = np.argsort(avg_importance)[::-1]
        
        analysis = {
            'channel_importance': channel_importance,
            'average_importance': avg_importance,
            'std_importance': std_importance,
            'ranked_indices': ranked_indices,
            'labels': labels,
            'predictions': predictions,
            'num_samples': len(labels)
        }
        
        print(f"\nTop 10 Most Important Channels:")
        print("-" * 60)
        for rank, idx in enumerate(ranked_indices[:10], 1):
            print(f"Rank {rank:2d}: Channel {idx+1:2d} "
                  f"(Mean: {avg_importance[idx]:.6f}, "
                  f"Std: {std_importance[idx]:.6f})")
        
        return analysis
    
    def visualize_results(self,
                         output_dir: str,
                         method: str = 'integrated_gradients',
                         channel_names: Optional[List[str]] = None,
                         class_names: Optional[List[str]] = None,
                         top_k: int = 10,
                         by_task_type: bool = True):
        """
        Generate all visualization plots.
        
        Args:
            output_dir: Directory to save plots
            method: Method name to visualize
            channel_names: List of channel names
            class_names: List of class names
            top_k: Number of top channels to show
            by_task_type: If True, generate separate visualizations for each task type
        """
        if method not in self.results:
            raise ValueError(f"No results found for method: {method}")
        
        results = self.results[method]
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze channel importance
        analysis = self.analyze_channel_importance(results=results, method=method)
        
        channel_importance = analysis['channel_importance']
        labels = analysis['labels']
        
        print(f"\nGenerating visualizations...")
        
        # 1. Average channel importance bar chart (overall)
        plot_average_channel_importance(
            channel_importance,
            channel_names=channel_names,
            title=f"Average Channel Importance ({self.target_type.capitalize()}) - All Tasks",
            save_path=os.path.join(output_dir, f"average_importance_{method}.png"),
            top_k=top_k
        )
        
        # 2. Channel importance distribution (overall)
        plot_channel_importance_distribution(
            channel_importance,
            channel_names=channel_names,
            title=f"Channel Importance Distribution ({self.target_type.capitalize()}) - All Tasks",
            save_path=os.path.join(output_dir, f"importance_distribution_{method}.png"),
            top_k=top_k
        )
        
        # 3. Comparison by class (if classification) - overall
        if self.target_type in ['gender', 'age', 'combined']:
            plot_comparison_by_class(
                channel_importance,
                labels,
                class_names=class_names,
                title=f"Channel Importance by Class ({self.target_type.capitalize()}) - All Tasks",
                save_path=os.path.join(output_dir, f"importance_by_class_{method}.png"),
                top_k=top_k
            )
        
        # 4. Sample saliency maps (first few samples)
        saliency_maps = results['saliency_maps']
        num_samples_to_plot = min(5, len(saliency_maps))
        for i in range(num_samples_to_plot):
            plot_saliency_map_sample(
                saliency_maps[i],
                channel_names=channel_names,
                title=f"Saliency Map - Sample {i+1} ({self.target_type.capitalize()})",
                save_path=os.path.join(output_dir, f"saliency_sample_{i+1}_{method}.png")
            )
        
        # 5. Save channel ranking (overall)
        save_channel_ranking(
            channel_importance,
            channel_names=channel_names,
            save_path=os.path.join(output_dir, f"channel_ranking_{method}.csv"),
            top_k=None  # Save all channels
        )
        
        # 6. Per-task-type visualizations (if requested and available)
        if by_task_type and 'task_types' in results:
            task_types = results['task_types']
            unique_task_types = np.unique(task_types)
            
            print(f"\nGenerating per-task-type visualizations...")
            
            for task_type in unique_task_types:
                task_mask = task_types == task_type
                task_channel_importance = channel_importance[task_mask]
                task_labels = labels[task_mask]
                
                # Create subdirectory for this task type
                task_output_dir = os.path.join(output_dir, task_type)
                os.makedirs(task_output_dir, exist_ok=True)
                
                # Average importance for this task type
                plot_average_channel_importance(
                    task_channel_importance,
                    channel_names=channel_names,
                    title=f"Average Channel Importance ({self.target_type.capitalize()}) - {task_type.capitalize()} Tasks",
                    save_path=os.path.join(task_output_dir, f"average_importance_{method}.png"),
                    top_k=top_k
                )
                
                # Distribution for this task type
                plot_channel_importance_distribution(
                    task_channel_importance,
                    channel_names=channel_names,
                    title=f"Channel Importance Distribution ({self.target_type.capitalize()}) - {task_type.capitalize()} Tasks",
                    save_path=os.path.join(task_output_dir, f"importance_distribution_{method}.png"),
                    top_k=top_k
                )
                
                # Comparison by class for this task type
                if self.target_type in ['gender', 'age', 'combined']:
                    plot_comparison_by_class(
                        task_channel_importance,
                        task_labels,
                        class_names=class_names,
                        title=f"Channel Importance by Class ({self.target_type.capitalize()}) - {task_type.capitalize()} Tasks",
                        save_path=os.path.join(task_output_dir, f"importance_by_class_{method}.png"),
                        top_k=top_k
                    )
                
                # Save ranking for this task type
                save_channel_ranking(
                    task_channel_importance,
                    channel_names=channel_names,
                    save_path=os.path.join(task_output_dir, f"channel_ranking_{method}.csv"),
                    top_k=None
                )
                
                print(f"  ✅ {task_type.capitalize()} task visualizations saved to {task_output_dir}")
        
        print(f"✅ All visualizations saved to {output_dir}")
    
    def get_top_channels(self,
                        method: str = 'integrated_gradients',
                        top_k: int = 10,
                        task_type: Optional[str] = None) -> List[int]:
        """
        Get indices of top K most important channels.
        
        Args:
            method: Method name
            top_k: Number of top channels
            task_type: Filter by task type ('active', 'passive', or None for all)
        
        Returns:
            List of channel indices (0-indexed)
        """
        if method not in self.results:
            raise ValueError(f"No results found for method: {method}")
        
        results = self.results[method]
        
        # Filter by task type if specified
        if task_type is not None and 'task_types' in results:
            task_mask = results['task_types'] == task_type
            if not task_mask.any():
                raise ValueError(f"No samples found for task type: {task_type}")
            
            # Create filtered results for analysis
            filtered_results = {
                'channel_importance': results['channel_importance'][task_mask],
                'labels': results['labels'][task_mask],
                'predictions': results['predictions'][task_mask]
            }
            analysis = self.analyze_channel_importance(results=filtered_results, method=method)
        else:
            analysis = self.analyze_channel_importance(method=method)
        
        ranked_indices = analysis['ranked_indices']
        
        return ranked_indices[:top_k].tolist()
    
    def analyze_by_task_type(self,
                            method: str = 'integrated_gradients') -> Dict[str, Dict[str, np.ndarray]]:
        """
        Analyze channel importance separately for each task type.
        
        Args:
            method: Method name to analyze
        
        Returns:
            Dictionary mapping task types to analysis results
            e.g., {'active': {...}, 'passive': {...}}
        """
        if method not in self.results:
            raise ValueError(f"No results found for method: {method}")
        
        results = self.results[method]
        
        if 'task_types' not in results:
            raise ValueError("Task types not available in results. "
                           "Ensure data loader includes 'task_types' in batch.")
        
        task_types = results['task_types']
        unique_task_types = np.unique(task_types)
        
        task_analyses = {}
        
        for task_type in unique_task_types:
            task_mask = task_types == task_type
            
            # Create filtered results
            filtered_results = {
                'channel_importance': results['channel_importance'][task_mask],
                'labels': results['labels'][task_mask],
                'predictions': results['predictions'][task_mask]
            }
            
            # Analyze this task type
            analysis = self.analyze_channel_importance(results=filtered_results, method=method)
            task_analyses[task_type] = analysis
            
            print(f"\n{task_type.upper()} Task Analysis:")
            print("-" * 60)
            print(f"Number of samples: {len(filtered_results['labels'])}")
            print(f"Top 10 Most Important Channels:")
            ranked_indices = analysis['ranked_indices']
            avg_importance = analysis['average_importance']
            for rank, idx in enumerate(ranked_indices[:10], 1):
                print(f"  Rank {rank:2d}: Channel {idx+1:2d} "
                      f"(importance: {avg_importance[idx]:.6f})")
        
        return task_analyses
    
    def save_results(self, output_dir: str, method: str = 'integrated_gradients'):
        """
        Save saliency results to disk.
        
        Args:
            output_dir: Directory to save results
            method: Method name to save
        """
        if method not in self.results:
            raise ValueError(f"No results found for method: {method}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = self.results[method]
        
        # Save as numpy arrays
        save_dict = {
            'saliency_maps': results['saliency_maps'],
            'channel_importance': results['channel_importance'],
            'labels': results['labels'],
            'predictions': results['predictions']
        }
        
        # Include task_types if available
        if 'task_types' in results:
            save_dict['task_types'] = results['task_types']
        
        np.savez_compressed(
            os.path.join(output_dir, f"saliency_results_{method}.npz"),
            **save_dict
        )
        
        print(f"✅ Results saved to {output_dir}")


