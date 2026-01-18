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
        
        print(f"\n{'='*60}")
        print(f"Computing saliency using {method}...")
        print(f"Target type: {self.target_type}")
        print(f"{'='*60}")
        
        # Display device information
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            print(f"Device: {self.device} ({gpu_name}, {gpu_memory:.1f} GB)")
            # Check if model is using DataParallel
            if isinstance(self.model, torch.nn.DataParallel):
                num_gpus = torch.cuda.device_count()
                print(f"Multi-GPU: Model wrapped with DataParallel (using {num_gpus} GPUs)")
            elif hasattr(self.model, 'module'):
                # Model might be wrapped in another way
                num_gpus = torch.cuda.device_count()
                print(f"Multi-GPU: Model appears to be using multiple GPUs ({num_gpus} GPUs)")
            else:
                print(f"Single GPU: Using one GPU for computation")
        else:
            print(f"Device: {self.device} (CPU)")
        
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
        
        # Print participant distribution if available
        if 'participant_ids' in results:
            participant_ids = results['participant_ids']
            unique_participants, participant_counts = np.unique(participant_ids, return_counts=True)
            print(f"Participant distribution: {len(unique_participants)} unique participants")
            print(f"  Total samples: {len(participant_ids)}")
            print(f"  Samples per participant: min={participant_counts.min()}, "
                  f"max={participant_counts.max()}, mean={participant_counts.mean():.1f}")
        
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
        saliency_maps = results['saliency_maps']  # Get saliency maps for aggregated visualization
        participant_ids = results.get('participant_ids', None)  # Get participant IDs if available
        
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
        
        # 4. Aggregated saliency map (across all participants)
        # BEST PRACTICE: Average first, then plot (following standard practice in literature)
        saliency_maps = results['saliency_maps']
        
        # Generate aggregated saliency map across all participants
        if 'participant_ids' in results:
            aggregated = self.aggregate_across_participants(method=method, aggregation='mean')
            aggregated_saliency_map = aggregated['saliency_maps']
            aggregated_variance = aggregated.get('saliency_maps_variance', None)
            
            # Plot mean aggregated saliency map
            # BEST PRACTICE: Use absolute values for aggregated maps (standard in literature)
            # This shows importance magnitude regardless of sign, which is clearer for population-level patterns
            norm_info = " (L2-normalized per sample before averaging)" if aggregated.get('normalized_before_aggregation', False) else ""
            plot_saliency_map_sample(
                aggregated_saliency_map,
                channel_names=channel_names,
                title=f"Aggregated Saliency Map - All Participants ({self.target_type.capitalize()}){norm_info}\n{aggregated['num_participants']} participants, {aggregated['num_samples']} samples, method={aggregated['aggregation_method']}\n(Absolute values - showing importance magnitude)",
                save_path=os.path.join(output_dir, f"saliency_aggregated_all_participants_{method}.png"),
                use_absolute=True  # Standard practice: absolute values for aggregated maps
            )
            print(f"  ✅ Saved aggregated saliency map (all {aggregated['num_participants']} participants)")
            
            # Also plot variance/consistency map to show how consistent patterns are
            if aggregated_variance is not None:
                plot_saliency_map_sample(
                    aggregated_variance,
                    channel_names=channel_names,
                    title=f"Saliency Map Consistency (Std Dev) - All Participants ({self.target_type.capitalize()})\n{aggregated['num_participants']} participants, {aggregated['num_samples']} samples",
                    save_path=os.path.join(output_dir, f"saliency_consistency_all_participants_{method}.png"),
                    cmap='viridis'  # Use different colormap for variance (no negative values)
                )
                print(f"  ✅ Saved consistency map (shows variance across participants)")
        
        # 5. Sample saliency maps (first few samples for reference)
        # Note: These are individual samples, not aggregated
        # For individual samples, we can show absolute values (standard) or signed values (shows directionality)
        num_samples_to_plot = min(3, len(saliency_maps))  # Reduced to 3 since we have aggregated map
        for i in range(num_samples_to_plot):
            # Use absolute values for consistency with aggregated maps (standard practice)
            # If you want to see signed values (directionality), set use_absolute=False
            plot_saliency_map_sample(
                saliency_maps[i],
                channel_names=channel_names,
                title=f"Saliency Map - Individual Sample {i+1} ({self.target_type.capitalize()})\n(Absolute values - showing importance magnitude)",
                save_path=os.path.join(output_dir, f"saliency_sample_{i+1}_{method}.png"),
                use_absolute=True  # Standard: absolute values show importance magnitude
            )
        
        # 6. Save channel ranking (overall)
        save_channel_ranking(
            channel_importance,
            channel_names=channel_names,
            save_path=os.path.join(output_dir, f"channel_ranking_{method}.csv"),
            top_k=None  # Save all channels
        )
        
        # 7. Per-task-type visualizations (if requested and available)
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
                
                # Aggregated saliency map for this task type (across all participants)
                # BEST PRACTICE: Normalize before aggregation (same as overall aggregation)
                if participant_ids is not None:
                    task_saliency_maps = saliency_maps[task_mask]
                    task_participant_ids = participant_ids[task_mask]
                    task_unique_participants = len(np.unique(task_participant_ids))
                    
                    # Normalize each sample's saliency map before aggregation (best practice)
                    task_saliency_maps_normalized = task_saliency_maps.copy()
                    for i in range(len(task_saliency_maps)):
                        sample_map = task_saliency_maps[i]
                        l2_norm = np.linalg.norm(sample_map)
                        if l2_norm > 0:
                            task_saliency_maps_normalized[i] = sample_map / l2_norm
                    
                    # Aggregate across all samples for this task type
                    task_aggregated_saliency = task_saliency_maps_normalized.mean(axis=0)
                    task_aggregated_variance = task_saliency_maps_normalized.std(axis=0)
                    
                    plot_saliency_map_sample(
                        task_aggregated_saliency,
                        channel_names=channel_names,
                        title=f"Aggregated Saliency Map - {task_type.capitalize()} Tasks ({self.target_type.capitalize()}) (L2-normalized)\n{task_unique_participants} participants, {len(task_saliency_maps)} samples\n(Absolute values - showing importance magnitude)",
                        save_path=os.path.join(task_output_dir, f"saliency_aggregated_all_participants_{method}.png"),
                        use_absolute=True  # Standard practice: absolute values for aggregated maps
                    )
                    
                    # Also plot consistency map for this task type
                    plot_saliency_map_sample(
                        task_aggregated_variance,
                        channel_names=channel_names,
                        title=f"Saliency Consistency (Std Dev) - {task_type.capitalize()} Tasks ({self.target_type.capitalize()})\n{task_unique_participants} participants, {len(task_saliency_maps)} samples",
                        save_path=os.path.join(task_output_dir, f"saliency_consistency_all_participants_{method}.png"),
                        cmap='viridis'
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
    
    def aggregate_across_participants(self,
                                     method: str = 'integrated_gradients',
                                     aggregation: str = 'mean') -> Dict[str, np.ndarray]:
        """
        Aggregate channel importance across all participants to get a single
        saliency map representing the average importance across all participants.
        
        This is useful for understanding general patterns that apply across
        the entire population, rather than participant-specific patterns.
        
        Args:
            method: Method name to aggregate
            aggregation: Aggregation method ('mean', 'median', or 'std')
        
        Returns:
            Dictionary with aggregated results:
            - 'channel_importance': (num_channels,) aggregated channel importance
            - 'saliency_maps': (num_channels, timepoints) aggregated saliency maps
            - 'num_participants': Number of unique participants
            - 'num_samples': Total number of samples
        """
        if method not in self.results:
            raise ValueError(f"No results found for method: {method}")
        
        results = self.results[method]
        
        if 'participant_ids' not in results:
            raise ValueError(
                "Participant IDs not available in results. "
                "This method requires participant tracking. "
                "Ensure compute_saliency() was called with a dataset that includes participant_ids."
            )
        
        channel_importance = results['channel_importance']
        saliency_maps = results['saliency_maps']
        participant_ids = results['participant_ids']
        
        # BEST PRACTICE: Normalize each sample's saliency map before aggregation
        # This prevents samples with high-magnitude gradients from dominating the average
        # Normalization options:
        # - 'none': No normalization (raw aggregation) - may be dominated by high-magnitude samples
        # - 'l2': L2 normalize each sample (preserves relative patterns)
        # - 'max_abs': Normalize by max absolute value (preserves sign, scales to [-1, 1])
        # - 'zscore': Z-score normalization (mean=0, std=1 per sample)
        normalize_before_aggregation = True  # Best practice: normalize before aggregating
        
        if normalize_before_aggregation:
            # Normalize each saliency map by its L2 norm to preserve relative patterns
            # This ensures each sample contributes equally regardless of gradient magnitude
            saliency_maps_normalized = saliency_maps.copy()
            for i in range(len(saliency_maps)):
                sample_map = saliency_maps[i]
                l2_norm = np.linalg.norm(sample_map)
                if l2_norm > 0:
                    saliency_maps_normalized[i] = sample_map / l2_norm
                # If norm is 0, keep as is (all zeros)
        else:
            saliency_maps_normalized = saliency_maps
        
        # Aggregate across all samples (all participants)
        # Note: We aggregate the normalized maps to ensure equal contribution from each sample
        if aggregation == 'mean':
            aggregated_channel_importance = channel_importance.mean(axis=0)
            aggregated_saliency_maps = saliency_maps_normalized.mean(axis=0)
        elif aggregation == 'median':
            aggregated_channel_importance = np.median(channel_importance, axis=0)
            aggregated_saliency_maps = np.median(saliency_maps_normalized, axis=0)
        elif aggregation == 'std':
            aggregated_channel_importance = channel_importance.std(axis=0)
            aggregated_saliency_maps = saliency_maps_normalized.std(axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}. "
                           "Must be 'mean', 'median', or 'std'")
        
        # Also compute variance/consistency map to show how consistent patterns are
        saliency_maps_variance = saliency_maps_normalized.std(axis=0)
        
        unique_participants = np.unique(participant_ids)
        
        aggregated_results = {
            'channel_importance': aggregated_channel_importance,
            'saliency_maps': aggregated_saliency_maps,
            'saliency_maps_variance': saliency_maps_variance,  # Consistency map
            'num_participants': len(unique_participants),
            'num_samples': len(participant_ids),
            'aggregation_method': aggregation,
            'normalized_before_aggregation': normalize_before_aggregation
        }
        
        print(f"\nAggregated saliency map across all participants:")
        print(f"  Aggregation method: {aggregation}")
        print(f"  Normalization: {'L2 normalized per sample before aggregation' if normalize_before_aggregation else 'No normalization (raw aggregation)'}")
        print(f"  Number of participants: {len(unique_participants)}")
        print(f"  Total samples: {len(participant_ids)}")
        print(f"  Note: Each sample's saliency map was normalized before averaging to ensure equal contribution")
        
        return aggregated_results
    
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
        
        # Include participant_ids if available
        if 'participant_ids' in results:
            save_dict['participant_ids'] = results['participant_ids']
        
        np.savez_compressed(
            os.path.join(output_dir, f"saliency_results_{method}.npz"),
            **save_dict
        )
        
        print(f"✅ Results saved to {output_dir}")


