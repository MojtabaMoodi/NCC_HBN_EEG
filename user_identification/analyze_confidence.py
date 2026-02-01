#!/usr/bin/env python3
"""
Analyze confidence scores for user identification.

This script:
1. Loads a trained model checkpoint
2. Runs inference on test/validation/unknown data
3. Computes confidence scores (max softmax probability) for each prediction
4. Separates correct vs incorrect classifications (known splits)
5. Computes mean, min, max confidence per user (known splits)
6. For unknown (held-out) users: reports feature-space OOD distance (min L2 to nearest
   known-user centroid; higher = more out-of-distribution)
7. Creates plots and saves summaries (unknown_confidence_summary.json includes OOD stats)
"""

import argparse
import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from einops import rearrange

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import required modules
from user_identification.labram_model import LaBraMUserIdentificationWrapper
from data_processing.target_transforms import (
    create_user_identification_transform_from_hdf5,
    UserIdentificationTransform
)
from LaBraM.labram_dataset import prepare_labram_dataset, prepare_labram_unknown_dataset
from CNN.models.arcface_loss import ArcFaceLoss
import utils as labram_utils


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model (strip DataParallel wrapper if present)."""
    return model.module if hasattr(model, 'module') else model


def ood_distance_to_known_user_score(ood_distances: np.ndarray) -> np.ndarray:
    """Convert OOD distance to known-user score: 1/(1+d). Low = unknown, high = known."""
    return (1.0 / (1.0 + np.asarray(ood_distances, dtype=np.float64))).astype(np.float64)


def _format_histogram_stats(arr: np.ndarray) -> str:
    """Format mean/median/min/max/std for histogram text box."""
    return (
        f'Mean: {np.mean(arr):.4f}\n'
        f'Median: {np.median(arr):.4f}\n'
        f'Min: {np.min(arr):.4f}\n'
        f'Max: {np.max(arr):.4f}\n'
        f'Std: {np.std(arr):.4f}'
    )


def load_model(checkpoint_path: str, hdf5_dir: str, device: torch.device) -> tuple:
    """
    Load trained LaBraM model and ArcFace criterion from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        hdf5_dir: Directory containing HDF5 files (to get num_classes)
        device: Device to load model on
        
    Returns:
        Tuple of (model, arcface_criterion, use_arcface)
        - model: Loaded model in eval mode
        - arcface_criterion: ArcFace criterion if available, None otherwise
        - use_arcface: Boolean indicating if ArcFace should be used
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Get number of classes from transform
    transform, num_classes = create_user_identification_transform_from_hdf5(hdf5_dir)
    print(f"Number of classes (users): {num_classes}")
    
    # Create model
    model = LaBraMUserIdentificationWrapper(
        num_channels=60,
        num_classes=num_classes,
        dropout_rate=0.1,
        model_name="labram_base_patch200_200"
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    state_dict = new_state_dict
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    # Check if checkpoint has ArcFace criterion
    arcface_criterion = None
    use_arcface = False
    if 'arcface_state_dict' in checkpoint:
        print("Found ArcFace criterion in checkpoint, loading it...")
        use_arcface = True
        
        # Get embedding dimension from model
        model_to_use = _unwrap_model(model)
        if hasattr(model_to_use, 'fc2_intermediate'):
            embedding_dim = model_to_use.fc2_intermediate.out_features
        else:
            embedding_dim = model_to_use.embed_dim
        
        # Create ArcFace criterion with default parameters (should match training)
        arcface_criterion = ArcFaceLoss(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            margin=0.5,  # Default from training script
            scale=256.0,  # Default from training script
            easy_margin=False
        )
        arcface_criterion.load_state_dict(checkpoint['arcface_state_dict'])
        arcface_criterion = arcface_criterion.to(device)
        arcface_criterion.eval()
        print(f"✅ ArcFace criterion loaded (embedding_dim={embedding_dim})")
    else:
        print("⚠️  No ArcFace criterion found in checkpoint, using standard forward()")
    
    print("Model loaded successfully!")
    return model, arcface_criterion, use_arcface


def run_inference(model: nn.Module, data_loader: DataLoader, device: torch.device,
                 arcface_criterion: nn.Module = None, use_arcface: bool = False,
                 return_embeddings: bool = False) -> tuple:
    """
    Run inference on data and collect predictions, confidence scores, and labels.
    
    Args:
        model: Model in eval mode
        data_loader: DataLoader for test/validation data
        device: Device to run inference on
        arcface_criterion: ArcFace criterion if model was trained with ArcFace (optional)
        use_arcface: Whether to use ArcFace for inference (default: False)
        return_embeddings: If True and use_arcface, also return L2-normalized embeddings (default: False)
        
    Returns:
        Tuple of (predictions, confidence_scores, true_labels, user_ids) or, when return_embeddings=True,
        (predictions, confidence_scores, true_labels, user_ids, embeddings).
    """
    model.eval()
    all_predictions = []
    all_confidence_scores = []
    all_true_labels = []
    all_user_ids = []
    all_embeddings = [] if return_embeddings else None
    
    # Get input_chans and patch_size for LaBraM
    input_chans = None
    patch_size = 200  # Default patch size
    model_to_use = _unwrap_model(model)
    if hasattr(model_to_use, 'input_chans'):
        input_chans = model_to_use.input_chans
    if hasattr(model_to_use, 'patch_size'):
        patch_size = model_to_use.patch_size
        print(f"Using patch_size={patch_size} from model")
    else:
        print(f"Warning: Model doesn't have patch_size attribute, using default={patch_size}")
    
    print("Running inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Handle both tuple and dict formats
            # LaBraM datasets return (X, Y) tuples
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                eeg_data, labels = batch
                user_ids_batch = [None] * len(labels)
            elif isinstance(batch, dict):
                eeg_data = batch['eeg_data']
                labels = batch.get('user_identification', batch.get('label'))
                # Try to get participant IDs if available
                user_ids_batch = batch.get('participant_id', [None] * len(labels))
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")
            
            eeg_data = eeg_data.float().to(device, non_blocking=True) / 100
            
            # Reshape to LaBraM format: [B, N, T] -> [B, N, num_patches, patch_size]
            # Match the model's internal reshape logic exactly
            batch_size, num_channels, sequence_length = eeg_data.shape
            num_patches = sequence_length // patch_size
            
            if sequence_length % patch_size != 0:
                # If sequence length is not divisible by patch_size, pad or truncate
                # (matching logic from LaBraMUserIdentificationWrapper._reshape_input_for_labram)
                target_length = num_patches * patch_size
                if sequence_length < target_length:
                    # Pad with zeros (shouldn't normally happen, but handle edge case)
                    padding = target_length - sequence_length
                    eeg_data = torch.nn.functional.pad(eeg_data, (0, padding), mode='constant', value=0)
                else:
                    # Truncate to fit exact number of patches (normal case)
                    eeg_data = eeg_data[:, :, :target_length]
            
            # Reshape using rearrange (now guaranteed to be divisible by patch_size)
            eeg_data = rearrange(eeg_data, 'B N (A T) -> B N A T', T=patch_size)
            labels = labels.to(device, non_blocking=True).long()
            
            # Get model output
            # For ArcFace models, extract features and compute logits via ArcFace
            # For standard models, use direct forward
            if use_arcface and arcface_criterion is not None:
                # Extract features and compute logits via ArcFace
                model_to_use = _unwrap_model(model)
                if hasattr(model_to_use, 'extract_features'):
                    features = model_to_use.extract_features(eeg_data, input_chans=input_chans)
                else:
                    raise AttributeError("Model must have extract_features for ArcFace inference")
                if return_embeddings:
                    all_embeddings.append(features.cpu().numpy())
                # Compute logits from ArcFace
                outputs = arcface_criterion.compute_logits(features)
                outputs = outputs.float() if outputs.dtype == torch.float16 else outputs
            else:
                # Standard forward pass
                outputs = model(eeg_data, input_chans=input_chans)
            
            # Debug: Check output shape and values for first batch
            if batch_idx == 0:
                print(f"  Debug - Model output shape: {outputs.shape}")
                print(f"  Debug - Model output range: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}")
                print(f"  Debug - Model output std: {outputs.std().item():.4f}")
                print(f"  Debug - Model output sample (first sample, first 10 classes): {outputs[0, :10].cpu().numpy()}")
                print(f"  Debug - Model output sample (first sample, last 10 classes): {outputs[0, -10:].cpu().numpy()}")
                
                # Check if all outputs are the same (would indicate a problem)
                if outputs.std().item() < 1e-6:
                    print(f"  ⚠️  WARNING: Model outputs have very low variance (std={outputs.std().item():.6f})")
                    print(f"     This suggests the model might not be working correctly!")
            
            # Convert to probabilities
            probabilities = torch.softmax(outputs, dim=1)
            
            # Debug: Check probabilities for first batch
            if batch_idx == 0:
                print(f"  Debug - Probabilities shape: {probabilities.shape}")
                print(f"  Debug - Probabilities sum per sample (should be ~1.0): {probabilities.sum(dim=1)[:5].cpu().numpy()}")
                max_probs = probabilities.max(dim=1)[0]
                print(f"  Debug - Max probability per sample (first 5): {max_probs[:5].cpu().numpy()}")
                print(f"  Debug - Max probability stats: min={max_probs.min().item():.6f}, max={max_probs.max().item():.6f}, mean={max_probs.mean().item():.6f}")
            
            # Get predictions and confidence scores
            # confidence_scores is the max probability (confidence in the predicted class)
            # predictions is the index of the class with max probability
            confidence_scores, predictions = torch.max(probabilities, dim=1)
            
            # Debug: Check confidence scores for first batch
            if batch_idx == 0:
                print(f"  Debug - Confidence scores (first 10): {confidence_scores[:10].cpu().numpy()}")
                print(f"  Debug - Confidence scores stats: min={confidence_scores.min().item():.6f}, max={confidence_scores.max().item():.6f}, mean={confidence_scores.mean().item():.6f}, std={confidence_scores.std().item():.6f}")
                print(f"  Debug - Predictions (first 10): {predictions[:10].cpu().numpy()}")
                print(f"  Debug - True labels (first 10): {labels[:10].cpu().numpy()}")
                print(f"  Debug - Unique predictions in batch: {len(torch.unique(predictions))} unique classes")
                print(f"  Debug - Unique true labels in batch: {len(torch.unique(labels))} unique classes")
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_confidence_scores.extend(confidence_scores.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_user_ids.extend(user_ids_batch)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
    
    print(f"Inference complete! Processed {len(all_predictions)} samples.")
    
    result = (
        np.array(all_predictions),
        np.array(all_confidence_scores),
        np.array(all_true_labels),
        all_user_ids
    )
    if return_embeddings and all_embeddings:
        result = result + (np.vstack(all_embeddings),)
    return result


def compute_centroids_from_embeddings(embeddings: np.ndarray, true_labels: np.ndarray,
                                      num_classes: int) -> np.ndarray:
    """
    Compute L2-normalized centroid per class from embeddings (feature-space OOD).
    Embeddings are assumed L2-normalized; centroids are normalized for consistent distance.
    """
    centroids = np.zeros((num_classes, embeddings.shape[1]), dtype=np.float32)
    for k in range(num_classes):
        mask = true_labels == k
        if mask.sum() > 0:
            centroids[k] = embeddings[mask].mean(axis=0)
    # L2-normalize so distance is consistent with normalized embeddings
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    centroids = centroids / norms
    return centroids


def compute_ood_distances(embeddings: np.ndarray, centroids: np.ndarray,
                          chunk_size: int = 1024) -> np.ndarray:
    """
    Min L2 distance from each embedding to any centroid (feature-space OOD).
    Higher distance = more out-of-distribution (likely unknown user).
    Uses chunked computation to avoid OOM for large n (e.g. 180k samples).
    """
    n, d = embeddings.shape
    k = centroids.shape[0]
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        emb_chunk = embeddings[start:end]  # (chunk, d)
        # (chunk, 1, d) - (1, k, d) -> (chunk, k, d); norm -> (chunk, k); min -> (chunk,)
        diff = emb_chunk[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        dists_chunk = np.linalg.norm(diff, axis=2).min(axis=1)
        out[start:end] = dists_chunk.astype(np.float32)
    return out


def load_or_compute_centroids(model: nn.Module, train_loader: DataLoader, device: torch.device,
                              arcface_criterion: nn.Module, use_arcface: bool,
                              output_dir: Path, num_classes: int) -> np.ndarray:
    """
    Load cached known-user centroids or compute from train set (one centroid per class).
    Used for feature-space OOD: distance to nearest centroid indicates in- vs out-of-distribution.
    """
    cache_path = output_dir / 'ood_centroids.npz'
    if cache_path.exists():
        data = np.load(cache_path)
        return data['centroids']
    if not use_arcface:
        raise ValueError("Centroids require use_arcface (embeddings from extract_features).")
    result = run_inference(
        model, train_loader, device, arcface_criterion, use_arcface, return_embeddings=True
    )
    _, _, true_labels, _, embeddings = result
    centroids = compute_centroids_from_embeddings(embeddings, true_labels, num_classes)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, centroids=centroids)
    print(f"Saved OOD centroids to {cache_path}")
    return centroids


# Open-set: reject when known_user_score < threshold (predict "unknown" = -1)
OPEN_SET_REJECT_LABEL = -1


def open_set_predict(softmax_pred: np.ndarray, known_user_scores: np.ndarray,
                    threshold: float) -> np.ndarray:
    """
    Open-set predictions: reject (predict unknown) when known_user_score < threshold.
    Returns same shape as softmax_pred; use OPEN_SET_REJECT_LABEL (-1) for rejected.
    """
    open_set_pred = np.where(known_user_scores >= threshold, softmax_pred, OPEN_SET_REJECT_LABEL)
    return open_set_pred.astype(np.int32)


def open_set_accuracy(open_set_pred: np.ndarray, true_labels: np.ndarray,
                     is_unknown: np.ndarray) -> float:
    """
    Open-set accuracy: (correct known + correct unknown rejections) / total.
    true_labels: class index for known, -1 for unknown. is_unknown: True where unknown.
    """
    correct_known = np.logical_and(~is_unknown, open_set_pred == true_labels)
    correct_unknown = np.logical_and(is_unknown, open_set_pred == OPEN_SET_REJECT_LABEL)
    return float(np.sum(correct_known | correct_unknown) / len(true_labels))


def tune_open_set_threshold(softmax_pred: np.ndarray, known_user_scores: np.ndarray,
                            true_labels: np.ndarray, is_unknown: np.ndarray,
                            thresholds: np.ndarray = None) -> tuple:
    """
    Find threshold on known_user_score that maximizes open-set accuracy.
    Returns (best_threshold, best_accuracy, list of (threshold, accuracy)).
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)
    results = []
    best_acc = -1.0
    best_t = float(thresholds[0])
    for t in thresholds:
        pred = open_set_predict(softmax_pred, known_user_scores, float(t))
        acc = open_set_accuracy(pred, true_labels, is_unknown)
        results.append((float(t), acc))
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t, best_acc, results


def compute_user_statistics(predictions: np.ndarray, confidence_scores: np.ndarray, 
                           true_labels: np.ndarray, mapping_file: Path) -> dict:
    """
    Compute mean, min, max confidence for each user, separated by correct/incorrect.
    
    Args:
        predictions: Array of predicted class indices
        confidence_scores: Array of confidence scores
        true_labels: Array of true class labels
        mapping_file: Path to participant_id_to_class_idx.json mapping file
        
    Returns:
        Dictionary with statistics for each user:
        {
            user_id: {
                'correct': {'mean': float, 'min': float, 'max': float, 'count': int},
                'incorrect': {'mean': float, 'min': float, 'max': float, 'count': int}
            }
        }
    """
    # Load mapping: participant_id_to_class_idx.json has participant_id (str) -> class_idx (int)
    with open(mapping_file, 'r') as f:
        participant_id_to_class = json.load(f)
    # Reverse: class_idx -> participant_id. Use int() so lookup works (JSON may load numbers as int/float).
    class_to_participant = {int(cls_idx): pid for pid, cls_idx in participant_id_to_class.items()}

    # Determine correct vs incorrect
    is_correct = np.asarray(predictions == true_labels)
    n_incorrect_expected = int(np.sum(~is_correct))

    # Group by user (by true label's participant_id)
    user_stats = defaultdict(lambda: {
        'correct': {'scores': [], 'count': 0},
        'incorrect': {'scores': [], 'count': 0}
    })

    for pred, conf, true_label, correct in zip(
        predictions, confidence_scores, true_labels, is_correct
    ):
        cls_idx = int(true_label)
        user_id = class_to_participant.get(cls_idx, f"class_{cls_idx}")

        if correct:
            user_stats[user_id]['correct']['scores'].append(conf)
            user_stats[user_id]['correct']['count'] += 1
        else:
            user_stats[user_id]['incorrect']['scores'].append(conf)
            user_stats[user_id]['incorrect']['count'] += 1

    n_incorrect_actual = sum(stats['incorrect']['count'] for stats in user_stats.values())
    if n_incorrect_actual != n_incorrect_expected:
        raise ValueError(
            f"Per-user incorrect count mismatch: sum of user incorrect counts = {n_incorrect_actual}, "
            f"but global incorrect count = {n_incorrect_expected}. "
            "Check mapping file and that predictions/true_labels are aligned."
        )

    # Compute statistics (when count=0 we only store count, not mean/min/max, to avoid null)
    result = {}
    for user_id, stats in user_stats.items():
        result[user_id] = {}
        for category in ['correct', 'incorrect']:
            scores = stats[category]['scores']
            if len(scores) > 0:
                result[user_id][category] = {
                    'mean': np.mean(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'count': len(scores)
                }
            else:
                result[user_id][category] = {'count': 0}
    
    return result


def create_plots(confidence_correct: np.ndarray, confidence_incorrect: np.ndarray,
                 output_dir: Path, suffix: str = ""):
    """
    Create two plots: one for correct classifications, one for incorrect.
    
    Args:
        confidence_correct: Array of confidence scores for correct predictions
        confidence_incorrect: Array of confidence scores for incorrect predictions
        output_dir: Directory to save plots
        suffix: Optional suffix for filename (e.g. '_test')
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1 = axes[0]
    ax1.hist(confidence_correct, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax1.axvline(np.mean(confidence_correct), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(confidence_correct):.4f}')
    ax1.axvline(np.median(confidence_correct), color='blue', linestyle='--', linewidth=2,
               label=f'Median: {np.median(confidence_correct):.4f}')
    ax1.set_xlabel('Confidence Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Distribution of Confidence Scores\nCorrect Classifications (n={len(confidence_correct)})', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax1.text(0.02, 0.98, _format_histogram_stats(confidence_correct), transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2 = axes[1]
    ax2.hist(confidence_incorrect, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax2.axvline(np.mean(confidence_incorrect), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(confidence_incorrect):.4f}')
    ax2.axvline(np.median(confidence_incorrect), color='blue', linestyle='--', linewidth=2,
               label=f'Median: {np.median(confidence_incorrect):.4f}')
    ax2.set_xlabel('Confidence Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Distribution of Confidence Scores\nIncorrect Classifications (n={len(confidence_incorrect)})',
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax2.text(0.02, 0.98, _format_histogram_stats(confidence_incorrect), transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filename = f'confidence_distributions{suffix}.png' if suffix else 'confidence_distributions.png'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def create_plots_unknown_only(confidence_scores: np.ndarray, output_dir: Path, suffix: str = "",
                              ood_distances: np.ndarray = None) -> None:
    """
    Create plots for unknown (held-out) users: softmax confidence and, if provided, OOD-based score.
    
    Args:
        confidence_scores: Array of softmax confidence (max prob over known classes)
        output_dir: Directory to save plots
        suffix: Optional suffix for filename (e.g. '_unknown')
        ood_distances: Optional OOD distances; if provided, plot known_user_score (1/(1+OOD)) — low = unknown
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    # Plot 1: Softmax confidence (closed-world; often high for unknown)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(confidence_scores, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(np.mean(confidence_scores), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(confidence_scores):.4f}')
    ax.axvline(np.median(confidence_scores), color='blue', linestyle='--', linewidth=2,
               label=f'Median: {np.median(confidence_scores):.4f}')
    ax.set_xlabel('Softmax Confidence (max over known classes)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Unknown Users — Softmax Confidence (closed-world; often high)\nn={len(confidence_scores)}',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, _format_histogram_stats(confidence_scores), transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    filename = f'confidence_distribution_unknown{suffix}.png' if suffix else 'confidence_distribution_unknown.png'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()

    # Plot 2: Known-user score (1/(1+OOD distance)); low = more OOD / unknown
    if ood_distances is not None:
        known_user_scores = ood_distance_to_known_user_score(ood_distances)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(known_user_scores, bins=50, alpha=0.7, color='teal', edgecolor='black')
        ax.axvline(np.mean(known_user_scores), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(known_user_scores):.4f}')
        ax.axvline(np.median(known_user_scores), color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(known_user_scores):.4f}')
        ax.set_xlabel('Known-User Score (1/(1+OOD distance)); low = unknown', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Unknown Users — Known-User Score (low = more OOD)\nn={len(known_user_scores)}',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, _format_histogram_stats(known_user_scores), transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        filename = f'known_user_score_unknown{suffix}.png' if suffix else 'known_user_score_unknown.png'
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        plt.close()


def save_unknown_summary(confidence_scores: np.ndarray, output_dir: Path,
                         ood_distances: np.ndarray = None) -> None:
    """
    Save aggregate confidence summary for unknown (held-out) users to JSON.
    Unknown users have no identity labels, so we only save overall stats, not per-user.
    Optionally include OOD distance and known-user score (low = more OOD / unknown).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'n_samples': int(len(confidence_scores)),
        'note': 'For unknown vs known, use known_user_score (low = unknown). Softmax confidence is closed-world and is often high for unknown users.',
    }
    if ood_distances is not None:
        known_user_scores = ood_distance_to_known_user_score(ood_distances)
        summary['known_user_score'] = {
            'mean': float(np.mean(known_user_scores)),
            'min': float(np.min(known_user_scores)),
            'max': float(np.max(known_user_scores)),
            'std': float(np.std(known_user_scores)),
            'note': '1/(1+OOD distance). Low = unknown; use this to flag unknown users.',
        }
        summary['ood_distance_to_nearest_known'] = {
            'mean': float(np.mean(ood_distances)),
            'min': float(np.min(ood_distances)),
            'max': float(np.max(ood_distances)),
            'std': float(np.std(ood_distances)),
            'note': 'Higher = more out-of-distribution (unknown).',
        }
    summary['softmax_confidence'] = {
        'mean': float(np.mean(confidence_scores)),
        'min': float(np.min(confidence_scores)),
        'max': float(np.max(confidence_scores)),
        'std': float(np.std(confidence_scores)),
        'note': 'Closed-world (max over known classes); often high for unknown users.',
    }
    output_path = output_dir / 'unknown_confidence_summary.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {output_path}")


def save_statistics(user_stats: dict, output_dir: Path):
    """
    Save user statistics to a JSON file.
    
    Args:
        user_stats: Dictionary with statistics for each user
        output_dir: Directory to save statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
    
    stats_native = convert_to_native(user_stats)
    
    output_path = output_dir / 'user_confidence_statistics.json'
    with open(output_path, 'w') as f:
        json.dump(stats_native, f, indent=2)
    print(f"Statistics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze confidence scores for user identification',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint file')
    parser.add_argument('--hdf5_dir', type=str,
                       default='/home/mojtabam/scratch/processed_eeg_data_user_identification',
                       help='Directory containing HDF5 files (default: /home/mojtabam/scratch/processed_eeg_data_user_identification)')
    parser.add_argument('--segment_length', type=str, default='4s',
                       choices=['1s', '2s', '4s'],
                       help='EEG segment length')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test', 'unknown', 'both'],
                       help='Data split to analyze (unknown=held-out users; both=test and unknown)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--output_dir', type=str, default='./confidence_analysis',
                       help='Directory to save results (plots, user_confidence_statistics.json); for split=both, test/ and unknown/ subdirs are created')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Validate hdf5_dir exists
    hdf5_dir_path = Path(args.hdf5_dir)
    if not hdf5_dir_path.exists():
        raise FileNotFoundError(
            f"HDF5 directory not found: {args.hdf5_dir}\n"
            f"Please provide a valid path to the directory containing HDF5 files.\n"
            f"Expected location: /home/mojtabam/scratch/processed_eeg_data_user_identification"
        )
    if not hdf5_dir_path.is_dir():
        raise ValueError(f"HDF5 path is not a directory: {args.hdf5_dir}")
    
    # Check for mapping file
    mapping_file = hdf5_dir_path / 'participant_id_to_class_idx.json'
    if not mapping_file.exists():
        raise FileNotFoundError(
            f"Mapping file not found: {mapping_file}\n"
            f"Please ensure you have run preprocessing with process_all_participants_user_identification() first.\n"
            f"The mapping file should be at: {mapping_file}"
        )
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model, arcface_criterion, use_arcface = load_model(str(checkpoint_path), args.hdf5_dir, device)
    
    def run_analysis_on_split(split_name: str, dataset, output_suffix: str = "", centroids=None,
                              return_results_for_open_set: bool = False):
        """Run inference and report/plot for one split. If centroids and return_results_for_open_set, return (pred, conf, true, emb)."""
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=(device.type == 'cuda')
        )
        need_embeddings = (centroids is not None and use_arcface and
                          (return_results_for_open_set or split_name == "unknown"))
        result = run_inference(
            model, data_loader, device, arcface_criterion, use_arcface,
            return_embeddings=need_embeddings
        )
        predictions, confidence_scores, true_labels, _ = result[:4]
        embeddings = result[4] if len(result) == 5 else None
        ood_distances = compute_ood_distances(embeddings, centroids) if (embeddings is not None and centroids is not None) else None
        is_unknown_split = split_name == "unknown"
        
        if is_unknown_split:
            # Unknown split: lead with OOD-based score (low = unknown); softmax confidence is closed-world and often high
            print(f"\n{'='*60}")
            print(f"Summary Statistics — Unknown Users (held-out)")
            print(f"{'='*60}")
            print(f"Total samples: {len(predictions)}")
            if ood_distances is not None:
                known_user_scores = ood_distance_to_known_user_score(ood_distances)
                print(f"  → Use this for unknown vs known: Known-user score (low = more unknown)")
                print(f"Known-user score (1/(1+OOD distance); low = unknown, high = known):")
                print(f"  Mean: {np.mean(known_user_scores):.4f}")
                print(f"  Min: {np.min(known_user_scores):.4f}")
                print(f"  Max: {np.max(known_user_scores):.4f}")
                print(f"  Std: {np.std(known_user_scores):.4f}")
                print(f"OOD distance (min L2 to nearest known-user centroid; higher = more OOD):")
                print(f"  Mean: {np.mean(ood_distances):.4f}")
                print(f"  Min: {np.min(ood_distances):.4f}")
                print(f"  Max: {np.max(ood_distances):.4f}")
                print(f"  Std: {np.std(ood_distances):.4f}")
            print(f"Softmax confidence (closed-world; often high for unknown — not a bug):")
            print(f"  Mean: {np.mean(confidence_scores):.4f}")
            print(f"  Min: {np.min(confidence_scores):.4f}")
            print(f"  Max: {np.max(confidence_scores):.4f}")
            print(f"  Std: {np.std(confidence_scores):.4f}")
            out_dir = output_dir / "unknown" if output_suffix else output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            save_unknown_summary(confidence_scores, out_dir, ood_distances=ood_distances)
            create_plots_unknown_only(confidence_scores, out_dir, suffix="", ood_distances=ood_distances)
            if return_results_for_open_set and embeddings is not None:
                return (predictions, confidence_scores, true_labels, embeddings)
            return
        
        # Known split: correct vs incorrect
        is_correct = predictions == true_labels
        confidence_correct = confidence_scores[is_correct]
        confidence_incorrect = confidence_scores[~is_correct]
        
        print(f"\n{'='*60}")
        print(f"Summary Statistics — {split_name.upper()}")
        print(f"{'='*60}")
        print(f"Total samples: {len(predictions)}")
        print(f"Correct predictions: {len(confidence_correct)} ({len(confidence_correct)/len(predictions)*100:.2f}%)")
        print(f"Incorrect predictions: {len(confidence_incorrect)} ({len(confidence_incorrect)/len(predictions)*100:.2f}%)")
        print(f"\nCorrect predictions - Confidence:")
        print(f"  Mean: {np.mean(confidence_correct):.4f}")
        print(f"  Min: {np.min(confidence_correct):.4f}")
        print(f"  Max: {np.max(confidence_correct):.4f}")
        print(f"  Std: {np.std(confidence_correct):.4f}")
        print(f"\nIncorrect predictions - Confidence:")
        print(f"  Mean: {np.mean(confidence_incorrect):.4f}")
        print(f"  Min: {np.min(confidence_incorrect):.4f}")
        print(f"  Max: {np.max(confidence_incorrect):.4f}")
        print(f"  Std: {np.std(confidence_incorrect):.4f}")
        
        out_dir = output_dir / split_name if output_suffix else output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        
        mapping_file = Path(args.hdf5_dir) / 'participant_id_to_class_idx.json'
        if mapping_file.exists():
            user_stats = compute_user_statistics(
                predictions, confidence_scores, true_labels, mapping_file
            )
            save_statistics(user_stats, out_dir)
        
        create_plots(confidence_correct, confidence_incorrect, out_dir, suffix=output_suffix)
        if return_results_for_open_set and embeddings is not None:
            return (predictions, confidence_scores, true_labels, embeddings)
        return None

    if args.split == 'both':
        # Run on test
        print(f"\nLoading test data...")
        transform, num_classes = create_user_identification_transform_from_hdf5(args.hdf5_dir)
        train_dataset, _, test_dataset = prepare_labram_dataset(
            dataset_type="user_identification",
            hdf5_dir=args.hdf5_dir,
            segment_length=args.segment_length,
            task_type="both",
            random_seed=42,
            user_identification_transform=transform,
            sampling_rate=200
        )
        # OOD centroids for unknown split and open-set (only when ArcFace)
        centroids = None
        if use_arcface:
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                shuffle=False, pin_memory=(device.type == 'cuda')
            )
            centroids = load_or_compute_centroids(
                model, train_loader, device, arcface_criterion, use_arcface, output_dir, num_classes
            )
        unknown_dataset = prepare_labram_unknown_dataset(
            hdf5_dir=args.hdf5_dir,
            segment_length=args.segment_length,
            task_type="both",
            sampling_rate=200
        )
        # Run test once (with embeddings for open-set when centroids and unknown exist)
        need_open_set = centroids is not None and unknown_dataset is not None
        test_result = run_analysis_on_split(
            "test", test_dataset, output_suffix="_test", centroids=centroids,
            return_results_for_open_set=need_open_set
        )
        if unknown_dataset is not None:
            print(f"\nLoading unknown (held-out) data...")
            unknown_result = run_analysis_on_split(
                "unknown", unknown_dataset, output_suffix="_unknown", centroids=centroids,
                return_results_for_open_set=(centroids is not None)
            )
            # Open-set: tune threshold on test+unknown and report (reuse test_result from above)
            if centroids is not None and use_arcface and test_result is not None and unknown_result is not None:
                (test_pred, _, test_true, test_emb) = test_result
                (unknown_pred, _, unknown_true, unknown_emb) = unknown_result
                test_known = ood_distance_to_known_user_score(compute_ood_distances(test_emb, centroids))
                unknown_known = ood_distance_to_known_user_score(compute_ood_distances(unknown_emb, centroids))
                all_pred = np.concatenate([test_pred, unknown_pred])
                all_true = np.concatenate([test_true, unknown_true])
                all_known = np.concatenate([test_known, unknown_known])
                is_unknown = (all_true == OPEN_SET_REJECT_LABEL)
                best_t, best_acc, sweep = tune_open_set_threshold(
                    all_pred, all_known, all_true, is_unknown
                )
                print(f"\n{'='*60}")
                print("Open-set (known + unknown) — threshold on known-user score")
                print(f"{'='*60}")
                print(f"Best threshold (known_user_score >= t → predict known): {best_t:.3f}")
                print(f"Best open-set accuracy: {best_acc:.4f}")
                out_path = output_dir / "open_set_summary.json"
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(out_path, 'w') as f:
                    json.dump({
                        'best_threshold_known_user_score': best_t,
                        'best_open_set_accuracy': best_acc,
                        'note': 'Reject (predict unknown) when known_user_score < threshold.',
                        'threshold_sweep': [{'threshold': t, 'accuracy': acc} for t, acc in sweep],
                    }, f, indent=2)
                print(f"Open-set summary saved to: {out_path}")
        else:
            print("\nNo unknown split found (run preprocessing with --consider_unknown to create it).")
    else:
        # Single split
        print(f"\nLoading {args.split} data...")
        centroids = None
        if args.split == 'unknown':
            dataset = prepare_labram_unknown_dataset(
                hdf5_dir=args.hdf5_dir,
                segment_length=args.segment_length,
                task_type="both",
                sampling_rate=200
            )
            if dataset is None:
                raise FileNotFoundError(
                    "Unknown split not found. Run preprocessing with --consider_unknown to create it."
                )
            if use_arcface:
                transform, num_classes = create_user_identification_transform_from_hdf5(args.hdf5_dir)
                train_dataset, _, _ = prepare_labram_dataset(
                    dataset_type="user_identification",
                    hdf5_dir=args.hdf5_dir,
                    segment_length=args.segment_length,
                    task_type="both",
                    random_seed=42,
                    user_identification_transform=transform,
                    sampling_rate=200
                )
                train_loader = DataLoader(
                    train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                    shuffle=False, pin_memory=(device.type == 'cuda')
                )
                centroids = load_or_compute_centroids(
                    model, train_loader, device, arcface_criterion, use_arcface, output_dir, num_classes
                )
            run_analysis_on_split(args.split, dataset, centroids=centroids)
        else:
            transform, _ = create_user_identification_transform_from_hdf5(args.hdf5_dir)
            train_dataset, val_dataset, test_dataset = prepare_labram_dataset(
                dataset_type="user_identification",
                hdf5_dir=args.hdf5_dir,
                segment_length=args.segment_length,
                task_type="both",
                random_seed=42,
                user_identification_transform=transform,
                sampling_rate=200
            )
            dataset = {
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset
            }[args.split]
            run_analysis_on_split(args.split, dataset)
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
