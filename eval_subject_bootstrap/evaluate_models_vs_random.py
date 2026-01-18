#!/usr/bin/env python3
"""
Main evaluation script for subject-level model evaluation with bootstrap.

Evaluates multiple models (CNN, LaBraM, ResNet) against stratified random baseline
using subject-level aggregation and bootstrap uncertainty estimation.
"""

import argparse
import json
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import h5py
import warnings
from datetime import datetime
import subprocess
import sys

# Add current directory to path for imports
import sys
from pathlib import Path
eval_dir = Path(__file__).parent
if str(eval_dir) not in sys.path:
    sys.path.insert(0, str(eval_dir))

# Import local modules
from io_utils import (
    load_train_manifest,
    load_test_manifest,
    validate_subject_labels,
    check_train_test_overlap,
    aggregate_segments_by_subject,
    compute_train_subject_proportions
)
from metrics import compute_metrics, compute_confusion_matrix
from bootstrap import subject_bootstrap, summarize_bootstrap
from model_loader import load_model_from_config, get_model_output


class ManifestDataset(Dataset):
    """
    Dataset that loads segments from HDF5 files based on manifest.
    """
    
    def __init__(self, manifest: pd.DataFrame, hdf5_base_dir: Optional[str] = None):
        """
        Initialize dataset from manifest.
        
        Args:
            manifest: DataFrame with columns: subject_id, label, and segment identifiers
            hdf5_base_dir: Base directory for HDF5 files (if paths in manifest are relative)
        """
        self.manifest = manifest.copy()
        # Convert to Path and resolve to absolute path if provided
        if hdf5_base_dir:
            self.hdf5_base_dir = Path(hdf5_base_dir).resolve()
        else:
            self.hdf5_base_dir = None
        
        # Determine segment identifier format
        if 'segment_path' in manifest.columns:
            self.use_segment_path = True
        elif 'hdf5_file' in manifest.columns and 'sample_id' in manifest.columns:
            self.use_segment_path = False
        else:
            raise ValueError("Manifest must have either 'segment_path' or ('hdf5_file', 'sample_id')")
        
        # Pre-load HDF5 file handles (lazy loading)
        self._hdf5_files = {}
    
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        subject_id = str(row['subject_id'])
        label = int(row['label'])
        
        # Load segment data
        if self.use_segment_path:
            segment_path = row['segment_path']
            if self.hdf5_base_dir:
                segment_path = self.hdf5_base_dir / segment_path
            else:
                segment_path = Path(segment_path)
            
            # Extract HDF5 file and sample ID from path
            # Format: path/to/file.h5:sample_000123 or path/to/file.h5/active/sample_000123
            hdf5_file = str(segment_path).split(':')[0]
            if ':' in str(segment_path):
                sample_id = str(segment_path).split(':')[1]
            else:
                # Assume format: path/to/file.h5/active/sample_000123
                parts = str(segment_path).split('/')
                sample_id = parts[-1]
                hdf5_file = '/'.join(parts[:-2]) if len(parts) > 2 else parts[0]
        else:
            # Get hdf5_file from manifest (should be just filename, e.g., 'eeg_data_test_4s.h5')
            # Path joining will be handled in _load_from_hdf5
            hdf5_file = row['hdf5_file']
            sample_id = row['sample_id']
        
        # Load from HDF5 (path joining happens here)
        eeg_data, participant_id = self._load_from_hdf5(hdf5_file, sample_id)
        
        # Convert to tensor
        eeg_tensor = torch.FloatTensor(eeg_data)
        
        return {
            'eeg_data': eeg_tensor,
            'label': label,
            'subject_id': subject_id,
            'participant_id': participant_id
        }
    
    def _load_from_hdf5(self, hdf5_file: str, sample_id: str) -> Tuple[np.ndarray, str]:
        """
        Load segment from HDF5 file.
        
        Args:
            hdf5_file: Path to HDF5 file
            sample_id: Sample identifier (e.g., 'sample_000123')
            
        Returns:
            Tuple of (eeg_data, participant_id)
        """
        # Handle hdf5_file path (could be absolute or relative)
        hdf5_path = Path(hdf5_file)
        if hdf5_path.is_absolute():
            # Already absolute path, use as-is
            hdf5_file = str(hdf5_path.resolve())
        elif self.hdf5_base_dir:
            # Relative path, join with base directory
            hdf5_file = str((self.hdf5_base_dir / hdf5_file).resolve())
        else:
            # Relative path but no base dir, try to resolve relative to current working directory
            hdf5_file = str(hdf5_path.resolve())
        
        # Open HDF5 file (cache handles)
        if hdf5_file not in self._hdf5_files:
            if not Path(hdf5_file).exists():
                raise FileNotFoundError(f"HDF5 file not found: {hdf5_file}")
            self._hdf5_files[hdf5_file] = h5py.File(hdf5_file, 'r')
        
        f = self._hdf5_files[hdf5_file]
        
        # Extract metadata key from sample_id (e.g., 'sample_000123' -> 'metadata_000123')
        # Use same logic as existing dataset code: replace 'sample_' with 'metadata_'
        if sample_id.startswith('sample_'):
            metadata_key = sample_id.replace('sample_', 'metadata_', 1)  # Only replace first occurrence
        else:
            # Fallback: extract number and construct metadata key
            sample_num = sample_id.split('_')[-1] if '_' in sample_id else sample_id.replace('sample_', '')
            metadata_key = f'metadata_{sample_num}'
        
        # Try to find sample in active or passive groups
        for task_type in ['active', 'passive']:
            if task_type in f:
                task_group = f[task_type]
                if sample_id in task_group:
                    eeg_data = task_group[sample_id][:]
                    
                    # Get participant_id from metadata
                    if metadata_key in task_group:
                        metadata = task_group[metadata_key]
                        participant_id = metadata.attrs.get('participant_id', '')
                        if isinstance(participant_id, bytes):
                            participant_id = participant_id.decode()
                        return eeg_data, str(participant_id)
                    else:
                        # Fallback: try to find any metadata that might match
                        # (sometimes metadata keys don't match exactly)
                        for key in task_group.keys():
                            if key.startswith('metadata_'):
                                metadata = task_group[key]
                                # Check if metadata sample number matches
                                meta_num = key.replace('metadata_', '')
                                if meta_num == sample_num:
                                    participant_id = metadata.attrs.get('participant_id', '')
                                    if isinstance(participant_id, bytes):
                                        participant_id = participant_id.decode()
                                    return eeg_data, str(participant_id)
        
        # If we get here, sample was not found
        available_samples = []
        for task_type in ['active', 'passive']:
            if task_type in f:
                available_samples.extend([k for k in f[task_type].keys() if k.startswith('sample_')])
        
        raise ValueError(
            f"Sample {sample_id} not found in {hdf5_file}\n"
            f"  Available samples (first 10): {available_samples[:10]}\n"
            f"  Total available: {len(available_samples)}"
        )
    
    def __del__(self):
        """Close HDF5 file handles."""
        for f in self._hdf5_files.values():
            try:
                f.close()
            except:
                pass


def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_key: Optional[str] = None,
    model_name: Optional[str] = None,
    model_config_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Run inference on all segments and collect predictions.
    
    Args:
        model: Model in eval mode
        dataloader: DataLoader for test segments
        device: Device to run inference on
        output_key: Optional key if model returns dict
        
    Returns:
        Tuple of:
        - probabilities: Array of shape (N_segments, num_classes)
        - labels: Array of shape (N_segments,)
        - subject_ids: List of subject IDs, length N_segments
    """
    model.eval()
    all_probs = []
    all_labels = []
    all_subject_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            eeg_data = batch['eeg_data'].to(device)
            labels = batch['label'].numpy()
            subject_ids = batch['subject_id']
            
            # Get model output
            # Use model_name if provided, otherwise use model_config_name, otherwise try to detect
            detected_model_name = model_name
            if not detected_model_name and model_config_name:
                detected_model_name = model_config_name.lower()
            elif not detected_model_name:
                # Fallback detection
                if hasattr(model, 'labram_model') or 'labram' in str(type(model)).lower() or 'neuraltransformer' in str(type(model)).lower():
                    detected_model_name = 'labram'
            
            input_chans = None
            if detected_model_name and detected_model_name.lower() == 'labram':
                # Get input_chans for LaBraM
                from model_loader import get_labram_input_chans
                input_chans = get_labram_input_chans()
            
            output = get_model_output(model, eeg_data, output_key, model_name=detected_model_name, input_chans=input_chans)
            
            # Convert to probabilities if logits
            if output.shape[-1] > 1:
                probs = torch.softmax(output, dim=-1).cpu().numpy()
            else:
                # Binary classification
                probs = torch.sigmoid(output).cpu().numpy()
                probs = np.concatenate([1 - probs, probs], axis=-1)
            
            all_probs.append(probs)
            all_labels.append(labels)
            all_subject_ids.extend(subject_ids)
    
    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_labels, axis=0),
        all_subject_ids
    )


def permutation_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    primary_metric: str,
    M: int = 1000,
    seed: int = 123
) -> float:
    """
    Perform subject-level permutation test.
    
    Args:
        y_true: True labels, shape (S,)
        y_pred: Predicted labels, shape (S,)
        primary_metric: Primary metric name
        M: Number of permutations
        seed: Random seed
        
    Returns:
        P-value
    """
    np.random.seed(seed)
    
    # Observed metric
    from metrics import compute_metrics, get_primary_metric
    observed_metrics = compute_metrics(y_true, y_pred)
    observed_value = get_primary_metric(observed_metrics, primary_metric)
    
    # Permutation distribution
    permuted_values = []
    for m in range(M):
        y_perm = np.random.permutation(y_true)
        perm_metrics = compute_metrics(y_perm, y_pred)
        perm_value = get_primary_metric(perm_metrics, primary_metric)
        permuted_values.append(perm_value)
    
    # P-value: P(metric_perm >= metric_obs)
    p_value = (1 + np.sum(np.array(permuted_values) >= observed_value)) / (M + 1)
    
    return float(p_value)


def get_git_commit_hash() -> Optional[str]:
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models vs random baseline with subject-level bootstrap"
    )
    parser.add_argument('--train_manifest', type=str, required=True,
                        help='Path to train manifest (subject-level CSV)')
    parser.add_argument('--test_manifest', type=str, required=True,
                        help='Path to test manifest (segment-level CSV)')
    parser.add_argument('--models_config', type=str, required=True,
                        help='Path to models config YAML/JSON file')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--bootstrap_B', type=int, default=2000,
                        help='Number of bootstrap replicates')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed')
    parser.add_argument('--primary_metric', type=str, default='balanced_accuracy',
                        help='Primary metric name')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: eval_subject_bootstrap/outputs/<timestamp>)')
    parser.add_argument('--save_npz', action='store_true',
                        help='Save intermediate arrays (segment preds, subject preds)')
    parser.add_argument('--permutation_M', type=int, default=0,
                        help='Number of permutations for permutation test (0 to disable)')
    parser.add_argument('--hdf5_base_dir', type=str, default=None,
                        help='Base directory for HDF5 files (if paths in manifest are relative)')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup output directory
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).parent / "outputs" / timestamp
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {out_dir}")
    
    # Load manifests
    print("Loading manifests...")
    train_manifest = load_train_manifest(args.train_manifest)
    test_manifest = load_test_manifest(args.test_manifest)
    
    # Validate subject labels
    print("Validating subject labels...")
    subject_labels = validate_subject_labels(test_manifest)
    
    # Check for train/test overlap
    print("Checking for train/test overlap...")
    overlap = check_train_test_overlap(train_manifest, test_manifest)
    
    # Compute train proportions
    print("Computing train subject proportions...")
    train_proportions, majority_class = compute_train_subject_proportions(train_manifest)
    
    # Load models config
    print("Loading models config...")
    models_config_path = Path(args.models_config)
    if models_config_path.suffix == '.yaml' or models_config_path.suffix == '.yml':
        with open(models_config_path, 'r') as f:
            models_config = yaml.safe_load(f)
    else:
        with open(models_config_path, 'r') as f:
            models_config = json.load(f)
    
    if 'models' not in models_config:
        raise ValueError("models_config must have 'models' key with list of model configs")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = ManifestDataset(test_manifest, hdf5_base_dir=args.hdf5_base_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )
    
    # Evaluate each model
    all_results = {}
    
    for model_cfg in models_config['models']:
        model_name = model_cfg.get('name', 'unknown')
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*60}")
        
        # Load model
        print(f"Loading model from {model_cfg['ckpt_path']}...")
        model = load_model_from_config(model_cfg, device=str(device))
        model.to(device)
        
        # Run inference
        print("Running inference...")
        # Determine model name for preprocessing
        model_name_for_inference = model_cfg.get('name', '').lower()
        segment_probs, segment_labels, segment_subject_ids = run_inference(
            model,
            dataloader,
            device,
            output_key=model_cfg.get('output_key', None),
            model_name=model_name_for_inference,
            model_config_name=model_cfg.get('name', '').lower()
        )
        
        # Aggregate to subject level
        print("Aggregating to subject level...")
        subjects, y_true_subject, y_pred_subject, p_subject_dict = aggregate_segments_by_subject(
            segment_probs,
            segment_labels,
            segment_subject_ids
        )
        
        # Compute point estimates
        print("Computing metrics...")
        point_metrics = compute_metrics(y_true_subject, y_pred_subject)
        conf_matrix = compute_confusion_matrix(y_true_subject, y_pred_subject)
        
        # Random baseline point estimate
        y_pred_random = np.random.choice([0, 1, 2], size=len(y_true_subject), p=train_proportions)
        random_metrics = compute_metrics(y_true_subject, y_pred_random)
        
        # Majority baseline
        y_pred_majority = np.full(len(y_true_subject), majority_class)
        majority_metrics = compute_metrics(y_true_subject, y_pred_majority)
        
        # Bootstrap
        print(f"Running bootstrap (B={args.bootstrap_B})...")
        bootstrap_results = subject_bootstrap(
            y_true_subject,
            y_pred_subject,
            y_pred_majority,
            train_proportions,
            B=args.bootstrap_B,
            seed=args.seed,
            primary_metric=args.primary_metric
        )
        
        # Summarize bootstrap
        print("Summarizing bootstrap results...")
        # Add random baseline to point metrics for delta computation
        point_metrics_with_random = point_metrics.copy()
        for key in random_metrics:
            point_metrics_with_random['random_' + key] = random_metrics[key]
        
        summary = summarize_bootstrap(bootstrap_results, point_metrics_with_random)
        
        # Permutation test (optional)
        perm_p_value = None
        if args.permutation_M > 0:
            print(f"Running permutation test (M={args.permutation_M})...")
            perm_p_value = permutation_test(
                y_true_subject,
                y_pred_subject,
                args.primary_metric,
                M=args.permutation_M,
                seed=args.seed
            )
            print(f"Permutation test p-value: {perm_p_value:.4f}")
        
        # Store results
        all_results[model_name] = {
            'point_metrics': point_metrics,
            'random_metrics': random_metrics,
            'majority_metrics': majority_metrics,
            'confusion_matrix': conf_matrix.tolist(),
            'bootstrap_summary': summary,
            'permutation_p_value': perm_p_value,
            'num_subjects': len(subjects),
            'num_segments': len(segment_subject_ids),
            'segments_per_subject': {
                'min': int(np.min([len([s for s in segment_subject_ids if s == subj]) for subj in subjects])),
                'median': float(np.median([len([s for s in segment_subject_ids if s == subj]) for subj in subjects])),
                'max': int(np.max([len([s for s in segment_subject_ids if s == subj]) for subj in subjects]))
            }
        }
        
        # Save intermediate arrays if requested
        if args.save_npz:
            npz_path = out_dir / f"intermediate_{model_name}.npz"
            np.savez(
                npz_path,
                segment_probs=segment_probs,
                segment_labels=segment_labels,
                segment_subject_ids=np.array(segment_subject_ids),
                subjects=subjects,
                y_true_subject=y_true_subject,
                y_pred_subject=y_pred_subject,
                p_subject=np.array([p_subject_dict[s] for s in subjects])
            )
            print(f"Saved intermediate arrays to {npz_path}")
    
    # Prepare final results
    final_results = {
        'train_counts_subject': train_manifest['label'].value_counts().to_dict(),
        'train_proportions_subject': {int(k): float(v) for k, v in enumerate(train_proportions)},
        'majority_class': int(majority_class),
        'test_subject_count': len(subjects),
        'test_segments_count': len(segment_subject_ids),
        'models': all_results,
        'bootstrap_B': args.bootstrap_B,
        'seed': args.seed,
        'primary_metric': args.primary_metric,
        'permutation_M': args.permutation_M,
        'timestamp': datetime.now().isoformat(),
        'git_commit': get_git_commit_hash()
    }
    
    # Add segments per subject summary (from first model)
    if all_results:
        first_model = list(all_results.keys())[0]
        final_results['segments_per_subject'] = all_results[first_model]['segments_per_subject']
    
    # Save results.json
    results_path = out_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Save summary.csv
    summary_rows = []
    for model_name, model_results in all_results.items():
        bootstrap_summary = model_results['bootstrap_summary']
        for metric_name, metric_summary in bootstrap_summary.items():
            summary_rows.append({
                'model': model_name,
                'metric': metric_name,
                'point_estimate': metric_summary['point_estimate'],
                'ci_low': metric_summary['ci_low'],
                'ci_high': metric_summary['ci_high'],
                'delta_point': metric_summary['delta_point'],
                'delta_ci_low': metric_summary['delta_ci_low'],
                'delta_ci_high': metric_summary['delta_ci_high'],
                'p_delta_gt_0': metric_summary['p_delta_gt_0']
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Primary metric: {args.primary_metric}")
    print(f"\nTrain proportions: {dict(zip([0,1,2], train_proportions))}")
    print(f"Majority class: {majority_class}")
    print(f"Test subjects: {len(subjects)}")
    print(f"Test segments: {len(segment_subject_ids)}")
    
    for model_name, model_results in all_results.items():
        print(f"\n{model_name}:")
        primary_summary = model_results['bootstrap_summary'][args.primary_metric]
        print(f"  {args.primary_metric}: {primary_summary['point_estimate']:.4f} "
              f"[{primary_summary['ci_low']:.4f}, {primary_summary['ci_high']:.4f}]")
        print(f"  Delta (vs random): {primary_summary['delta_point']:.4f} "
              f"[{primary_summary['delta_ci_low']:.4f}, {primary_summary['delta_ci_high']:.4f}]")
        print(f"  P(delta > 0): {primary_summary['p_delta_gt_0']:.4f}")
        if model_results['permutation_p_value'] is not None:
            print(f"  Permutation p-value: {model_results['permutation_p_value']:.4f}")
    
    print(f"\nResults saved to: {out_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
