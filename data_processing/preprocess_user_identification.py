#!/usr/bin/env python3
"""
EEG Data Preprocessing Script for User Identification

This script processes EEG data for user identification task:
- Reads each user's EEG data
- Splits each user's data at sample level: 70% train, 15% val, 15% test
- Creates 1-second, 2-second, and 4-second segments
- Organizes data by task type (active and passive)
- Saves data in HDF5 format with train/val/test splits
- Creates participant_id_to_class_idx.json mapping file

Optional (--consider_unknown): Hold out a fraction of users as "unknown" for open-set evaluation.
- Known users: 70/15/15 train/val/test; unknown users: all samples in "unknown" split.

Optional (--n_folds): 5-fold (or N-fold) data preparation with disjoint unknown sets.
- Participants are partitioned into n_folds disjoint groups (~20%% each for n_folds=5).
- Each fold k is written to output_dir/fold_k with that fold's group as unknown and the rest as known.
- No overlap: each participant is unknown in exactly one fold.
"""

import argparse
import time
from pathlib import Path
import logging
import numpy as np
from eeg_data_preprocessing import EEGDataPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _partition_into_folds(all_dirs, n_folds: int, random_seed: int):
    """
    Partition participant dirs into n_folds disjoint groups (as equal sizes as possible).
    Returns list of length n_folds; each element is a list of Paths (unknown for that fold).
    """
    n = len(all_dirs)
    if n < n_folds:
        raise ValueError(
            f"Number of participants ({n}) is less than n_folds ({n_folds}). "
            "Cannot create disjoint folds."
        )
    rng = np.random.RandomState(random_seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    # Assign each index to a fold: fold 0 gets indices 0, n_folds, 2*n_folds, ...
    fold_indices = [indices[i::n_folds].tolist() for i in range(n_folds)]
    return [[all_dirs[j] for j in fi] for fi in fold_indices]


def parse_args():
    """Parse command-line arguments (backward compatible: no args = previous behavior)."""
    parser = argparse.ArgumentParser(
        description='Preprocess EEG data for user identification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_root', type=str,
                        default='/home/mojtabam/scratch/preprocessed_new',
                        help='Path to preprocessed data root')
    parser.add_argument('--output_dir', type=str,
                        default='/home/mojtabam/scratch/processed_eeg_data_user_identification',
                        help='Output directory for HDF5 files and mapping')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Fraction of each participant samples for train')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Fraction of each participant samples for val')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Fraction of each participant samples for test')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for splits')
    parser.add_argument('--consider_unknown', action='store_true',
                        help='Hold out a fraction of users as "unknown" for open-set evaluation')
    parser.add_argument('--unknown_ratio', type=float, default=0.2,
                        help='Fraction of participants to treat as unknown when --consider_unknown')
    parser.add_argument('--n_folds', type=int, default=None,
                        help='If set (e.g. 5), create n_folds disjoint folds: each fold has 1/n_folds participants as unknown, no overlap')
    parser.add_argument('--segment_length', type=str, default='4s',
                        choices=['1s', '2s', '4s'],
                        help='Segment length to preprocess (only this length will be generated)')
    return parser.parse_args()


def main():
    """Main function to run the user identification preprocessing pipeline."""
    args = parse_args()
    
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, "
            f"got {args.train_ratio + args.val_ratio + args.test_ratio}"
        )
    if args.n_folds is not None:
        if args.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {args.n_folds}")
    else:
        if args.consider_unknown and (args.unknown_ratio <= 0 or args.unknown_ratio >= 1.0):
            raise ValueError(f"unknown_ratio must be in (0, 1), got {args.unknown_ratio}")
    
    # Record start time
    start_time = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.n_folds is not None:
        # 5-fold (or N-fold): disjoint unknown sets, one output dir per fold
        preprocessor_for_dirs = EEGDataPreprocessor(
            args.data_root,
            str(output_dir),
            window_sizes=[args.segment_length]
        )
        all_participant_dirs = preprocessor_for_dirs.get_all_participant_dirs()
        if not all_participant_dirs:
            raise ValueError(f"No participant directories found under {args.data_root}")
        fold_unknown_dirs = _partition_into_folds(
            all_participant_dirs, args.n_folds, args.random_seed
        )
        logger.info(
            "Starting %d-fold user identification preprocessing (disjoint unknown sets)...",
            args.n_folds
        )
        logger.info(f"Data root: {args.data_root}")
        logger.info(f"Output dir: {args.output_dir} (fold_0 .. fold_%d)", args.n_folds - 1)
        logger.info(f"Segment length: {args.segment_length}, split ratios: %s/%s/%s",
                    args.train_ratio, args.val_ratio, args.test_ratio)
        for k in range(args.n_folds):
            fold_output = output_dir / f"fold_{k}"
            fold_output.mkdir(parents=True, exist_ok=True)
            unknown_dirs = fold_unknown_dirs[k]
            unknown_names = {p.name for p in unknown_dirs}
            known_dirs = [d for d in all_participant_dirs if d.name not in unknown_names]
            logger.info("Fold %d: %d known, %d unknown", k, len(known_dirs), len(unknown_dirs))
            preprocessor = EEGDataPreprocessor(
                args.data_root,
                str(fold_output),
                window_sizes=[args.segment_length]
            )
            try:
                preprocessor.process_all_participants_user_identification(
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    random_seed=args.random_seed,
                    consider_unknown_users=False,
                    unknown_user_ratio=0.2,
                    known_participant_dirs=known_dirs,
                    unknown_participant_dirs=unknown_dirs
                )
            except Exception as e:
                logger.error("Preprocessing failed for fold %d: %s", k, e)
                import traceback
                logger.error(traceback.format_exc())
                raise
        logger.info("All %d folds written under %s", args.n_folds, args.output_dir)
    else:
        # Single run (original behavior)
        preprocessor = EEGDataPreprocessor(
            args.data_root,
            args.output_dir,
            window_sizes=[args.segment_length]
        )
        logger.info("Starting EEG data preprocessing for user identification...")
        logger.info(f"Data root: {args.data_root}")
        logger.info(f"Output dir: {args.output_dir}")
        logger.info(f"Creating {args.segment_length} segments with train/val/test splits")
        logger.info(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
        if args.consider_unknown:
            logger.info(f"consider_unknown=True: {args.unknown_ratio:.1%} of users will be held out as 'unknown'")
            logger.info("  - Known users: 70/15/15 train/val/test; unknown users: all samples in 'unknown' split")
        else:
            logger.info("  - ALL participants appear in ALL splits (train/val/test)")
        logger.info("  - Each participant: 70%% samples -> train, 15%% -> val, 15%% -> test")
        logger.info("Each segment will have participant_id stored in metadata")
        
        try:
            preprocessor.process_all_participants_user_identification(
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                random_seed=args.random_seed,
                consider_unknown_users=args.consider_unknown,
                unknown_user_ratio=args.unknown_ratio
            )
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    logger.info("Preprocessing completed successfully!")
    logger.info(f"All data saved to: {args.output_dir}")
    logger.info(f"Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d} ({elapsed_time:.2f} seconds)")


if __name__ == "__main__":
    main()

