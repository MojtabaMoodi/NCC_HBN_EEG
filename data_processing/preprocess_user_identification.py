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
"""

import argparse
import time
from pathlib import Path
import logging
from eeg_data_preprocessing import EEGDataPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    if args.consider_unknown and (args.unknown_ratio <= 0 or args.unknown_ratio >= 1.0):
        raise ValueError(f"unknown_ratio must be in (0, 1), got {args.unknown_ratio}")
    
    # Record start time
    start_time = time.time()
    
    # Only process the requested segment length (e.g. 4s)
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

