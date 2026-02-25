#!/usr/bin/env python3
"""
Preprocess EEG data to multipart HDF5 for age and gender identification.

Supports any channel count (e.g. 60, 64, 128) via --num_channels. Input can be
.npy or .h5 (layout: cmi_bids_R* / sub-* / demographics.csv + data files).
Outputs train/val/test HDF5 with gender and age in metadata; splits are stratified
by --stratify_by (default: both age and gender).

Multi-part output: When a split exceeds 100K samples, files are split automatically
into part00, part01, ... (e.g. eeg_data_train_4s_part00.h5, eeg_data_train_4s_part01.h5)
for train, val, and test separately.

Usage:
    # 128-channel .h5 (default)
    python preprocess_eeg_to_hdf5.py --window_sizes 1s 4s

    # 60- or 64-channel .npy for age and gender
    python preprocess_eeg_to_hdf5.py --data_root /path/to/preprocessed_new --output_dir /path/to/out --num_channels 60 --window_sizes 1s 2s 4s
    python preprocess_eeg_to_hdf5.py --data_root /path/to/data --output_dir /path/to/out --num_channels 64 --window_sizes 1s 2s 4s
"""

import argparse
import time
import logging
from pathlib import Path
from eeg_data_preprocessing import EEGDataPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess EEG to multipart HDF5 for age/gender (any channel count: 60, 64, 128, etc.)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 128ch .h5: 1s and 4s windows
  python preprocess_eeg_to_hdf5.py --window_sizes 1s 4s

  # 60ch or 64ch .npy for age and gender, 1s/2s/4s (output: multi-part when >100K samples/split)
  python preprocess_eeg_to_hdf5.py --data_root /path/to/preprocessed_new --output_dir /path/to/out --num_channels 60 --window_sizes 1s 2s 4s
  python preprocess_eeg_to_hdf5.py --data_root /path/to/data --output_dir /path/to/out --num_channels 64 --window_sizes 1s 2s 4s
        """
    )
    
    parser.add_argument('--data_root', type=str,
                       default='/home/mojtabam/scratch/preprocessed_128_channels',
                       help='Path to input data (cmi_bids_R* with sub-*; .npy or .h5)')
    parser.add_argument('--output_dir', type=str,
                       default='/home/mojtabam/scratch/processed_eeg_data_128ch',
                       help='Path to output directory for HDF5 files')
    parser.add_argument('--num_channels', type=int, default=128,
                       help='Number of EEG channels (e.g. 60, 64, 128); must match input data')
    parser.add_argument('--window_sizes', type=str, nargs='+', default=['4s'],
                       choices=['1s', '2s', '4s'],
                       help='Window sizes to process (default: 4s). Can specify multiple: --window_sizes 1s 4s')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Ratio of data for training (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Ratio of data for validation (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Ratio of data for testing (default: 0.15)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducible splits (default: 42)')
    parser.add_argument('--stratify_by', type=str, default='both',
                       choices=['gender', 'age', 'both'],
                       help='Field to stratify by (default: both)')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {args.train_ratio + args.val_ratio + args.test_ratio}")
    
    # Ensure window_sizes is a list (handle both single string and list)
    if isinstance(args.window_sizes, str):
        window_sizes = [args.window_sizes]
    else:
        window_sizes = list(args.window_sizes)
    
    # Validate data root exists
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root directory not found: {data_root}")
    
    # Record start time
    start_time = time.time()
    
    # Create preprocessor
    logger.info("=" * 80)
    logger.info("EEG Data Preprocessing (%d-channel, age & gender in metadata)", args.num_channels)
    logger.info("=" * 80)
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Number of channels: {args.num_channels}")
    logger.info(f"Window sizes: {window_sizes}")
    logger.info(f"Train/Val/Test ratios: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
    logger.info(f"Stratify by: {args.stratify_by}")
    logger.info(f"Random seed: {args.random_seed}")
    logger.info("=" * 80)
    
    preprocessor = EEGDataPreprocessor(
        data_root=str(data_root),
        output_dir=args.output_dir,
        num_channels=args.num_channels,
        window_sizes=window_sizes
    )
    
    # Process all participants and create train/val/test splits
    logger.info("Starting EEG data preprocessing...")
    preprocessor.process_all_participants(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
        stratify_by=args.stratify_by
    )
    
    # Calculate and log elapsed time
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    logger.info("=" * 80)
    logger.info("Preprocessing completed successfully!")
    logger.info(f"All data saved to: {args.output_dir}")
    logger.info(f"Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d} ({elapsed_time:.2f} seconds)")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
