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
"""

import time
from pathlib import Path
import logging
from eeg_data_preprocessing import EEGDataPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to run the user identification preprocessing pipeline."""
    # Configuration - make these explicit and configurable
    data_root = "/home/mojtabam/scratch/preprocessed_new"
    output_dir = "/home/mojtabam/scratch/processed_eeg_data_user_identification"
    
    # Split ratios - explicit, no hardcoding
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    # Random seed for reproducibility
    random_seed = 42
    
    # Record start time
    start_time = time.time()
    
    # Create preprocessor (sequential processing only)
    preprocessor = EEGDataPreprocessor(data_root, output_dir)
    
    # Process all participants for user identification
    logger.info("Starting EEG data preprocessing for user identification...")
    logger.info(f"Data will be saved to: {output_dir}")
    logger.info("Creating 1s, 2s, and 4s segments with train/val/test splits")
    logger.info(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    logger.info("CRITICAL: Each participant's samples will be split (not participants)")
    logger.info("  - ALL participants appear in ALL splits (train/val/test)")
    logger.info("  - Each participant: 70% samples -> train, 15% -> val, 15% -> test")
    logger.info("This allows model to learn features for all 3145 users during training")
    logger.info("Validation/test check if model can identify known users on new samples")
    logger.info("Each segment will have participant_id stored in metadata")
    
    try:
        preprocessor.process_all_participants_user_identification(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed
        )
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # Calculate and log elapsed time
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    logger.info("Preprocessing completed successfully!")
    logger.info(f"All data saved to: {output_dir}")
    logger.info(f"Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d} ({elapsed_time:.2f} seconds)")

if __name__ == "__main__":
    main()

