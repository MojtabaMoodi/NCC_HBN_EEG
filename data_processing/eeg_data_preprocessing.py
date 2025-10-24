#!/usr/bin/env python3
"""
EEG Data Preprocessing Script

This script processes EEG data from the preprocessed_new directory structure:
- Reads participant directories from multiple releases
- Processes EEG data files (ccd = active task, sus = passive task)
- Converts (2, 60, 240) EEG data to (60, 200) by dropping first 40 time points
- Creates both 1-second and 4-second segments for different model requirements
- Organizes data by task type and participant demographics
- Creates pickle files for every 10 participants
"""

import os
import numpy as np
import pandas as pd
import pickle
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EEGDataPreprocessor:
    """Class to handle EEG data preprocessing and organization."""
    
    def __init__(self, data_root: str, output_dir_1s: str, output_dir_4s: str):
        """
        Initialize the preprocessor.
        
        Args:
            data_root: Path to the preprocessed_new directory
            output_dir_1s: Directory to save 1-second pickle files
            output_dir_4s: Directory to save 4-second pickle files
        """
        self.data_root = Path(data_root)
        self.output_dir_1s = Path(output_dir_1s)
        self.output_dir_4s = Path(output_dir_4s)
        self.output_dir_1s.mkdir(parents=True, exist_ok=True)
        self.output_dir_4s.mkdir(parents=True, exist_ok=True)
        
    def load_demographics(self, participant_dir: Path) -> Dict[str, Any]:
        """
        Load demographic information for a participant.
        
        Args:
            participant_dir: Path to participant directory
            
        Returns:
            Dictionary with demographic information, or None if demographics file doesn't exist
            
        Raises:
            ValueError: If age or gender is missing or invalid when demographics file exists
        """
        demo_file = participant_dir / "demographics.csv"
        if not demo_file.exists():
            logger.warning(f"No demographics file found for {participant_dir.name}")
            return None
            
        try:
            df = pd.read_csv(demo_file)
            if len(df) == 0:
                raise ValueError(f"Empty demographics file for {participant_dir.name}")
                
            # Extract age and gender - these are required
            age = df.iloc[0]['age']
            gender = df.iloc[0]['sex']
            
            # Validate required fields
            if pd.isna(age) or age is None:
                raise ValueError(f"Missing age for {participant_dir.name}")
            if pd.isna(gender) or gender is None:
                raise ValueError(f"Missing gender for {participant_dir.name}")
                
            return {
                'age': float(age),
                'gender': int(gender),  # 1.0 = male, 0.0 = female
                'handedness': float(df.iloc[0]['handedness']) if not pd.isna(df.iloc[0]['handedness']) else None,
                'p_factor': float(df.iloc[0]['p_factor']) if not pd.isna(df.iloc[0]['p_factor']) else None,
                'attention': float(df.iloc[0]['attention']) if not pd.isna(df.iloc[0]['attention']) else None,
                'internalizing': float(df.iloc[0]['internalizing']) if not pd.isna(df.iloc[0]['internalizing']) else None,
                'externalizing': float(df.iloc[0]['externalizing']) if not pd.isna(df.iloc[0]['externalizing']) else None
            }
        except Exception as e:
            logger.error(f"Error loading demographics for {participant_dir.name}: {e}")
            raise
    
    def process_eeg_file(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single EEG file by dropping first 40 time points (0.2 seconds at 200 Hz).
        
        Args:
            file_path: Path to the .npy file
            
        Returns:
            Tuple of two processed EEG data arrays (run1, run2), each with shape (60, 200)
        """
        try:
            data = np.load(file_path)
            if data.shape != (2, 60, 240):
                logger.warning(f"Unexpected shape {data.shape} for file {file_path}")
                return None, None
                
            # Drop first 40 time points for both runs
            run1_data = data[0, :, 40:]  # Shape: (60, 200) - First run
            run2_data = data[1, :, 40:]  # Shape: (60, 200) - Second run
            return run1_data, run2_data
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None, None
    
    def create_4s_segments(self, eeg_runs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Create 4-second segments by concatenating 4 consecutive 1-second runs.
        Each run is 1 second (200 samples), so 4 runs = 4 seconds (800 samples).
        
        Args:
            eeg_runs: List of (60, 200) arrays representing 1-second runs
            
        Returns:
            List of (60, 800) arrays representing 4-second segments
        """
        if len(eeg_runs) == 0:
            return []
            
        segments_4s = []
        
        # Group runs into sets of 4 for concatenation
        for i in range(0, len(eeg_runs) - 3, 4):  # Step by 4, ensuring we have 4 consecutive runs
            # Concatenate 4 consecutive runs along time axis
            # Each run is (60, 200), so 4 runs concatenated = (60, 800)
            segment = np.concatenate(eeg_runs[i:i+4], axis=1)  # Shape: (60, 800)
            segments_4s.append(segment)
            logger.debug(f"Created 4s segment from runs {i} to {i+3}")
        
        logger.debug(f"Created {len(segments_4s)} 4-second segments from {len(eeg_runs)} runs")
        return segments_4s
    
    def process_participant(self, participant_dir: Path) -> Dict[str, Any]:
        """
        Process all EEG files for a single participant.
        
        Args:
            participant_dir: Path to participant directory
            
        Returns:
            Dictionary containing processed participant data, or None if demographics missing
            
        Raises:
            ValueError: If age or gender is missing or invalid when demographics file exists
        """
        participant_id = participant_dir.name
        
        # Load demographics - this will return None if file doesn't exist
        demographics = self.load_demographics(participant_dir)
        
        # Skip participant if no demographics
        if demographics is None:
            return None
        
        # Initialize data structures
        passive_eeg = []
        active_eeg = []
        
        # Process all .npy files
        npy_files = list(participant_dir.glob("*.npy"))
        
        # Sort by trial number (extract number from filename for proper sorting)
        def get_trial_number(file_path):
            filename = file_path.stem
            if filename.startswith("ccd_data_trial_"):
                return int(filename.split("_")[-1])
            elif filename.startswith("sus_data_trial_"):
                return int(filename.split("_")[-1])
            else:
                raise ValueError(f"Unknown file type: {filename}")
        
        npy_files.sort(key=get_trial_number)  # Sort by trial number

        for file_path in npy_files:
            filename = file_path.stem  # e.g., "ccd_data_trial_0" or "sus_data_trial_0"
            
            # Determine task type
            if filename.startswith("ccd"):
                task_type = "active"
            elif filename.startswith("sus"):
                task_type = "passive"
            else:
                logger.warning(f"Unknown file type: {filename}")
                continue
            
            # Process EEG data
            run1_data, run2_data = self.process_eeg_file(file_path)
            if run1_data is not None and run2_data is not None:
                if task_type == "active":
                    active_eeg.append(run1_data)  # Add run1
                    active_eeg.append(run2_data)  # Add run2
                else:  # passive
                    passive_eeg.append(run1_data)  # Add run1
                    passive_eeg.append(run2_data)  # Add run2
        
        # Create 4-second segments for both task types
        passive_eeg_4s = self.create_4s_segments(passive_eeg)
        active_eeg_4s = self.create_4s_segments(active_eeg)
        
        return {
            'participant_id': participant_id,
            'gender': demographics['gender'],  # Now guaranteed to exist
            'age': demographics['age'],  # Now guaranteed to exist
            'passive_eeg': passive_eeg,  # 1-second segments (list of arrays)
            'active_eeg': active_eeg,    # 1-second segments (list of arrays)
            'passive_eeg_4s': passive_eeg_4s,  # 4-second segments (list of arrays)
            'active_eeg_4s': active_eeg_4s,    # 4-second segments (list of arrays)
            'demographics': demographics
        }
    
    def process_all_participants(self, batch_size: int = 10) -> None:
        """
        Process all participants from all releases and save in batches.
        
        Args:
            batch_size: Number of participants per pickle file (exactly this many)
        """
        # Get all release directories
        release_dirs = [d for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith("cmi_bids_R")]
        release_dirs.sort()  # Process in order
        
        logger.info(f"Found {len(release_dirs)} release directories")
        
        # Collect all participant directories first
        all_participant_dirs = []
        for release_dir in release_dirs:
            participant_dirs = [d for d in release_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")]
            all_participant_dirs.extend(participant_dirs)
        
        total_participants = len(all_participant_dirs)
        logger.info(f"Found {total_participants} total participants")
        
        # Process participants to get exactly batch_size per pickle file
        batch_idx = 0
        participant_idx = 0
        processed_count = 0
        skipped_count = 0
        
        while participant_idx < total_participants:
            batch_participants = []
            logger.info(f"Processing batch {batch_idx + 1} (looking for {batch_size} valid participants)")
            
            # Keep processing until we have exactly batch_size participants
            while len(batch_participants) < batch_size and participant_idx < total_participants:
                participant_dir = all_participant_dirs[participant_idx]
                logger.info(f"Processing participant {participant_idx + 1}/{total_participants}: {participant_dir.name}")
                
                try:
                    participant_data = self.process_participant(participant_dir)
                    if participant_data is not None:
                        batch_participants.append(participant_data)
                        processed_count += 1
                        logger.info(f"Successfully processed participant {participant_dir.name}")
                    else:
                        logger.info(f"Skipping participant {participant_dir.name}: No demographics available")
                        skipped_count += 1
                except ValueError as e:
                    logger.warning(f"Skipping participant {participant_dir.name}: {e}")
                    skipped_count += 1
                except Exception as e:
                    logger.error(f"Unexpected error processing participant {participant_dir.name}: {e}")
                    skipped_count += 1
                
                participant_idx += 1
            
            # Save this batch if we have any participants
            if batch_participants:
                self._save_batch(batch_participants, batch_idx, batch_size, "1s", self.output_dir_1s)
                self._save_batch(batch_participants, batch_idx, batch_size, "4s", self.output_dir_4s)
                batch_idx += 1
                
                # Clear memory
                del batch_participants
                import gc
                gc.collect()
            else:
                logger.warning(f"No valid participants found in batch {batch_idx + 1}")
                break
        
        logger.info(f"Processing completed!")
        logger.info(f"Total participants processed: {processed_count}")
        logger.info(f"Total participants skipped: {skipped_count}")
        logger.info(f"Total batches created: {batch_idx}")
    
    def _save_batch(self, batch_participants: List[Dict[str, Any]], batch_idx: int, batch_size: int, 
                   segment_type: str, output_dir: Path):
        """
        Save a batch of participants to a pickle file.
        
        Args:
            batch_participants: List of processed participant data
            batch_idx: Batch index
            batch_size: Expected batch size
            segment_type: Type of segments ("1s" or "4s")
            output_dir: Directory to save the pickle file
        """
        actual_count = len(batch_participants)
        
        # Create version with appropriate segment type
        batch_data = []
        for participant in batch_participants:
            if segment_type == "1s":
                participant_data = {
                    'participant_id': participant['participant_id'],
                    'gender': participant['gender'],
                    'age': participant['age'],
                    'passive_eeg': participant['passive_eeg'],  # List of (60, 200) arrays
                    'active_eeg': participant['active_eeg'],    # List of (60, 200) arrays
                    'demographics': participant['demographics']
                }
            else:  # 4s
                participant_data = {
                    'participant_id': participant['participant_id'],
                    'gender': participant['gender'],
                    'age': participant['age'],
                    'passive_eeg': participant['passive_eeg_4s'],  # List of (60, 800) arrays
                    'active_eeg': participant['active_eeg_4s'],    # List of (60, 800) arrays
                    'demographics': participant['demographics']
                }
            batch_data.append(participant_data)
        
        # Use actual count in filename to reflect the real number of participants
        pickle_filename = f"eeg_batch_{actual_count}-participants_{batch_idx:03d}.pkl"
        pickle_path = output_dir / pickle_filename
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(batch_data, f)
        
        logger.info(f"Saved {segment_type} batch {batch_idx + 1} to {pickle_path}")
        logger.info(f"{segment_type} batch contains {actual_count} participants")
        
        # Log warning if we have fewer than expected (except for the last batch)
        if actual_count < batch_size:
            logger.warning(f"{segment_type} batch {batch_idx + 1} has {actual_count} participants instead of expected {batch_size}")


def main():
    """Main function to run the preprocessing pipeline."""
    data_root = "/home/mojtabam/scratch/preprocessed_new"
    output_dir_1s = "/home/mojtabam/projects/def-aghodsib/mojtabam/EEG/data_processing/processed_eeg_data_1s_segments"
    output_dir_4s = "/home/mojtabam/projects/def-aghodsib/mojtabam/EEG/data_processing/processed_eeg_data_4s_segments"
    
    # Create preprocessor
    preprocessor = EEGDataPreprocessor(data_root, output_dir_1s, output_dir_4s)
    
    # Process all participants in batches and save immediately
    logger.info("Starting EEG data preprocessing...")
    logger.info(f"1-second segments will be saved to: {output_dir_1s}")
    logger.info(f"4-second segments will be saved to: {output_dir_4s}")
    preprocessor.process_all_participants(batch_size=10)
    
    logger.info("Preprocessing completed successfully!")
    logger.info(f"1-second segments saved to: {output_dir_1s}")
    logger.info(f"4-second segments saved to: {output_dir_4s}")

if __name__ == "__main__":
    main()
