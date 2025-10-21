#!/usr/bin/env python3
"""
EEG Data Preprocessing Script

This script processes EEG data from the preprocessed_new directory structure:
- Reads participant directories from multiple releases
- Processes EEG data files (ccd = active task, sus = passive task)
- Converts (2, 60, 240) EEG data to (60, 200) by dropping first 40 time points
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
    
    def __init__(self, data_root: str, output_dir: str):
        """
        Initialize the preprocessor.
        
        Args:
            data_root: Path to the preprocessed_new directory
            output_dir: Directory to save pickle files
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_demographics(self, participant_dir: Path) -> Dict[str, Any]:
        """
        Load demographic information for a participant.
        
        Args:
            participant_dir: Path to participant directory
            
        Returns:
            Dictionary with demographic information
        """
        demo_file = participant_dir / "demographics.csv"
        if not demo_file.exists():
            logger.warning(f"No demographics file found for {participant_dir.name}")
            return {}
            
        try:
            df = pd.read_csv(demo_file)
            if len(df) > 0:
                return {
                    'age': float(df.iloc[0]['age']),
                    'gender': int(df.iloc[0]['sex']),  # 1.0 = male, 0.0 = female
                    'handedness': float(df.iloc[0]['handedness']),
                    'p_factor': float(df.iloc[0]['p_factor']),
                    'attention': float(df.iloc[0]['attention']),
                    'internalizing': float(df.iloc[0]['internalizing']),
                    'externalizing': float(df.iloc[0]['externalizing'])
                }
        except Exception as e:
            logger.error(f"Error loading demographics for {participant_dir.name}: {e}")
            
        return {}
    
    def process_eeg_file(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single EEG file by dropping first 40 time points.
        
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
    
    def process_participant(self, participant_dir: Path) -> Dict[str, Any]:
        """
        Process all EEG files for a single participant.
        
        Args:
            participant_dir: Path to participant directory
            
        Returns:
            Dictionary containing processed participant data
        """
        participant_id = participant_dir.name
        
        # Load demographics
        demographics = self.load_demographics(participant_dir)
        
        # Initialize data structures
        passive_eeg = {}
        active_eeg = {}
        
        # Process all .npy files
        npy_files = list(participant_dir.glob("*.npy"))
        
        for file_path in npy_files:
            filename = file_path.stem  # e.g., "ccd_data_trial_0" or "sus_data_trial_0"
            
            # Determine task type and extract task name
            if filename.startswith("ccd"):
                task_type = "active"
                # Extract trial number and create base task name
                trial_num = filename.split("_")[-1]  # e.g., "0" from "ccd_data_trial_0"
                base_task_name = f"ccd_trial_{trial_num}"
            elif filename.startswith("sus"):
                task_type = "passive"
                # Extract trial number and create base task name
                trial_num = filename.split("_")[-1]  # e.g., "0" from "sus_data_trial_0"
                base_task_name = f"sus_trial_{trial_num}"
            else:
                logger.warning(f"Unknown file type: {filename}")
                continue
            
            # Process EEG data
            run1_data, run2_data = self.process_eeg_file(file_path)
            if run1_data is not None and run2_data is not None:
                # Create task names for both runs
                task_name_run1 = f"{base_task_name}_run1"
                task_name_run2 = f"{base_task_name}_run2"
                
                if task_type == "active":
                    if task_name_run1 not in active_eeg:
                        active_eeg[task_name_run1] = []
                    else:
                        raise ValueError(f"Task name {task_name_run1} already exists in active_eeg")
                    if task_name_run2 not in active_eeg:
                        active_eeg[task_name_run2] = []
                    else:
                        raise ValueError(f"Task name {task_name_run2} already exists in active_eeg")
                    active_eeg[task_name_run1].append(run1_data)
                    active_eeg[task_name_run2].append(run2_data)
                else:  # passive
                    if task_name_run1 not in passive_eeg:
                        passive_eeg[task_name_run1] = []
                    else:
                        raise ValueError(f"Task name {task_name_run1} already exists in passive_eeg")
                    if task_name_run2 not in passive_eeg:
                        passive_eeg[task_name_run2] = []
                    else:
                        raise ValueError(f"Task name {task_name_run2} already exists in passive_eeg")
                    passive_eeg[task_name_run1].append(run1_data)
                    passive_eeg[task_name_run2].append(run2_data)
        
        return {
            'participant_id': participant_id,
            'gender': demographics.get('gender', None),
            'age': demographics.get('age', None),
            'passive_eeg': passive_eeg,
            'active_eeg': active_eeg,
            'demographics': demographics
        }
    
    def process_all_participants(self) -> List[Dict[str, Any]]:
        """
        Process all participants from all releases.
        
        Returns:
            List of processed participant data
        """
        all_participants = []
        
        # Get all release directories
        release_dirs = [d for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith("cmi_bids_R")]
        release_dirs.sort()  # Process in order
        
        logger.info(f"Found {len(release_dirs)} release directories")
        
        for release_dir in release_dirs:
            logger.info(f"Processing release: {release_dir.name}")
            
            # Get all participant directories
            participant_dirs = [d for d in release_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")]
            
            for participant_dir in participant_dirs:
                logger.info(f"Processing participant: {participant_dir.name}")
                participant_data = self.process_participant(participant_dir)
                all_participants.append(participant_data)
        
        logger.info(f"Processed {len(all_participants)} participants total")
        return all_participants
    
    def create_pickle_batches(self, participants: List[Dict[str, Any]], batch_size: int = 10):
        """
        Create pickle files for batches of participants.
        
        Args:
            participants: List of processed participant data
            batch_size: Number of participants per pickle file
        """
        total_participants = len(participants)
        num_batches = (total_participants + batch_size - 1) // batch_size
        
        logger.info(f"Creating {num_batches} pickle files with {batch_size} participants each")
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_participants)
            batch_participants = participants[start_idx:end_idx]
            
            pickle_filename = f"eeg_batch_{end_idx-start_idx}-participants_{i:03d}.pkl"
            pickle_path = self.output_dir / pickle_filename
            
            with open(pickle_path, 'wb') as f:
                pickle.dump(batch_participants, f)
            
            logger.info(f"Saved batch {i+1}/{num_batches} to {pickle_path}")
            logger.info(f"Batch contains {len(batch_participants)} participants")

def main():
    """Main function to run the preprocessing pipeline."""
    data_root = "/home/mojtabam/scratch/preprocessed_new"
    output_dir = "/home/mojtabam/projects/def-aghodsib/mojtabam/EEG/data_processing/processed_eeg_data"
    
    # Create preprocessor
    preprocessor = EEGDataPreprocessor(data_root, output_dir)
    
    # Process all participants
    logger.info("Starting EEG data preprocessing...")
    participants = preprocessor.process_all_participants()
    
    # Create pickle batches
    logger.info("Creating pickle batches...")
    preprocessor.create_pickle_batches(participants, batch_size=10)
    
    logger.info("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
