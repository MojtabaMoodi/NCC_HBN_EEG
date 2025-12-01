#!/usr/bin/env python3
"""
Script to balance and copy EEG data from preprocessed_new to preprocessed_new2.

This script:
1. Reads all demographics from all R directories
2. Categorizes participants by gender (male/female) and age groups (<8.5, 8.5-12.5, >12.5)
3. Randomly samples equal numbers from each of the 6 groups
4. Copies selected participants' data to the new location
"""

import os
import sys
import shutil
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import random

# Age classification thresholds (same as in constants.py)
AGE_CLASS_1_MAX = 8.5
AGE_CLASS_2_MAX = 12.5

def get_age_group(age: float) -> int:
    """Get age group index: 0 (<8.5), 1 (8.5-12.5), 2 (>12.5)"""
    if age < AGE_CLASS_1_MAX:
        return 0
    elif age <= AGE_CLASS_2_MAX:
        return 1
    else:
        return 2

def get_gender_label(gender: float) -> str:
    """Get gender label: 'F' for female (0.0), 'M' for male (1.0)"""
    if gender == 0.0:
        return 'F'
    elif gender == 1.0:
        return 'M'
    else:
        raise ValueError(f"Invalid gender value: {gender}")

def collect_all_participants(src_root: str) -> Tuple[List[Dict], Dict[Tuple[str, int], List[str]]]:
    """
    Collect all participants from all R directories.
    
    Returns:
        - List of all participant records
        - Dictionary mapping (gender, age_group) to list of participant paths
    """
    all_participants = []
    participants_by_group = defaultdict(list)
    
    print("Scanning for participants...")
    
    for r_dir in sorted(os.listdir(src_root)):
        if not r_dir.startswith('cmi_bids_R'):
            continue
        
        r_path = os.path.join(src_root, r_dir)
        if not os.path.isdir(r_path):
            continue
        
        print(f"  Processing {r_dir}...")
        participant_count = 0
        
        for sub_dir in sorted(os.listdir(r_path)):
            if not sub_dir.startswith('sub-'):
                continue
            
            sub_path = os.path.join(r_path, sub_dir)
            demographics_file = os.path.join(sub_path, 'demographics.csv')
            
            if not os.path.exists(demographics_file):
                continue
            
            try:
                # Read demographics
                df = pd.read_csv(demographics_file)
                if len(df) == 0:
                    continue
                
                age = df['age'].iloc[0]
                sex = df['sex'].iloc[0]
                
                # Validate data
                if pd.isna(age) or pd.isna(sex):
                    continue
                
                age_group = get_age_group(age)
                gender_label = get_gender_label(sex)
                
                participant_record = {
                    'r_dir': r_dir,
                    'sub_dir': sub_dir,
                    'full_path': sub_path,
                    'age': age,
                    'gender': gender_label,
                    'age_group': age_group,
                    'group_key': (gender_label, age_group)
                }
                
                all_participants.append(participant_record)
                participants_by_group[(gender_label, age_group)].append(sub_path)
                participant_count += 1
                
            except Exception as e:
                print(f"    Warning: Could not process {sub_path}: {e}")
                continue
        
        print(f"    Found {participant_count} participants")
    
    return all_participants, participants_by_group

def print_distribution(participants_by_group: Dict[Tuple[str, int], List[str]]):
    """Print the distribution of participants across groups."""
    age_group_names = ["<8.5", "8.5-12.5", ">12.5"]
    
    print("\n" + "="*60)
    print("PARTICIPANT DISTRIBUTION")
    print("="*60)
    
    for gender in ['F', 'M']:
        print(f"\n{gender}emales:")
        for age_group in range(3):
            key = (gender, age_group)
            count = len(participants_by_group.get(key, []))
            print(f"  {age_group_names[age_group]}: {count}")
    
    # Find minimum
    min_count = min(len(participants_by_group.get((g, a), [])) 
                    for g in ['F', 'M'] for a in range(3))
    print(f"\nMinimum count across all groups: {min_count}")

def balance_and_copy(participants_by_group: Dict[Tuple[str, int], List[str]], 
                    src_root: str, dst_root: str, min_count: int = None):
    """
    Balance participants and copy to destination.
    
    Args:
        participants_by_group: Dictionary mapping (gender, age_group) to participant paths
        src_root: Source directory root
        dst_root: Destination directory root
        min_count: Minimum count per group (if None, will compute from data)
    """
    if min_count is None:
        min_count = min(len(participants_by_group.get((g, a), [])) 
                       for g in ['F', 'M'] for a in range(3))
    
    print(f"\nSampling {min_count} participants from each of the 6 groups...")
    print("="*60)
    
    selected_participants = []
    
    for gender in ['F', 'M']:
        for age_group in range(3):
            key = (gender, age_group)
            group_participants = participants_by_group.get(key, [])
            
            if len(group_participants) == 0:
                print(f"Warning: No participants for {gender}, age_group {age_group}")
                continue
            
            # Randomly sample
            np.random.seed(42)  # For reproducibility
            selected = np.random.choice(group_participants, 
                                       size=min(min_count, len(group_participants)),
                                       replace=False)
            
            selected_participants.extend(selected)
            
            age_group_names = ["<8.5", "8.5-12.5", ">12.5"]
            print(f"  Selected {len(selected)} from {gender}emales, {age_group_names[age_group]}")
    
    print(f"\nTotal participants to copy: {len(selected_participants)}")
    
    # Create destination directory structure and copy
    print("\nCopying participants...")
    copied_count = 0
    
    for participant_path in selected_participants:
        # Determine the R directory and subject ID from the path
        rel_path = os.path.relpath(participant_path, src_root)
        parts = rel_path.split(os.sep)
        r_dir = parts[0]
        sub_dir = parts[1]
        
        # Create destination structure: preprocessed_new2/R_dir/sub_dir
        dst_path = os.path.join(dst_root, r_dir, sub_dir)
        dst_parent = os.path.dirname(dst_path)
        
        # Create parent directory if needed
        os.makedirs(dst_parent, exist_ok=True)
        
        # Copy entire directory using rsync for reliability
        if os.path.exists(dst_path):
            print(f"  Skipping {dst_path} (already exists)")
        else:
            try:
                # Ensure path is a string, not bytes
                participant_path_str = str(participant_path)
                dst_path_str = str(dst_path)
                
                # Use shell command for more robust copying
                result = subprocess.run(['cp', '-r', participant_path_str, dst_path_str], 
                                       capture_output=True, text=True)
                if result.returncode == 0:
                    copied_count += 1
                    if copied_count % 50 == 0:
                        print(f"  Copied {copied_count} participants...")
                else:
                    # Try with rsync as fallback
                    result2 = subprocess.run(['rsync', '-av', participant_path_str + '/', dst_path_str + '/'], 
                                           capture_output=True, text=True)
                    if result2.returncode == 0:
                        copied_count += 1
                        if copied_count % 50 == 0:
                            print(f"  Copied {copied_count} participants...")
                    else:
                        print(f"  Error copying {participant_path}: {result.stderr if result.stderr else result2.stderr}")
            except Exception as e:
                print(f"  Error copying {participant_path}: {e}")
    
    print(f"\n✅ Successfully copied {copied_count} participants")
    
    # Verify final distribution
    print("\nVerifying final distribution in destination...")
    verify_distribution(dst_root)

def verify_distribution(dst_root: str):
    """Verify the final distribution in the destination directory."""
    participants_by_group = defaultdict(list)
    
    for r_dir in sorted(os.listdir(dst_root)):
        if not r_dir.startswith('cmi_bids_R'):
            continue
        
        r_path = os.path.join(dst_root, r_dir)
        if not os.path.isdir(r_path):
            continue
        
        for sub_dir in sorted(os.listdir(r_path)):
            if not sub_dir.startswith('sub-NDAR'):
                continue
            
            sub_path = os.path.join(r_path, sub_dir)
            demographics_file = os.path.join(sub_path, 'demographics.csv')
            
            if not os.path.exists(demographics_file):
                continue
            
            try:
                df = pd.read_csv(demographics_file)
                if len(df) == 0:
                    continue
                
                age = df['age'].iloc[0]
                sex = df['sex'].iloc[0]
                
                if pd.isna(age) or pd.isna(sex):
                    continue
                
                age_group = get_age_group(age)
                gender_label = get_gender_label(sex)
                participants_by_group[(gender_label, age_group)].append(sub_path)
                
            except Exception as e:
                continue
    
    # Print final distribution
    age_group_names = ["<8.5", "8.5-12.5", ">12.5"]
    
    print("\n" + "="*60)
    print("FINAL DISTRIBUTION IN DESTINATION")
    print("="*60)
    
    for gender in ['F', 'M']:
        print(f"\n{gender}emales:")
        for age_group in range(3):
            key = (gender, age_group)
            count = len(participants_by_group.get(key, []))
            print(f"  {age_group_names[age_group]}: {count}")
    
    # Check if balanced
    counts = [len(participants_by_group.get((g, a), [])) 
              for g in ['F', 'M'] for a in range(3)]
    if len(set(counts)) == 1:
        print(f"\n✅ Data is balanced! Each group has {counts[0]} participants.")
    else:
        print(f"\n⚠️  Data is not perfectly balanced. Counts: {counts}")

def main():
    src_root = "/home/mojtabam/projects/aip-aghodsib/mojtabam/preprocessed_new"
    dst_root = "/home/mojtabam/projects/aip-aghodsib/mojtabam/preprocessed_new2"
    
    print("="*60)
    print("BALANCING AND COPYING EEG DATA")
    print("="*60)
    print(f"Source: {src_root}")
    print(f"Destination: {dst_root}")
    print("="*60)
    
    # Step 1: Collect all participants
    all_participants, participants_by_group = collect_all_participants(src_root)
    
    print(f"\nTotal participants found: {len(all_participants)}")
    
    # Step 2: Print distribution
    print_distribution(participants_by_group)
    
    # Step 3: Balance and copy
    balance_and_copy(participants_by_group, src_root, dst_root)
    
    print("\n" + "="*60)
    print("✅ PROCESSING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()

