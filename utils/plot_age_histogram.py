#!/usr/bin/env python3
"""
Temporary script to plot histogram of participant ages from preprocessed_new directory.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_demographics(data_root: str) -> list:
    """
    Load demographic information from all participant directories.
    
    Args:
        data_root: Path to the preprocessed_new directory
        
    Returns:
        List of dictionaries containing demographic information
    """
    data_root = Path(data_root)
    demographics_list = []
    
    # Get all release directories
    release_dirs = [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("cmi_bids_R")]
    release_dirs.sort()
    
    logger.info(f"Found {len(release_dirs)} release directories")
    
    for release_dir in release_dirs:
        logger.info(f"Processing release: {release_dir.name}")
        
        # Get all participant directories
        participant_dirs = [d for d in release_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")]
        
        for participant_dir in participant_dirs:
            demo_file = participant_dir / "demographics.csv"
            if demo_file.exists():
                try:
                    df = pd.read_csv(demo_file)
                    if len(df) > 0:
                        demo_info = {
                            'participant_id': participant_dir.name,
                            'release': release_dir.name,
                            'age': float(df.iloc[0]['age']),
                            'gender': int(df.iloc[0]['sex']),  # 1.0 = male, 0.0 = female
                            'handedness': float(df.iloc[0]['handedness']),
                            'p_factor': float(df.iloc[0]['p_factor']),
                            'attention': float(df.iloc[0]['attention']),
                            'internalizing': float(df.iloc[0]['internalizing']),
                            'externalizing': float(df.iloc[0]['externalizing'])
                        }
                        demographics_list.append(demo_info)
                except Exception as e:
                    logger.error(f"Error loading demographics for {participant_dir.name}: {e}")
            else:
                logger.warning(f"No demographics file found for {participant_dir.name}")
    
    logger.info(f"Loaded demographics for {len(demographics_list)} participants")
    return demographics_list

def plot_age_histogram(demographics_list: list, save_path: str = None):
    """
    Plot histogram of participant ages.
    
    Args:
        demographics_list: List of demographic dictionaries
        save_path: Optional path to save the plot
    """
    # Extract ages
    ages = [demo['age'] for demo in demographics_list if demo['age'] is not None]
    
    if not ages:
        logger.error("No age data found!")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create histogram
    n, bins, patches = plt.hist(ages, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add statistics
    mean_age = np.mean(ages)
    median_age = np.median(ages)
    std_age = np.std(ages)
    min_age = np.min(ages)
    max_age = np.max(ages)
    
    # Add vertical lines for mean and median
    plt.axvline(mean_age, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_age:.2f}')
    plt.axvline(median_age, color='green', linestyle='--', linewidth=2, label=f'Median: {median_age:.2f}')
    
    # Customize the plot
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Number of Participants', fontsize=12)
    plt.title('Distribution of Participant Ages', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Total Participants: {len(ages)}\n'
    stats_text += f'Mean Age: {mean_age:.2f} ± {std_age:.2f}\n'
    stats_text += f'Median Age: {median_age:.2f}\n'
    stats_text += f'Age Range: {min_age:.2f} - {max_age:.2f}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def plot_age_by_gender(demographics_list: list, save_path: str = None):
    """
    Plot age distribution by gender.
    
    Args:
        demographics_list: List of demographic dictionaries
        save_path: Optional path to save the plot
    """
    # Separate by gender
    male_ages = [demo['age'] for demo in demographics_list if demo['gender'] == 1.0 and demo['age'] is not None]
    female_ages = [demo['age'] for demo in demographics_list if demo['gender'] == 0.0 and demo['age'] is not None]
    
    if not male_ages and not female_ages:
        logger.error("No age data found!")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot histograms
    if male_ages:
        plt.hist(male_ages, bins=20, alpha=0.7, label=f'Male (n={len(male_ages)})', color='lightblue', edgecolor='black')
    if female_ages:
        plt.hist(female_ages, bins=20, alpha=0.7, label=f'Female (n={len(female_ages)})', color='lightpink', edgecolor='black')
    
    # Customize the plot
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Number of Participants', fontsize=12)
    plt.title('Distribution of Participant Ages by Gender', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    if male_ages:
        male_mean = np.mean(male_ages)
        male_std = np.std(male_ages)
        plt.axvline(male_mean, color='blue', linestyle='--', alpha=0.7, label=f'Male Mean: {male_mean:.2f}')
    
    if female_ages:
        female_mean = np.mean(female_ages)
        female_std = np.std(female_ages)
        plt.axvline(female_mean, color='red', linestyle='--', alpha=0.7, label=f'Female Mean: {female_mean:.2f}')
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def print_demographic_summary(demographics_list: list):
    """
    Print summary statistics of demographics.
    
    Args:
        demographics_list: List of demographic dictionaries
    """
    if not demographics_list:
        logger.error("No demographic data found!")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(demographics_list)
    
    print("\n" + "="*60)
    print("DEMOGRAPHIC SUMMARY")
    print("="*60)
    
    # Basic counts
    print(f"Total participants: {len(df)}")
    print(f"Releases: {df['release'].nunique()}")
    print(f"Unique participants: {df['participant_id'].nunique()}")
    
    # Age statistics
    ages = df['age'].dropna()
    if len(ages) > 0:
        print(f"\nAge Statistics:")
        print(f"  Mean: {ages.mean():.2f} ± {ages.std():.2f}")
        print(f"  Median: {ages.median():.2f}")
        print(f"  Range: {ages.min():.2f} - {ages.max():.2f}")
        print(f"  Q1: {ages.quantile(0.25):.2f}")
        print(f"  Q3: {ages.quantile(0.75):.2f}")
    
    # Gender distribution
    gender_counts = df['gender'].value_counts()
    print(f"\nGender Distribution:")
    for gender, count in gender_counts.items():
        gender_label = "Male" if gender == 1.0 else "Female" if gender == 0.0 else f"Unknown ({gender})"
        print(f"  {gender_label}: {count} ({count/len(df)*100:.1f}%)")
    
    # Age by gender
    if 'gender' in df.columns and 'age' in df.columns:
        print(f"\nAge by Gender:")
        for gender in df['gender'].unique():
            if pd.notna(gender):
                gender_ages = df[df['gender'] == gender]['age'].dropna()
                if len(gender_ages) > 0:
                    gender_label = "Male" if gender == 1.0 else "Female" if gender == 0.0 else f"Unknown ({gender})"
                    print(f"  {gender_label}: {gender_ages.mean():.2f} ± {gender_ages.std():.2f} (n={len(gender_ages)})")
    
    # Release distribution
    print(f"\nRelease Distribution:")
    release_counts = df['release'].value_counts().sort_index()
    for release, count in release_counts.items():
        print(f"  {release}: {count} participants")
    
    print("="*60)

def main():
    """Main function to run the age histogram analysis."""
    data_root = "/home/mojtabam/scratch/preprocessed_new"
    
    logger.info("Loading demographic data...")
    demographics_list = load_demographics(data_root)
    
    if not demographics_list:
        logger.error("No demographic data found!")
        return
    
    # Print summary
    print_demographic_summary(demographics_list)
    
    # Plot age histogram
    logger.info("Creating age histogram...")
    plot_age_histogram(demographics_list, save_path="age_histogram.png")
    
    # Plot age by gender
    logger.info("Creating age by gender plot...")
    plot_age_by_gender(demographics_list, save_path="age_by_gender.png")
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()
