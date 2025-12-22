#!/usr/bin/env python3
"""
Check .npy files for unexpected dimensions.
Files with similar prefixes (e.g., sus_1, sus_2) should have the same dimensions.
"""

import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import csv
import matplotlib.pyplot as plt

def extract_prefix(filename):
    """
    Extract prefix from filename.
    Handles patterns like:
    - 'sus_1' from 'sus_1_data_trial_0.npy'
    - 'sus' from 'sus_data_trial_0.npy'
    - 'ccd_1' from 'ccd_1_data_trial_0.npy'
    - 'ccd' from 'ccd_data_trial_0.npy'
    """
    base_name = os.path.basename(filename)
    
    # First, try to match pattern with number: "sus_1", "ccd_1", etc.
    match_with_num = re.match(r'([a-zA-Z]+)_(\d+)(?:_|\.)', base_name)
    if match_with_num:
        return f"{match_with_num.group(1)}_{match_with_num.group(2)}"
    
    # Then, try to match pattern without number: "sus", "ccd", etc.
    # Look for known prefixes (sus, ccd, etc.) followed by underscore or end
    match_without_num = re.match(r'^(sus|ccd|CNDA|NDAR|TUAB|TUEV)(?:_|\.|$)', base_name)
    if match_without_num:
        return match_without_num.group(1)
    
    # Fallback: try to get first part before underscore
    parts = base_name.split('_')
    if len(parts) >= 1:
        # If first part is a known prefix, return it
        first_part = parts[0]
        if first_part.lower() in ['sus', 'ccd'] or first_part.startswith(('CNDA', 'NDAR', 'TUAB', 'TUEV')):
            return first_part
        # Otherwise return first two parts if available
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return parts[0]
    
    return "unknown"

def check_npy_files(directory, expected_dim=(2, 60, 240)):
    """
    Check all .npy files in directory and report dimension mismatches.
    
    Args:
        directory: Root directory to search
        expected_dim: Expected dimension tuple (default: (2, 60, 240))
    """
    directory = Path(directory)
    all_files = list(directory.rglob("*.npy"))
    
    print(f"Found {len(all_files)} .npy files")
    print(f"Expected dimension: {expected_dim}")
    print("="*80)
    
    # Group files by prefix
    files_by_prefix = defaultdict(list)
    dimension_issues = []
    
    print("Scanning files...")
    for npy_file in all_files:
        try:
            data = np.load(npy_file)
            shape = data.shape
            prefix = extract_prefix(str(npy_file))
            files_by_prefix[prefix].append((str(npy_file), shape))
            
            # Check if dimension matches expected
            if shape != expected_dim:
                dimension_issues.append((str(npy_file), shape, prefix))
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")
            dimension_issues.append((str(npy_file), f"ERROR: {e}", "unknown"))
    
    # Report dimension issues
    print(f"\n{'='*80}")
    print("FILES WITH UNEXPECTED DIMENSIONS:")
    print(f"{'='*80}")
    
    if dimension_issues:
        # Group issues by prefix
        issues_by_prefix = defaultdict(list)
        for filepath, shape, prefix in dimension_issues:
            issues_by_prefix[prefix].append((filepath, shape))
        
        for prefix, issues in sorted(issues_by_prefix.items()):
            print(f"\nPrefix: {prefix}")
            print(f"  Found {len(issues)} files with unexpected dimensions:")
            for filepath, shape in issues[:10]:  # Show first 10
                print(f"    {os.path.basename(filepath)}: {shape}")
            if len(issues) > 10:
                print(f"    ... and {len(issues) - 10} more files")
    else:
        print("No files with unexpected dimensions found!")
    
    # Report dimension distribution by prefix
    print(f"\n{'='*80}")
    print("DIMENSION DISTRIBUTION BY PREFIX:")
    print(f"{'='*80}")
    
    for prefix in sorted(files_by_prefix.keys()):
        files = files_by_prefix[prefix]
        dimensions = {}
        for _, shape in files:
            dimensions[shape] = dimensions.get(shape, 0) + 1
        
        print(f"\nPrefix: {prefix}")
        print(f"  Total files: {len(files)}")
        print(f"  Dimension distribution:")
        for dim, count in sorted(dimensions.items()):
            percentage = (count / len(files)) * 100
            marker = " ⚠️" if dim != expected_dim else ""
            print(f"    {dim}: {count} files ({percentage:.1f}%){marker}")
        
        # Check if all files have same dimension
        unique_dims = set(dim for _, dim in files)
        if len(unique_dims) > 1:
            print(f"  ⚠️  WARNING: Multiple dimensions found for this prefix!")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"{'='*80}")
    print(f"Total files checked: {len(all_files)}")
    print(f"Files with unexpected dimensions: {len(dimension_issues)}")
    print(f"Unique prefixes found: {len(files_by_prefix)}")
    
    return dimension_issues, files_by_prefix

def analyze_tasks_per_participant(directory):
    """
    Analyze how many passive (SUS) and active (CCD) tasks each participant has.
    Tasks are counted based on the first dimension of the .npy file:
    - (n, 60, 240) -> n tasks
    - (2, 60, 240) -> 2 tasks
    
    Only participants with demographics CSV files are included.
    
    Args:
        directory: Root directory to search for .npy files
        
    Returns:
        Dictionary with participant_id -> {'passive': int, 'active': int}
    """
    directory = Path(directory)
    all_files = list(directory.rglob("*.npy"))
    
    # First, find all participants who have demographics CSV files
    print(f"\n{'='*80}")
    print("ANALYZING TASKS PER PARTICIPANT")
    print(f"{'='*80}")
    print("Finding participants with demographics CSV files...")
    
    participants_with_demographics = set()
    for csv_file in directory.rglob("*demographics*.csv"):
        # Extract participant ID from path
        parts = csv_file.parts
        for part in parts:
            if part.startswith('sub-'):
                participants_with_demographics.add(part)
                break
    
    print(f"Found {len(participants_with_demographics)} participants with demographics CSV files")
    
    # Group files by participant and count tasks based on dimensions
    participant_tasks = defaultdict(lambda: {'passive': 0, 'active': 0})
    files_with_issues = []
    
    print(f"Scanning {len(all_files)} .npy files...")
    
    for npy_file in all_files:
        # Extract participant ID from path (e.g., sub-NDARAC904DMU)
        parts = npy_file.parts
        participant_id = None
        for part in parts:
            if part.startswith('sub-'):
                participant_id = part
                break
        
        if participant_id is None:
            continue
        
        # Only process participants with demographics CSV files
        if participant_id not in participants_with_demographics:
            continue
        
        try:
            # Load file to check dimensions
            data = np.load(npy_file)
            shape = data.shape
            
            # Count tasks based on first dimension
            # Expected shapes: (n, 60, 240) or (n, 60, 200)
            # If shape is (n, 60, 240), count as n tasks
            # If shape is (n, 60, 200), count as n tasks
            if len(shape) == 3:
                num_tasks = shape[0]
            else:
                # Unexpected shape, skip or count as 1
                num_tasks = 1
                files_with_issues.append((str(npy_file), shape))
            
            # Determine task type from filename
            filename = npy_file.name.lower()
            if filename.startswith('sus'):
                participant_tasks[participant_id]['passive'] += num_tasks
            elif filename.startswith('ccd'):
                participant_tasks[participant_id]['active'] += num_tasks
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")
            files_with_issues.append((str(npy_file), f"ERROR: {e}"))
    
    if files_with_issues:
        print(f"\nWarning: {len(files_with_issues)} files had issues (unexpected shapes or load errors)")
        if len(files_with_issues) <= 20:
            for filepath, issue in files_with_issues[:20]:
                print(f"  {os.path.basename(filepath)}: {issue}")
    
    print(f"Processed {len(participant_tasks)} participants (with demographics CSV files)")
    
    # Extract counts for visualization
    passive_counts = [info['passive'] for info in participant_tasks.values()]
    active_counts = [info['active'] for info in participant_tasks.values()]
    total_counts = [info['passive'] + info['active'] for info in participant_tasks.values()]
    
    # Print statistics
    print(f"\n{'='*80}")
    print("TASK COUNT STATISTICS")
    print(f"{'='*80}")
    print(f"\nPassive Tasks (SUS):")
    print(f"  Mean: {np.mean(passive_counts):.2f}")
    print(f"  Median: {np.median(passive_counts):.2f}")
    print(f"  Min: {np.min(passive_counts)}")
    print(f"  Max: {np.max(passive_counts)}")
    print(f"  Std: {np.std(passive_counts):.2f}")
    
    print(f"\nActive Tasks (CCD):")
    print(f"  Mean: {np.mean(active_counts):.2f}")
    print(f"  Median: {np.median(active_counts):.2f}")
    print(f"  Min: {np.min(active_counts)}")
    print(f"  Max: {np.max(active_counts)}")
    print(f"  Std: {np.std(active_counts):.2f}")
    
    print(f"\nTotal Tasks:")
    print(f"  Mean: {np.mean(total_counts):.2f}")
    print(f"  Median: {np.median(total_counts):.2f}")
    print(f"  Min: {np.min(total_counts)}")
    print(f"  Max: {np.max(total_counts)}")
    print(f"  Std: {np.std(total_counts):.2f}")
    
    # Count distribution
    passive_dist = defaultdict(int)
    active_dist = defaultdict(int)
    total_dist = defaultdict(int)
    
    for info in participant_tasks.values():
        passive_dist[info['passive']] += 1
        active_dist[info['active']] += 1
        total_dist[info['passive'] + info['active']] += 1
    
    print(f"\nPassive Task Count Distribution:")
    for count in sorted(passive_dist.keys()):
        print(f"  {count} tasks: {passive_dist[count]} participants")
    
    print(f"\nActive Task Count Distribution:")
    for count in sorted(active_dist.keys()):
        print(f"  {count} tasks: {active_dist[count]} participants")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histogram 1: Passive tasks per participant
    ax1 = axes[0, 0]
    ax1.hist(passive_counts, bins=min(50, len(set(passive_counts))), edgecolor='black', alpha=0.7, color='lightblue')
    ax1.set_xlabel('Number of Passive Tasks (SUS)', fontsize=12)
    ax1.set_ylabel('Number of Participants', fontsize=12)
    ax1.set_title('Distribution of Passive Tasks per Participant', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axvline(np.mean(passive_counts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(passive_counts):.1f}')
    ax1.axvline(np.median(passive_counts), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(passive_counts):.1f}')
    ax1.legend()
    
    # Histogram 2: Active tasks per participant
    ax2 = axes[0, 1]
    ax2.hist(active_counts, bins=min(50, len(set(active_counts))), edgecolor='black', alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Number of Active Tasks (CCD)', fontsize=12)
    ax2.set_ylabel('Number of Participants', fontsize=12)
    ax2.set_title('Distribution of Active Tasks per Participant', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axvline(np.mean(active_counts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(active_counts):.1f}')
    ax2.axvline(np.median(active_counts), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(active_counts):.1f}')
    ax2.legend()
    
    # Histogram 3: Total tasks per participant
    ax3 = axes[1, 0]
    ax3.hist(total_counts, bins=min(50, len(set(total_counts))), edgecolor='black', alpha=0.7, color='lightgreen')
    ax3.set_xlabel('Total Number of Tasks', fontsize=12)
    ax3.set_ylabel('Number of Participants', fontsize=12)
    ax3.set_title('Distribution of Total Tasks per Participant', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.axvline(np.mean(total_counts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(total_counts):.1f}')
    ax3.axvline(np.median(total_counts), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(total_counts):.1f}')
    ax3.legend()
    
    # Scatter plot: Active vs Passive
    ax4 = axes[1, 1]
    scatter = ax4.scatter(passive_counts, active_counts, alpha=0.5, s=50, c=total_counts, cmap='viridis')
    ax4.set_xlabel('Number of Passive Tasks (SUS)', fontsize=12)
    ax4.set_ylabel('Number of Active Tasks (CCD)', fontsize=12)
    ax4.set_title('Active vs Passive Tasks per Participant', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Total Tasks')
    
    plt.tight_layout()
    
    # Save figure
    output_file = "task_distribution_per_participant.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    # Save detailed data to CSV
    csv_file = "participant_task_counts.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['participant_id', 'passive_tasks', 'active_tasks', 'total_tasks'])
        for participant_id, info in sorted(participant_tasks.items()):
            writer.writerow([participant_id, info['passive'], info['active'], info['passive'] + info['active']])
    print(f"Detailed data saved to: {csv_file}")
    
    return participant_tasks

if __name__ == "__main__":
    directory = "/home/mojtabam/projects/aip-aghodsib/mojtabam/preprocessed_new"
    expected_dim = (2, 60, 240)
    
    print("Checking .npy file dimensions...")
    print(f"Directory: {directory}")
    print(f"Expected dimension: {expected_dim}")
    print()
    
    # Check dimensions
    issues, grouped = check_npy_files(directory, expected_dim)
    
    # Save dimension results to file
    output_file = "npy_dimension_issues.txt"
    with open(output_file, 'w') as f:
        f.write("FILES WITH UNEXPECTED DIMENSIONS:\n")
        f.write("="*80 + "\n\n")
        if issues:
            for filepath, shape, prefix in issues:
                f.write(f"{filepath}\t{shape}\t{prefix}\n")
        else:
            f.write("No issues found!\n")
    
    print(f"\nDimension analysis results saved to: {output_file}")
    
    # Analyze tasks per participant
    participant_tasks = analyze_tasks_per_participant(directory)

