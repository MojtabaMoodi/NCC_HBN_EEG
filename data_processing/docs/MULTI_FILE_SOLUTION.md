# Multi-File HDF5 Solution for 1s Segments

## Problem

For 1s segments with ~955K samples, storing all samples in a single HDF5 file creates a large B-tree index that causes exponential slowdown during data loading. Each sample is stored as a separate HDF5 dataset, and with ~955K datasets, the B-tree traversal becomes extremely slow.

## Solution

Split 1s segment data into multiple HDF5 files, with each file containing approximately 100K samples. This significantly reduces B-tree size per file and improves data loading performance.

## Implementation

### 1. Preprocessing Changes

**File Naming:**
- 1s segments: `eeg_data_train_1s_part0.h5`, `eeg_data_train_1s_part1.h5`, etc.
- 2s and 4s segments: Continue using single files (`eeg_data_train_2s.h5`, etc.)

**File Rotation:**
- When saving samples for 1s segments, automatically rotates to next file when `MAX_SAMPLES_PER_FILE` (100,000) is reached
- Tracks sample count per task_type (active/passive) separately
- Each file maintains the same structure: `active/` and `passive/` groups

**Key Methods:**
- `_initialize_hdf5_file_part()`: Initializes a part file
- `_get_current_hdf5_file_and_part()`: Determines which file to use based on sample count
- `_get_current_global_sample_idx_for_task()`: Counts samples for a specific task_type across all files
- `_save_samples_to_multiple_files()`: Handles file rotation when saving samples

### 2. Dataset Loading Changes

**MultiFileEEGDataset:**
- New dataset class that wraps multiple `EEGDataset` instances
- Concatenates samples from all files
- Supports shuffling across files
- Handles worker sharding for multiprocessing

**Automatic Detection:**
- `EEGDataLoader._get_hdf5_file_paths()` automatically detects multiple part files for 1s segments
- Falls back to single file if no part files found (backward compatibility)
- For 2s and 4s segments, continues using single files

### 3. Experiment Integration

**Experiment.py Updates:**
- Handles multiple files when evaluating by task type
- Uses `MultiFileEEGDataset` when multiple files are detected
- For age regression, uses first training file for age range computation

## Usage

### Preprocessing

The preprocessing automatically handles file splitting for 1s segments:

```bash
cd data_processing
python preprocess_user_identification.py
```

This will create:
- `eeg_data_train_1s_part0.h5`, `eeg_data_train_1s_part1.h5`, ... (multiple files)
- `eeg_data_val_1s_part0.h5`, `eeg_data_val_1s_part1.h5`, ... (multiple files)
- `eeg_data_test_1s_part0.h5`, `eeg_data_test_1s_part1.h5`, ... (multiple files)

### Training

No changes needed! The dataset loader automatically detects and uses multiple files:

```bash
cd CNN
python main_user_identification.py --mode 1s
```

## Performance Benefits

1. **Reduced B-tree size**: Each file has ~100K datasets instead of ~955K
2. **Faster key collection**: Collecting sample names is much faster per file
3. **Better cache efficiency**: Smaller files fit better in HDF5 cache
4. **Parallel loading potential**: Multiple files can be loaded in parallel (future optimization)

## Configuration

The maximum samples per file is configurable via `MAX_SAMPLES_PER_FILE` constant:

```python
MAX_SAMPLES_PER_FILE = 100000  # ~100K samples per file
```

Adjust this value based on your performance needs:
- Lower value = more files, smaller B-trees, faster loading
- Higher value = fewer files, larger B-trees, slower loading

## Backward Compatibility

- Existing single-file datasets continue to work
- The loader automatically detects whether files are single or multiple
- 2s and 4s segments continue using single files (no change needed)

## Technical Details

### File Structure

Each part file has the same structure as single files:
```
eeg_data_train_1s_part0.h5
├── active/
│   ├── sample_000000
│   ├── sample_000001
│   ├── ...
│   ├── metadata_000000/
│   └── metadata_000001/
└── passive/
    ├── sample_000000
    ├── ...
```

### Sample Indexing

- Global index: Tracks samples across all files (0, 1, 2, ..., 955K)
- Local index: Sample index within a file (0, 1, 2, ..., 100K)
- File rotation: `part_idx = global_idx // MAX_SAMPLES_PER_FILE`
- Local index: `local_idx = global_idx % MAX_SAMPLES_PER_FILE`

### Task Type Handling

Active and passive samples are tracked separately:
- Active samples: Counted across all files in `active/` groups
- Passive samples: Counted across all files in `passive/` groups
- File rotation happens independently for each task type

## Notes

- The first time you run preprocessing with this code, it will create multiple files
- Existing single-file datasets will continue to work (backward compatible)
- For best performance with 1s segments, use the multi-file format
- 2s and 4s segments don't need this optimization (fewer samples)
