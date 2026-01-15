# User Identification Module

This module provides tools for user identification experiments using both CNN and LaBraM models.

## Structure

- `main.py`: Main script for running user identification experiments (supports both CNN and LaBraM)
- `validate.py`: Validation script to verify user identification setup
- `labram_model.py`: LaBraM wrapper model for user identification
- `run_4s.sh`: Bash script to run 4s segment experiments
- `run_4s.slurm`: SLURM script to run 4s segment experiments on cluster

## Usage

### Running Experiments

```bash
# Run both CNN and LaBraM (recommended for comparison)
python user_identification/main.py \
    --mode 4s \
    --model both \
    --labram_checkpoint /path/to/pretrained_labram.pth \
    --epochs 50 \
    --learning_rate 0.0001 \
    --num_gpus 4

# Run only CNN
python user_identification/main.py \
    --mode 4s \
    --model cnn \
    --epochs 50 \
    --learning_rate 0.0001

# Run only LaBraM
python user_identification/main.py \
    --mode 4s \
    --model labram \
    --labram_checkpoint /path/to/pretrained_labram.pth \
    --epochs 50 \
    --learning_rate 0.0001
```

### Validating Setup

```bash
python user_identification/validate.py \
    --hdf5_dir /path/to/processed_eeg_data_user_identification
```

## Arguments

- `--mode`: EEG segment length ('1s', '2s', or '4s')
- `--model`: Model type ('cnn', 'labram', or 'both')
- `--labram_checkpoint`: Path to pre-trained LaBraM checkpoint (required for LaBraM)
- `--epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--batch_size`: Batch size (optional, auto-selected if not provided)
- `--num_gpus`: Number of GPUs to use
- `--hdf5_dir`: Directory containing HDF5 files
- `--results_dir`: Directory to save results
- `--reports_dir`: Directory to save reports

## Notes

- LaBraM requires pre-trained weights for best performance. The checkpoint should be a pre-trained LaBraM model.
- CNN model is trained from scratch.
- Both models use the same data preprocessing and evaluation pipeline.
- Results are saved separately for each model type.
