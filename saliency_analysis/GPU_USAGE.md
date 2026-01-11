# GPU Usage Guide for Saliency Computation

## Quick Start

### Option 1: Using salloc (Interactive Session)

Request GPU resources interactively:

```bash
# Request 4 GPUs for 24 hours
salloc --gres=gpu:4 --time=24:00:00 --mem=64G --cpus-per-task=16

# Once allocated, activate your environment and run:
cd /home/mojtabam/projects/aip-aghodsib/mojtabam/EEG
conda activate eeg_env  # or your environment name
python3 saliency_analysis/main.py \
    --batch \
    --device cuda \
    --num_gpus 4 \
    --hdf5_dir data_processing/processed_eeg_data_hdf5_no_compression
```

### Option 2: Using sbatch (Batch Job)

Submit as a batch job:

```bash
sbatch saliency_analysis/run_saliency_slurm.sh
```

### Option 3: Using CPU (No GPU Required)

If you don't have GPU access or want to test on CPU:

```bash
python3 saliency_analysis/main.py \
    --batch \
    --device cpu \
    --num_gpus 1 \
    --hdf5_dir data_processing/processed_eeg_data_hdf5_no_compression \
    --max_samples 100  # Use fewer samples for CPU testing
```

## GPU Requirements

### Can I use CPU?
**Yes!** Saliency computation works on CPU, but it will be much slower:
- **GPU**: ~10-30 seconds per 100 samples
- **CPU**: ~5-15 minutes per 100 samples

For full dataset analysis, GPU is strongly recommended.

### Can I use 4 GPUs?
**Yes!** The script supports multi-GPU using PyTorch's DataParallel:
- Automatically distributes batches across GPUs
- Provides ~3-4x speedup compared to single GPU
- Works seamlessly with the existing code

### GPU Memory Requirements
- **Single GPU**: ~4-8 GB VRAM per model
- **4 GPUs**: ~2-4 GB VRAM per GPU (distributed)
- **Batch size**: Adjust `--batch_size` if you run out of memory

## SLURM salloc Options

### Basic GPU Request
```bash
# Request 1 GPU
salloc --gres=gpu:1 --time=4:00:00

# Request 4 GPUs
salloc --gres=gpu:4 --time=24:00:00 --mem=64G
```

### Recommended Settings
```bash
# For full saliency computation (all models, all samples)
salloc \
    --gres=gpu:4 \
    --time=24:00:00 \
    --mem=64G \
    --cpus-per-task=16 \
    --partition=gpu  # Adjust partition name as needed
```

### Quick Test (Fewer Resources)
```bash
# For testing with limited samples
salloc \
    --gres=gpu:1 \
    --time=1:00:00 \
    --mem=16G \
    --cpus-per-task=4
```

## Command Line Options

### Device Selection
- `--device auto`: Auto-detect (prefers GPU if available)
- `--device cuda`: Force CUDA (fails if no GPU)
- `--device cpu`: Force CPU

### Multi-GPU Usage
- `--num_gpus 1`: Single GPU or CPU
- `--num_gpus 4`: Use 4 GPUs (DataParallel)
- Script automatically adjusts if fewer GPUs available

### Performance Tuning
- `--batch_size 32`: Default (adjust based on GPU memory)
- `--batch_size 64`: For larger GPUs
- `--batch_size 16`: If running out of memory

### Sample Limiting
- `--max_samples 1000`: Process 1000 samples per model (for testing)
- `--max_samples None`: Process all samples (default, remove flag)

## Example Workflows

### 1. Quick Test (1 GPU, 100 samples per model)
```bash
salloc --gres=gpu:1 --time=1:00:00 --mem=16G
python3 saliency_analysis/main.py \
    --batch \
    --device cuda \
    --num_gpus 1 \
    --max_samples 100 \
    --hdf5_dir data_processing/processed_eeg_data_hdf5_no_compression
```

### 2. Full Computation (4 GPUs, all samples)
```bash
salloc --gres=gpu:4 --time=24:00:00 --mem=64G --cpus-per-task=16
python3 saliency_analysis/main.py \
    --batch \
    --device cuda \
    --num_gpus 4 \
    --hdf5_dir data_processing/processed_eeg_data_hdf5_no_compression \
    --method integrated_gradients \
    --batch_size 32
```

### 3. Process Specific Models Only
```bash
python3 saliency_analysis/main.py \
    --batch \
    --device cuda \
    --num_gpus 4 \
    --models gender_baseline_1s age_classification_2s \
    --hdf5_dir data_processing/processed_eeg_data_hdf5_no_compression
```

### 4. CPU Fallback (No GPU Available)
```bash
python3 saliency_analysis/main.py \
    --batch \
    --device cpu \
    --num_gpus 1 \
    --max_samples 50 \
    --hdf5_dir data_processing/processed_eeg_data_hdf5_no_compression
```

### 5. Single Model (Original Functionality)
```bash
python3 saliency_analysis/main.py \
    --checkpoint CNN/report_2025-12-16/results_2s/checkpoints/gender_baseline_2s_best.pth \
    --target_type gender \
    --segment_length 2s \
    --device cuda \
    --num_gpus 1
```

## Monitoring GPU Usage

While running, monitor GPU usage in another terminal:
```bash
watch -n 1 nvidia-smi
```

## Troubleshooting

### "CUDA out of memory"
- Reduce `--batch_size` (try 16 or 8)
- Process fewer samples with `--max_samples`
- Use fewer GPUs (each GPU needs memory)

### "No GPUs available"
- Check `nvidia-smi` to see available GPUs
- Verify `CUDA_VISIBLE_DEVICES` is set correctly
- Try `--device cpu` as fallback

### "Requested 4 GPUs but only X available"
- Script automatically adjusts to available GPUs
- Or manually set `--num_gpus` to match available count

## Output

Results are saved to:
- `saliency_results/{model_name}/` - Per-model results
- Contains:
  - `saliency_results_{method}.npz` - Raw saliency data
  - Visualization plots (PNG files)
  - Channel rankings (CSV files)

## Estimated Time

For 8 models with full test sets:
- **4 GPUs**: ~2-4 hours
- **1 GPU**: ~8-12 hours  
- **CPU**: ~2-3 days (not recommended)

For testing with 1000 samples per model:
- **4 GPUs**: ~30-60 minutes
- **1 GPU**: ~2-3 hours
- **CPU**: ~4-6 hours

