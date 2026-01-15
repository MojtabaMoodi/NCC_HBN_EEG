"""
Main Script for User Identification Experiments
Supports both CNN and LaBraM models for user identification tasks.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for absolute imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import from CNN module using absolute imports
from CNN.config import SystemConfig, ModelConfig, TrainingConfig, ExperimentConfig
from CNN.experiment import run_experiments
from CNN.utils import create_data_config_for_segment_length
from CNN.gpu_utils import detect_available_gpus


def create_user_identification_experiments(segment_length: str, epochs: int, 
                                          learning_rate: float, 
                                          batch_size: int,
                                          num_gpus: int,
                                          random_seed: int,
                                          hdf5_dir: str,
                                          model_type: str = "both",
                                          labram_checkpoint: str = None):
    """
    Create user identification experiments for a specific segment length.
    
    Args:
        segment_length: '1s', '2s', or '4s'
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size (None means use default based on segment length)
        num_gpus: Number of GPUs to use
        random_seed: Random seed for reproducibility
        hdf5_dir: Directory containing HDF5 files
        model_type: 'cnn', 'labram', or 'both' (default: 'both')
        labram_checkpoint: Path to pre-trained LaBraM checkpoint (optional)
        
    Returns:
        List of experiment configurations
        
    Raises:
        ValueError: If segment_length is invalid
        FileNotFoundError: If hdf5_dir does not exist
    """
    if segment_length not in ['1s', '2s', '4s']:
        raise ValueError(f"Invalid segment_length: {segment_length}. Must be '1s', '2s', or '4s'")
    
    if model_type not in ['cnn', 'labram', 'both']:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'cnn', 'labram', or 'both'")
    
    hdf5_path = Path(hdf5_dir)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 directory does not exist: {hdf5_dir}")
    if not hdf5_path.is_dir():
        raise ValueError(f"HDF5 path is not a directory: {hdf5_dir}")
    
    experiments = []

    # Create base data config for this segment length
    base_data_config = create_data_config_for_segment_length(
        segment_length, batch_size, random_seed=random_seed, num_gpus=num_gpus
    )
    # Set task_type to "both" to train on both active and passive tasks
    base_data_config.task_type = "both"
    # Override hdf5_dir with provided path
    base_data_config.hdf5_dir = hdf5_dir
    
    # User identification training config
    training_config = TrainingConfig(
        epochs=epochs, 
        learning_rate=learning_rate, 
        target_key="user_identification",
        weight_decay=1e-4,  # L2 regularization for user identification
        scheduler_patience=3,  # More aggressive LR reduction for user identification
        scheduler_factor=0.5  # Standard LR reduction factor
    )
    
    # Create CNN experiment (if requested)
    if model_type in ['cnn', 'both']:
        experiments.append(ExperimentConfig(
            name=f"user_identification_{segment_length}_cnn",
            model_type="user_identification_cnn",
            target_type="user_identification",
            description=f"User identification with {segment_length} segments using CNN (train on both, evaluate separately)",
            data_config=base_data_config,
            model_config=ModelConfig(num_channels=60, dropout_rate=0.65),  # Increased dropout for regularization
            training_config=training_config
        ))
    
    # Create LaBraM experiment (if requested and available)
    if model_type in ['labram', 'both']:
        try:
            from CNN.models.model_factory import ModelFactory
            if 'user_identification_labram' in ModelFactory.get_available_models():
                # Store checkpoint path in model_config for later use
                model_config = ModelConfig(num_channels=60, dropout_rate=0.1)  # Lower dropout for LaBraM
                if labram_checkpoint:
                    # Store checkpoint path as an attribute (will be used during model creation)
                    model_config.labram_checkpoint = labram_checkpoint
                    print(f"✅ Will load pre-trained LaBraM weights from: {labram_checkpoint}")
                else:
                    print("⚠️  WARNING: No pre-trained checkpoint provided. LaBraM will be trained from scratch.")
                
                experiments.append(ExperimentConfig(
                    name=f"user_identification_{segment_length}_labram",
                    model_type="user_identification_labram",
                    target_type="user_identification",
                    description=f"User identification with {segment_length} segments using LaBraM (train on both, evaluate separately)",
                    data_config=base_data_config,
                    model_config=model_config,
                    training_config=training_config
                ))
            else:
                print("⚠️  WARNING: LaBraM model not available. Skipping LaBraM experiment.")
        except ImportError as e:
            print(f"⚠️  WARNING: LaBraM not available: {e}. Skipping LaBraM experiment.")

    return experiments


def _validate_and_create_directories(results_dir: str, reports_dir: str) -> None:
    """Validate and create output directories."""
    results_path = Path(results_dir)
    reports_path = Path(reports_dir)
    
    try:
        results_path.mkdir(parents=True, exist_ok=True)
        reports_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create output directories: {e}") from e


def _validate_gpu_config(num_gpus: int) -> None:
    """Validate GPU configuration."""
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
    
    if num_gpus > 1:
        available_gpus = detect_available_gpus()
        if available_gpus == 0:
            raise RuntimeError(
                f"Requested {num_gpus} GPUs but CUDA is not available. "
                "Set num_gpus=1 to use CPU, or ensure CUDA is properly configured."
            )
        if num_gpus > available_gpus:
            raise RuntimeError(
                f"Requested {num_gpus} GPUs but only {available_gpus} available. "
                f"Please set num_gpus={available_gpus} or lower."
            )


def main():
    """Main function for running user identification experiments."""
    parser = argparse.ArgumentParser(
        description='User Identification Experiment Runner (CNN and LaBraM)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both CNN and LaBraM:
  python main.py --mode 4s --model both --labram_checkpoint /path/to/checkpoint.pth
  
  # Run only CNN:
  python main.py --mode 4s --model cnn
  
  # Run only LaBraM:
  python main.py --mode 4s --model labram --labram_checkpoint /path/to/checkpoint.pth
        """
    )
    parser.add_argument('--mode', choices=['1s', '2s', '4s'], default='4s',
                       help='EEG segment length: 1s (1-second segments), 2s (2-second segments), or 4s (4-second segments)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, 
                       help='Learning rate (default: 0.0001 for user identification)')
    parser.add_argument('--batch_size', type=int, default=None, 
                       help='Batch size (if None, uses default: 256 for 1s, 192 for 2s, 128 for 4s)')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_gpus', type=int, default=None, 
                       help='Number of GPUs to use for DataParallel (default: auto-detect)')
    parser.add_argument('--results_dir', type=str, default='results_user_id', 
                       help='Directory to save results')
    parser.add_argument('--reports_dir', type=str, default='reports_user_id', 
                       help='Directory to save reports')
    parser.add_argument('--hdf5_dir', type=str, 
                       default='/home/mojtabam/scratch/processed_eeg_data_user_identification',
                       help='Directory containing HDF5 files for user identification')
    parser.add_argument('--model', choices=['cnn', 'labram', 'both'], default='both',
                       help='Model type to use: cnn, labram, or both (default: both)')
    parser.add_argument('--labram_checkpoint', type=str, default=None,
                       help='Path to pre-trained LaBraM checkpoint (required for LaBraM, recommended)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {args.epochs}")
    if args.learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {args.learning_rate}")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {args.batch_size}")
    
    # Warn if LaBraM requested without checkpoint
    if args.model in ['labram', 'both'] and not args.labram_checkpoint:
        print("⚠️  WARNING: LaBraM requested but no checkpoint provided.")
        print("   LaBraM will be trained from scratch (not recommended).")
        print("   Use --labram_checkpoint to provide pre-trained weights.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return 1
    
    # Determine number of GPUs to use
    if args.num_gpus is None:
        actual_num_gpus = detect_available_gpus()
        if actual_num_gpus == 0:
            actual_num_gpus = 1  # Fallback to CPU
    else:
        actual_num_gpus = args.num_gpus
    
    # Validate GPU configuration
    _validate_gpu_config(actual_num_gpus)
    
    # Validate and create output directories
    _validate_and_create_directories(args.results_dir, args.reports_dir)
    
    # Create system configuration
    system_config = SystemConfig(
        results_dir=args.results_dir,
        reports_dir=args.reports_dir,
        num_gpus=actual_num_gpus,
        device='auto'
    )
    
    print("="*80)
    print("USER IDENTIFICATION EXPERIMENT RUNNER")
    print("="*80)
    print(f"Mode: {args.mode} (EEG segment length)")
    print(f"Model type: {args.model}")
    print(f"HDF5 directory: {args.hdf5_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Reports directory: {args.reports_dir}")
    print(f"Epochs: {args.epochs}, Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size if args.batch_size is not None else 'auto-selected'}, Random seed: {args.random_seed}")
    print(f"Number of GPUs: {actual_num_gpus}")
    if args.labram_checkpoint:
        print(f"LaBraM checkpoint: {args.labram_checkpoint}")
    print("="*80)
    
    # Create experiments for the specified segment length
    print(f"Creating user identification experiments for {args.mode} EEG segments...")
    
    try:
        experiments = create_user_identification_experiments(
            segment_length=args.mode,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_gpus=actual_num_gpus,
            random_seed=args.random_seed,
            hdf5_dir=args.hdf5_dir,
            model_type=args.model,
            labram_checkpoint=args.labram_checkpoint
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: Failed to create experiments: {e}", file=sys.stderr)
        return 1
    
    if not experiments:
        print("ERROR: No experiments were created.", file=sys.stderr)
        return 1
    
    # Display configuration from the first experiment
    print(f"Number of workers: {experiments[0].data_config.num_workers}")
    print(f"Batch size: {experiments[0].data_config.batch_size}")
    print(f"\n⚠️  NOTE: User identification with ~3000 classes is a challenging task.")
    print(f"   - The model will have many parameters in the final classification layer")
    print(f"   - Consider using more epochs and potentially adjusting learning rate")
    print(f"   - Training may take longer due to the large number of classes")
    print("="*80)
    
    print(f"Running {len(experiments)} experiment(s) for {args.mode} segments...")
    
    # Run all experiments
    try:
        results = run_experiments(experiments, system_config, args.random_seed, generate_reports=True)
    except Exception as e:
        print(f"ERROR: Failed to run experiments: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    # Check if any experiments failed
    if results:
        failed_experiments = [r for r in results if not r.success]
        if failed_experiments:
            print(f"\nWARNING: {len(failed_experiments)} experiment(s) failed:", file=sys.stderr)
            for result in failed_experiments:
                print(f"  - {result.experiment_name}: {result.error}", file=sys.stderr)
            return 1
    
    print(f"All {args.mode} user identification experiments completed successfully!")
    print(f"Results saved to: {args.results_dir}")
    print(f"Reports generated in: {args.reports_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
