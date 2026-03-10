#!/usr/bin/env python3
"""
Script to run ResNet gender classification experiment.
Supports ResNet18, ResNet34, and ResNet50 architectures.
Trains on both active and passive tasks, evaluates separately on active and passive.
Uses the same multipart train/val/test HDF5 data pipeline as the CNN (EEGDataLoader.create_train_val_test_loaders).
"""

import sys
from pathlib import Path

# Ensure project root (EEG) is on path so "CNN" and "data_processing" resolve when run from any CWD
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
from CNN.config import SystemConfig, ModelConfig, TrainingConfig, ExperimentConfig
from CNN.experiment import run_experiments
from CNN.utils import create_data_config_for_segment_length, get_resnet_model_type
from CNN.gpu_utils import detect_available_gpus


def main():
    """Main function for running ResNet gender classification experiment (gender prediction)."""
    parser = argparse.ArgumentParser(
        description='ResNet gender prediction: train or evaluate (--eval_only) a ResNet on EEG for gender classification (2 classes).'
    )
    parser.add_argument('--mode', choices=['1s', '2s', '4s'], default='1s',
                       help='EEG segment length: 1s (1-second segments), 2s (2-second segments), or 4s (4-second segments)')
    parser.add_argument('--resnet_type', type=int, choices=[18, 34, 50], default=18,
                       help='ResNet architecture type: 18, 34, or 50 (default: 18)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=None, 
                       help='Batch size (if None, uses default: 512 for 1s, 192 for 2s, 128 for 4s)')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_gpus', type=int, default=None,
                       help='Number of GPUs for DataParallel (default: auto-detect, use all available)')
    parser.add_argument('--results_dir', type=str, default='experiment_results', 
                       help='Directory to save results')
    parser.add_argument('--reports_dir', type=str, default='reports', 
                       help='Directory to save reports')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Override HDF5 data directory (train/val/test multipart or single files). If not set, uses CNN.utils.DATA_PATHS for the segment length.')
    parser.add_argument('--no_stratified', action='store_true',
                       help='Disable stratified train batches (avoids slow metadata scan on large 4s HDF5; uses class weights in loss instead).')
    parser.add_argument('--eval_only', action='store_true',
                       help='Skip training; load saved checkpoint and run evaluation only (e.g. with majority vote).')
    parser.add_argument('--aggregate_by_participant', type=str, default=None, metavar='METHOD',
                       help="Participant-level aggregation: 'majority_vote' for confidence-weighted majority. Use with --eval_only to evaluate with majority vote.")
    parser.add_argument('--checkpoint', type=str, default=None, metavar='PATH',
                       help='Path to the saved best model (e.g. .../gender_resnet34_2s_best.pth). Use with --eval_only. If not set, uses results_dir/checkpoints/<experiment_name>_best.pth.')
    
    args = parser.parse_args()
    
    if args.num_gpus is None:
        actual_num_gpus = detect_available_gpus()
    else:
        actual_num_gpus = args.num_gpus
    if actual_num_gpus < 1:
        actual_num_gpus = 1  # Use 1 (CPU) if no GPUs detected
    
    # Create system configuration
    system_config = SystemConfig(
        results_dir=args.results_dir,
        reports_dir=args.reports_dir,
        num_gpus=actual_num_gpus
    )
    
    # Determine actual batch size (auto-select if None)
    if args.batch_size is not None:
        actual_batch_size = args.batch_size
    else:
        if args.mode == '1s':
            actual_batch_size = 512  # Default for 1s
        elif args.mode == '2s':
            actual_batch_size = 192
        else:  # 4s
            actual_batch_size = 128
    
    # Get ResNet model type and configuration
    model_type, model_config_dict = get_resnet_model_type(args.resnet_type, task='gender')
    
    print("="*80)
    print(f"RESNET{args.resnet_type} GENDER CLASSIFICATION EXPERIMENT")
    print("="*80)
    print(f"Model: ResNet{args.resnet_type}")
    print(f"Task: Gender Classification (2 classes)")
    print(f"Mode: {args.mode} (EEG segment length)")
    print(f"Results directory: {args.results_dir}")
    print(f"Reports directory: {args.reports_dir}")
    print(f"Epochs: {args.epochs}, Learning rate: {args.learning_rate}")
    print(f"Batch size: {actual_batch_size} {'(auto-selected)' if args.batch_size is None else ''}, Random seed: {args.random_seed}")
    print(f"Number of GPUs: {actual_num_gpus}")
    
    # Create data config using shared utility function (use actual_batch_size so displayed and used batch size match)
    base_data_config = create_data_config_for_segment_length(
        args.mode, 
        batch_size=actual_batch_size, 
        random_seed=args.random_seed, 
        num_gpus=actual_num_gpus
    )
    # Set task_type to "both" to train on both active and passive tasks
    base_data_config.task_type = "both"
    if args.data_path is not None:
        base_data_config.hdf5_dir = args.data_path
        print(f"Using data path (override): {args.data_path}")
    
    training_config_kw = dict(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        target_key="gender",
        prediction_type="classification",
    )
    if args.no_stratified:
        training_config_kw["use_stratified_train_batches"] = False
        print("Stratified train batches disabled (--no_stratified); using class weights in loss.")
    
    num_workers = base_data_config.num_workers
    print(f"Number of workers: {num_workers}")
    print("="*80)
    
    # Create ResNet gender classification experiment
    # Merge model_config_dict with ModelConfig
    model_config = ModelConfig(num_channels=60, **model_config_dict)
    
    training_config = TrainingConfig(**training_config_kw)
    if args.aggregate_by_participant:
        training_config.aggregate_by_participant = args.aggregate_by_participant
        print(f"Evaluation will use participant-level aggregation: {args.aggregate_by_participant}")
    
    experiment = ExperimentConfig(
        name=f"gender_resnet{args.resnet_type}_{args.mode}",
        model_type=model_type,
        target_type="gender",
        description=f"ResNet{args.resnet_type} gender classification with {args.mode} segments (train on both, evaluate separately on active and passive)",
        data_config=base_data_config,
        model_config=model_config,
        training_config=training_config,
        checkpoint_path=args.checkpoint,
    )
    
    print(f"\nStarting ResNet{args.resnet_type} gender classification experiment...")
    print(f"Experiment name: {experiment.name}")
    print(f"Model type: {model_type}")
    print(f"Training: Both active and passive tasks")
    print(f"Evaluation: Separate accuracy reports for active and passive tasks")
    print("-" * 80)
    
    # Run the experiment
    results = run_experiments(
        [experiment], system_config, args.random_seed,
        generate_reports=True, eval_only=args.eval_only
    )
    
    if results[0].success:
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Results saved to: {args.results_dir}/{experiment.name}/")
        print(f"Reports generated in: {args.reports_dir}/")
        
        # Print key metrics
        if results[0].metrics:
            print("\nOverall Test Set Metrics:")
            if 'accuracy' in results[0].metrics:
                print(f"  Accuracy: {results[0].metrics['accuracy']:.4f}")
            if 'f1_weighted' in results[0].metrics:
                print(f"  F1-Score (weighted): {results[0].metrics['f1_weighted']:.4f}")
            
            # Print task-specific metrics if available
            if 'task_type_metrics' in results[0].metrics:
                print("\nTask-Specific Metrics:")
                task_metrics = results[0].metrics['task_type_metrics']
                if 'active' in task_metrics:
                    print(f"  Active Tasks:")
                    print(f"    Accuracy: {task_metrics['active'].get('accuracy', 'N/A'):.4f}")
                    print(f"    F1-Score: {task_metrics['active'].get('f1_weighted', 'N/A'):.4f}")
                if 'passive' in task_metrics:
                    print(f"  Passive Tasks:")
                    print(f"    Accuracy: {task_metrics['passive'].get('accuracy', 'N/A'):.4f}")
                    print(f"    F1-Score: {task_metrics['passive'].get('f1_weighted', 'N/A'):.4f}")
    else:
        print("\n" + "="*80)
        print("EXPERIMENT FAILED!")
        print("="*80)
        print(f"Error: {results[0].error}")
    
    print("="*80)


if __name__ == "__main__":
    main()

