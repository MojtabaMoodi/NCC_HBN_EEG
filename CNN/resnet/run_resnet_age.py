#!/usr/bin/env python3
"""
Script to run ResNet age prediction experiments (classification or regression).
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


def _try_line_buffer_stdio() -> None:
    """Use line-buffered stdout/stderr when supported (e.g. piped to tee) so output appears promptly."""
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(line_buffering=True)
            except (OSError, ValueError, AttributeError, TypeError):
                pass


def main():
    """Run ResNet age classification (3-class) or age regression (normalized continuous age)."""
    _try_line_buffer_stdio()
    parser = argparse.ArgumentParser(
        description='ResNet age prediction: train or evaluate (--eval_only) for age classification (3 classes) '
        'or age regression (continuous), with separate metrics for active vs passive tasks.'
    )
    parser.add_argument('--mode', choices=['1s', '2s', '4s'], default='1s',
                       help='EEG segment length: 1s (1-second segments), 2s (2-second segments), or 4s (4-second segments)')
    parser.add_argument('--resnet_type', type=int, choices=[18, 34, 50], default=18,
                       help='ResNet architecture type: 18, 34, or 50 (default: 18)')
    parser.add_argument(
        '--prediction-type',
        dest='prediction_type',
        choices=['classification', 'regression'],
        default='classification',
        help="Learning target: 'classification' (3 age bins) or 'regression' (normalized age in [0,1], same as CNN).",
    )
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
                       help='Path to the saved best model. Use with --eval_only. If not set, uses results_dir/checkpoints/<experiment_name>_best.pth.')
    
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
    
    resnet_task = 'age_regression' if args.prediction_type == 'regression' else 'age'
    model_type, model_config_dict = get_resnet_model_type(args.resnet_type, task=resnet_task)
    
    print("="*80)
    if args.prediction_type == 'classification':
        print(f"RESNET{args.resnet_type} AGE CLASSIFICATION EXPERIMENT")
    else:
        print(f"RESNET{args.resnet_type} AGE REGRESSION EXPERIMENT")
    print("="*80)
    print(f"Model: ResNet{args.resnet_type} ({model_type})")
    print(f"Task: {'Age classification (3 classes)' if args.prediction_type == 'classification' else 'Age regression (normalized age)'}")
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
        target_key="age",
        prediction_type=args.prediction_type,
    )
    if args.no_stratified:
        training_config_kw["use_stratified_train_batches"] = False
        print("Stratified train batches disabled (--no_stratified); using class weights in loss.")
    
    num_workers = base_data_config.num_workers
    print(f"Number of workers: {num_workers}")
    print("="*80)
    
    if args.prediction_type == 'classification':
        experiment_name = f"age_resnet{args.resnet_type}_{args.mode}"
        description = (
            f"ResNet{args.resnet_type} age classification with {args.mode} segments "
            "(train on both, evaluate separately on active and passive)"
        )
    else:
        experiment_name = f"age_regression_resnet{args.resnet_type}_{args.mode}"
        description = (
            f"ResNet{args.resnet_type} age regression with {args.mode} segments "
            "(train on both, evaluate separately on active and passive)"
        )
    
    model_config = ModelConfig(num_channels=60, **model_config_dict)
    if args.prediction_type == 'regression':
        model_config.num_classes = 1
    
    training_config = TrainingConfig(**training_config_kw)
    if args.aggregate_by_participant:
        training_config.aggregate_by_participant = args.aggregate_by_participant
        print(f"Evaluation will use participant-level aggregation: {args.aggregate_by_participant}")
    
    experiment = ExperimentConfig(
        name=experiment_name,
        model_type=model_type,
        target_type="age",
        description=description,
        data_config=base_data_config,
        model_config=model_config,
        training_config=training_config,
        checkpoint_path=args.checkpoint,
    )
    
    print(f"\nStarting ResNet{args.resnet_type} age {args.prediction_type} experiment...")
    print(f"Experiment name: {experiment.name}")
    print(f"Model type: {model_type}")
    print(f"Training: Both active and passive tasks")
    print(f"Evaluation: Separate metrics for active and passive tasks")
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
            print("\nOverall test set metrics:")
            m = results[0].metrics
            if args.prediction_type == 'regression':
                if 'mae' in m:
                    print(f"  MAE: {m['mae']:.4f} years")
                if 'rmse' in m:
                    print(f"  RMSE: {m['rmse']:.4f} years")
                if 'r2' in m:
                    print(f"  R²: {m['r2']:.4f}")
            else:
                if 'accuracy' in m:
                    print(f"  Accuracy: {m['accuracy']:.4f}")
                if 'f1_weighted' in m:
                    print(f"  F1-Score (weighted): {m['f1_weighted']:.4f}")
            
            if 'task_type_metrics' in m:
                print("\nMetrics by task type (active vs passive):")
                task_metrics = m['task_type_metrics']
                for split_name in ('active', 'passive'):
                    if split_name not in task_metrics:
                        continue
                    tm = task_metrics[split_name]
                    label = split_name.upper()
                    if args.prediction_type == 'regression':
                        mae_v = tm.get('mae')
                        rmse_v = tm.get('rmse')
                        r2_v = tm.get('r2')
                        mae_s = f"{mae_v:.4f} years" if isinstance(mae_v, (int, float)) else "N/A"
                        rmse_s = f"{rmse_v:.4f} years" if isinstance(rmse_v, (int, float)) else "N/A"
                        r2_s = f"{r2_v:.4f}" if isinstance(r2_v, (int, float)) else "N/A"
                        print(f"  {label}: MAE = {mae_s}, RMSE = {rmse_s}, R² = {r2_s}")
                    else:
                        acc = tm.get('accuracy')
                        f1w = tm.get('f1_weighted')
                        acc_s = f"{acc:.4f}" if isinstance(acc, (int, float)) else "N/A"
                        f1_s = f"{f1w:.4f}" if isinstance(f1w, (int, float)) else "N/A"
                        print(f"  {label}: Accuracy = {acc_s}, F1 (weighted) = {f1_s}")
    else:
        print("\n" + "="*80)
        print("EXPERIMENT FAILED!")
        print("="*80)
        print(f"Error: {results[0].error}")
    
    print("="*80)


if __name__ == "__main__":
    main()
