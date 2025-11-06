"""
Systematic Main Script for EEG Classification Experiments
Provides a comprehensive interface for running multiple experiments and generating reports.
"""

import argparse
from config import SystemConfig, DataConfig, ModelConfig, TrainingConfig, ExperimentConfig
from experiment import run_experiments


# Data paths and segment lengths
DATA_PATHS = {
    '1s': "/home/mojtabam/projects/def-aghodsib/mojtabam/EEG/data_processing/processed_eeg_data_1s_segments_v2",
    '4s': "/home/mojtabam/projects/def-aghodsib/mojtabam/EEG/data_processing/processed_eeg_data_4s_segments_v2"
}

SEGMENT_LENGTHS = {
    '1s': 200,  # 1 second = 200 samples at 200Hz
    '4s': 800   # 4 seconds = 800 samples at 200Hz
}


def create_data_config_for_segment_length(segment_length: str, batch_size: int = None, 
                                        random_seed: int = 42, num_gpus: int = 1) -> DataConfig:
    """
    Create DataConfig for a specific segment length.
    
    Args:
        segment_length: '1s' or '4s'
        batch_size: Batch size (if None, uses default: 256 for 1s, 128 for 4s)
        random_seed: Random seed
        
    Returns:
        DataConfig configured for the specified segment length
    """
    if segment_length not in DATA_PATHS:
        raise ValueError(f"Invalid segment_length: {segment_length}. Must be '1s' or '4s'")
    
    # Use smaller batch size for 4s segments to avoid OOM (4x larger data)
    if batch_size is None:
        batch_size = 256 if segment_length == '1s' else 128  # Increased: 256 for 1s, 128 for 4s
    
    # Use num_workers based on segment length and number of GPUs
    # For 1s: always use 4 workers for parallel loading
    # For 4s: optimize for speed with more workers (since RES is low, we have memory headroom)
    #   - With 2+ GPUs: use 3 workers per GPU for faster parallel loading
    #   - With 1 GPU: use 4 workers for parallel loading (RES is only 2GB, plenty of room)
    if segment_length == '1s':
        num_workers = 4
    else:  # 4s mode
        if num_gpus >= 2:
            num_workers = 3 * num_gpus  # 3 workers per GPU for better parallelization
        else:
            num_workers = 4  # Use 4 workers for faster loading with single GPU
    
    config = DataConfig(
        pickle_dir=DATA_PATHS[segment_length],
        segment_length=SEGMENT_LENGTHS[segment_length],
        batch_size=batch_size,
        random_seed=random_seed,
        num_workers=num_workers
    )
    
    return config


def create_all_experiments_for_segment_length(segment_length: str, epochs: int = 50, 
                                            learning_rate: float = 0.001, batch_size: int = None,
                                            num_gpus: int = 1):
    """
    Create all experiment types for a specific segment length.
    
    Args:
        segment_length: '1s' or '4s'
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size (if None, uses default: 256 for 1s, 128 for 4s)
        
    Returns:
        List of all experiment configurations
    """
    experiments = []

    # Create base data config for this segment length
    # This will auto-select batch_size if None
    base_data_config = create_data_config_for_segment_length(segment_length, batch_size, random_seed=42, num_gpus=num_gpus)
    
    # 1-2. Baseline experiments
    experiments.append(ExperimentConfig(
        name=f"gender_baseline_{segment_length}",
        model_type="gender_cnn",
        target_type="gender",
        description=f"Gender classification baseline with {segment_length} segments",
        data_config=base_data_config,
        model_config=ModelConfig(num_channels=60),
        training_config=TrainingConfig(epochs=epochs, learning_rate=learning_rate, target_key="gender")
    ))
    
    experiments.append(ExperimentConfig(
        name=f"age_baseline_{segment_length}",
        model_type="age_cnn",
        target_type="age",
        description=f"Age classification baseline with {segment_length} segments",
        data_config=base_data_config,
        model_config=ModelConfig(num_channels=60),
        training_config=TrainingConfig(epochs=epochs, learning_rate=learning_rate, target_key="age")
    ))
    
    # 3-4. Cross-validation experiments
    gender_cv_config = create_data_config_for_segment_length(segment_length, batch_size, num_gpus=num_gpus)
    gender_cv_config.use_cross_validation = True
    gender_cv_config.n_folds = 5
    gender_cv_config.cv_strategy = "gender"
    
    experiments.append(ExperimentConfig(
        name=f"gender_cv_gender_stratified_{segment_length}",
        model_type="gender_cnn",
        target_type="gender",
        description=f"Gender classification with 5-fold CV (gender stratified) using {segment_length} segments",
        data_config=gender_cv_config,
        model_config=ModelConfig(num_channels=60),
        training_config=TrainingConfig(epochs=epochs, learning_rate=learning_rate, target_key="gender")
    ))
    
    age_cv_config = create_data_config_for_segment_length(segment_length, batch_size, num_gpus=num_gpus)
    age_cv_config.use_cross_validation = True
    age_cv_config.n_folds = 5
    age_cv_config.cv_strategy = "age"
    
    experiments.append(ExperimentConfig(
        name=f"age_cv_age_stratified_{segment_length}",
        model_type="age_cnn",
        target_type="age",
        description=f"Age classification with 5-fold CV (age stratified) using {segment_length} segments",
        data_config=age_cv_config,
        model_config=ModelConfig(num_channels=60),
        training_config=TrainingConfig(epochs=epochs, learning_rate=learning_rate, target_key="age")
    ))
    
    # 5-8. Cross-task experiments (only valid combinations)
    for train_task, val_test_task in [("active", "passive"), ("passive", "active")]:
        # Gender cross-task
        gender_cross_config = create_data_config_for_segment_length(segment_length, batch_size, num_gpus=num_gpus)
        gender_cross_config.train_task_type = train_task
        gender_cross_config.val_test_task_type = val_test_task
        
        experiments.append(ExperimentConfig(
            name=f"gender_cross_task_{train_task}_to_{val_test_task}_{segment_length}",
            model_type="gender_cnn",
            target_type="gender",
            description=f"Gender classification cross-task ({train_task}->{val_test_task}) using {segment_length} segments",
            data_config=gender_cross_config,
            model_config=ModelConfig(num_channels=60),
            training_config=TrainingConfig(epochs=epochs, learning_rate=learning_rate, target_key="gender")
        ))
        
        # Age cross-task
        age_cross_config = create_data_config_for_segment_length(segment_length, batch_size, num_gpus=num_gpus)
        age_cross_config.train_task_type = train_task
        age_cross_config.val_test_task_type = val_test_task
        
        experiments.append(ExperimentConfig(
            name=f"age_cross_task_{train_task}_to_{val_test_task}_{segment_length}",
            model_type="age_cnn",
            target_type="age",
            description=f"Age classification cross-task ({train_task}->{val_test_task}) using {segment_length} segments",
            data_config=age_cross_config,
            model_config=ModelConfig(num_channels=60),
            training_config=TrainingConfig(epochs=epochs, learning_rate=learning_rate, target_key="age")
        ))
    
    # 9-12. Cross-task cross-validation experiments (only valid combinations)
    for train_task, val_test_task in [("active", "passive"), ("passive", "active")]:
        # Gender cross-task CV
        gender_cross_cv_config = create_data_config_for_segment_length(segment_length, batch_size, num_gpus=num_gpus)
        gender_cross_cv_config.use_cross_validation = True
        gender_cross_cv_config.n_folds = 5
        gender_cross_cv_config.cv_strategy = "gender"
        gender_cross_cv_config.train_task_type = train_task
        gender_cross_cv_config.val_test_task_type = val_test_task
        
        experiments.append(ExperimentConfig(
            name=f"gender_cross_task_cv_{train_task}_to_{val_test_task}_gender_stratified_{segment_length}",
            model_type="gender_cnn",
            target_type="gender",
            description=f"Gender classification cross-task CV ({train_task}->{val_test_task}, gender stratified) using {segment_length} segments",
            data_config=gender_cross_cv_config,
            model_config=ModelConfig(num_channels=60),
            training_config=TrainingConfig(epochs=epochs, learning_rate=learning_rate, target_key="gender")
        ))
        
        # Age cross-task CV
        age_cross_cv_config = create_data_config_for_segment_length(segment_length, batch_size, num_gpus=num_gpus)
        age_cross_cv_config.use_cross_validation = True
        age_cross_cv_config.n_folds = 5
        age_cross_cv_config.cv_strategy = "age"
        age_cross_cv_config.train_task_type = train_task
        age_cross_cv_config.val_test_task_type = val_test_task
        
        experiments.append(ExperimentConfig(
            name=f"age_cross_task_cv_{train_task}_to_{val_test_task}_age_stratified_{segment_length}",
            model_type="age_cnn",
            target_type="age",
            description=f"Age classification cross-task CV ({train_task}->{val_test_task}, age stratified) using {segment_length} segments",
            data_config=age_cross_cv_config,
            model_config=ModelConfig(num_channels=60),
            training_config=TrainingConfig(epochs=epochs, learning_rate=learning_rate, target_key="age")
        ))

    # 21-30. Combined age+gender classification experiments (6 classes)
    # 21. Combined baseline (train/val/test split)
    combined_baseline_config = create_data_config_for_segment_length(segment_length, batch_size, num_gpus=num_gpus)
    
    experiments.append(ExperimentConfig(
        name=f"combined_baseline_{segment_length}",
        model_type="combined_cnn",
        target_type="combined",
        description=f"Combined age+gender classification baseline with {segment_length} segments (6 classes)",
        data_config=combined_baseline_config,
        model_config=ModelConfig(num_channels=60, num_classes=6),
        training_config=TrainingConfig(epochs=epochs, learning_rate=learning_rate, target_key="combined")
    ))
    
    # 22. Combined cross-validation (combined stratified)
    combined_cv_config = create_data_config_for_segment_length(segment_length, batch_size, num_gpus=num_gpus)
    combined_cv_config.use_cross_validation = True
    combined_cv_config.n_folds = 5
    combined_cv_config.cv_strategy = "combined"
    
    experiments.append(ExperimentConfig(
        name=f"combined_cv_combined_stratified_{segment_length}",
        model_type="combined_cnn",
        target_type="combined",
        description=f"Combined age+gender classification with 5-fold CV (combined stratified) using {segment_length} segments (6 classes)",
        data_config=combined_cv_config,
        model_config=ModelConfig(num_channels=60, num_classes=6),
        training_config=TrainingConfig(epochs=epochs, learning_rate=learning_rate, target_key="combined")
    ))
    
    # 23-24. Combined cross-task experiments (only valid combinations)
    for train_task, val_test_task in [("active", "passive"), ("passive", "active")]:
        combined_cross_config = create_data_config_for_segment_length(segment_length, batch_size, num_gpus=num_gpus)
        combined_cross_config.train_task_type = train_task
        combined_cross_config.val_test_task_type = val_test_task
        
        experiments.append(ExperimentConfig(
            name=f"combined_cross_task_{train_task}_to_{val_test_task}_{segment_length}",
            model_type="combined_cnn",
            target_type="combined",
            description=f"Combined age+gender classification cross-task ({train_task}->{val_test_task}) using {segment_length} segments (6 classes)",
            data_config=combined_cross_config,
            model_config=ModelConfig(num_channels=60, num_classes=6),
            training_config=TrainingConfig(epochs=epochs, learning_rate=learning_rate, target_key="combined")
        ))
    
    # 25-28. Combined cross-task cross-validation experiments (only valid combinations)
    for train_task, val_test_task in [("active", "passive"), ("passive", "active")]:
        combined_cross_cv_config = create_data_config_for_segment_length(segment_length, batch_size, num_gpus=num_gpus)
        combined_cross_cv_config.use_cross_validation = True
        combined_cross_cv_config.n_folds = 5
        combined_cross_cv_config.cv_strategy = "combined"
        combined_cross_cv_config.train_task_type = train_task
        combined_cross_cv_config.val_test_task_type = val_test_task
        
        experiments.append(ExperimentConfig(
            name=f"combined_cross_task_cv_{train_task}_to_{val_test_task}_combined_stratified_{segment_length}",
            model_type="combined_cnn",
            target_type="combined",
            description=f"Combined age+gender classification cross-task CV ({train_task}->{val_test_task}, combined stratified) using {segment_length} segments (6 classes)",
            data_config=combined_cross_cv_config,
            model_config=ModelConfig(num_channels=60, num_classes=6),
            training_config=TrainingConfig(epochs=epochs, learning_rate=learning_rate, target_key="combined")
        ))
   
    # 31-40. Multi-output gender+age classification experiments (2 separate heads)
    # 31. Multi-output baseline (train/val/test split)
    multi_output_baseline_config = create_data_config_for_segment_length(segment_length, batch_size, num_gpus=num_gpus)
    
    experiments.append(ExperimentConfig(
        name=f"multi_output_baseline_{segment_length}",
        model_type="multi_output_cnn",
        target_type="multi_output",
        description=f"Multi-output gender+age classification baseline with {segment_length} segments (2 separate heads)",
        data_config=multi_output_baseline_config,
        model_config=ModelConfig(num_channels=60, num_classes=2),  # Will be overridden by multi-output heads
        training_config=TrainingConfig(epochs=epochs, learning_rate=learning_rate, target_key="multi_output")
    ))
    
    # 32. Multi-output cross-validation (gender stratified)
    multi_output_cv_config = create_data_config_for_segment_length(segment_length, batch_size, num_gpus=num_gpus)
    multi_output_cv_config.use_cross_validation = True
    multi_output_cv_config.n_folds = 5
    multi_output_cv_config.cv_strategy = "gender"  # Use gender for stratification
    
    experiments.append(ExperimentConfig(
        name=f"multi_output_cv_gender_stratified_{segment_length}",
        model_type="multi_output_cnn",
        target_type="multi_output",
        description=f"Multi-output gender+age classification with 5-fold CV (gender stratified) using {segment_length} segments",
        data_config=multi_output_cv_config,
        model_config=ModelConfig(num_channels=60, num_classes=2),
        training_config=TrainingConfig(epochs=epochs, learning_rate=learning_rate, target_key="multi_output")
    ))
    
    # 33-34. Multi-output cross-task experiments (only valid combinations)
    for train_task, val_test_task in [("active", "passive"), ("passive", "active")]:
        multi_output_cross_config = create_data_config_for_segment_length(segment_length, batch_size, num_gpus=num_gpus)
        multi_output_cross_config.train_task_type = train_task
        multi_output_cross_config.val_test_task_type = val_test_task
        
        experiments.append(ExperimentConfig(
            name=f"multi_output_cross_task_{train_task}_to_{val_test_task}_{segment_length}",
            model_type="multi_output_cnn",
            target_type="multi_output",
            description=f"Multi-output gender+age classification cross-task ({train_task}->{val_test_task}) using {segment_length} segments",
            data_config=multi_output_cross_config,
            model_config=ModelConfig(num_channels=60, num_classes=2),
            training_config=TrainingConfig(epochs=epochs, learning_rate=learning_rate, target_key="multi_output")
        ))
    
    # 35-38. Multi-output cross-task cross-validation experiments (only valid combinations)
    for train_task, val_test_task in [("active", "passive"), ("passive", "active")]:
        multi_output_cross_cv_config = create_data_config_for_segment_length(segment_length, batch_size, num_gpus=num_gpus)
        multi_output_cross_cv_config.use_cross_validation = True
        multi_output_cross_cv_config.n_folds = 5
        multi_output_cross_cv_config.cv_strategy = "gender"  # Use gender for stratification
        multi_output_cross_cv_config.train_task_type = train_task
        multi_output_cross_cv_config.val_test_task_type = val_test_task
        
        experiments.append(ExperimentConfig(
            name=f"multi_output_cross_task_cv_{train_task}_to_{val_test_task}_gender_stratified_{segment_length}",
            model_type="multi_output_cnn",
            target_type="multi_output",
            description=f"Multi-output gender+age classification cross-task CV ({train_task}->{val_test_task}, gender stratified) using {segment_length} segments",
            data_config=multi_output_cross_cv_config,
            model_config=ModelConfig(num_channels=60, num_classes=2),
            training_config=TrainingConfig(epochs=epochs, learning_rate=learning_rate, target_key="multi_output")
        ))

    return experiments


def main():
    """Main function for running EEG classification experiments."""
    parser = argparse.ArgumentParser(description='EEG Classification Experiment Runner')
    parser.add_argument('--mode', choices=['1s', '4s'], default='1s',
                       help='EEG segment length: 1s (1-second segments) or 4s (4-second segments)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=None, 
                       help='Batch size (if None, uses default: 256 for 1s, 128 for 4s)')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of GPUs to use for DataParallel (default: 2)')
    parser.add_argument('--results_dir', type=str, default='experiment_results', 
                       help='Directory to save results')
    parser.add_argument('--reports_dir', type=str, default='reports', 
                       help='Directory to save reports')
    
    args = parser.parse_args()
    
    # Auto-select fewer GPUs for 4s mode to reduce memory usage
    actual_num_gpus = args.num_gpus
    if args.mode == '4s' and args.num_gpus == 2:
        actual_num_gpus = 1  # Use single GPU for 4s to reduce memory pressure
    
    # Create system configuration
    system_config = SystemConfig(
        results_dir=args.results_dir,
        reports_dir=args.reports_dir,
        num_gpus=actual_num_gpus
    )
    
    # Determine actual batch size (auto-select if None)
    actual_batch_size = args.batch_size if args.batch_size is not None else (256 if args.mode == '1s' else 128)
    
    # Adjust num_workers based on number of GPUs for 4s mode
    if args.mode == '4s':
        if actual_num_gpus >= 2:
            # With multiple GPUs, use more workers for parallel loading
            # Each worker has its own cache, so we use smaller cache per worker
            print(f"ℹ️  4s mode with {actual_num_gpus} GPUs: using 2 workers per GPU for parallel loading")
        else:
            # Single GPU: use fewer workers to allow larger cache
            print(f"ℹ️  4s mode with 1 GPU: using 0 workers to maximize cache size")
    
    print("="*80)
    print("EEG CLASSIFICATION EXPERIMENT RUNNER")
    print("="*80)
    print(f"Mode: {args.mode} (EEG segment length)")
    print(f"Results directory: {args.results_dir}")
    print(f"Reports directory: {args.reports_dir}")
    print(f"Epochs: {args.epochs}, Learning rate: {args.learning_rate}")
    print(f"Batch size: {actual_batch_size} {'(auto-selected)' if args.batch_size is None else ''}, Random seed: {args.random_seed}")
    print(f"Number of GPUs: {actual_num_gpus}")
    # Get num_workers for display
    config = create_data_config_for_segment_length(args.mode, batch_size=actual_batch_size, num_gpus=actual_num_gpus)
    num_workers = config.num_workers
    print(f"Number of workers: {num_workers}")
    if args.mode == '4s':
        if actual_num_gpus >= 2:
            print(f"NOTE: Using batch_size={actual_batch_size}, num_workers={num_workers} ({2*actual_num_gpus} workers with {actual_num_gpus} GPUs)")
            print(f"      (4x larger data than 1s segments, but lazy loading enabled)")
            print(f"      (Parallel loading with {num_workers} workers across {actual_num_gpus} GPUs)")
        else:
            print(f"NOTE: Using batch_size={actual_batch_size}, num_workers={num_workers}, cache=35 files (~11.7GB)")
            print(f"      (4x larger data than 1s segments, but lazy loading enabled)")
            print(f"      (num_workers=0 allows larger cache without worker memory overhead)")
    print("="*80)
    
    # Create all experiments for the specified segment length
    print(f"Creating all experiments for {args.mode} EEG segments...")
    experiments = create_all_experiments_for_segment_length(
        segment_length=args.mode,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_gpus=actual_num_gpus
    )
    
    print(f"Running {len(experiments)} experiments for {args.mode} segments...")
    
    # Run all experiments
    results = run_experiments(experiments, system_config, args.random_seed, generate_reports=True)
    
    print(f"All {args.mode} experiments completed successfully!")
    print(f"Results saved to: {args.results_dir}")
    print(f"Reports generated in: {args.reports_dir}")

if __name__ == "__main__":
    main()
