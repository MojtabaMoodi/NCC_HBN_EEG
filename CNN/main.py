"""
Systematic Main Script for EEG Classification Experiments
Provides a comprehensive interface for running multiple experiments and generating reports.
"""

import argparse
from config import SystemConfig, DataConfig, ModelConfig, TrainingConfig, ExperimentConfig
from experiment import run_experiments


# Data paths and segment lengths
DATA_PATHS = {
    '1s': "/home/mojtabam/projects/def-aghodsib/mojtabam/EEG/data_processing/processed_eeg_data_1s_segments",
    '4s': "/home/mojtabam/projects/def-aghodsib/mojtabam/EEG/data_processing/processed_eeg_data_4s_segments"
}

SEGMENT_LENGTHS = {
    '1s': 200,  # 1 second = 200 samples at 200Hz
    '4s': 800   # 4 seconds = 800 samples at 200Hz
}


def create_data_config_for_segment_length(segment_length: str, batch_size: int = 128, 
                                        random_seed: int = 42) -> DataConfig:
    """
    Create DataConfig for a specific segment length.
    
    Args:
        segment_length: '1s' or '4s'
        batch_size: Batch size
        random_seed: Random seed
        
    Returns:
        DataConfig configured for the specified segment length
    """
    if segment_length not in DATA_PATHS:
        raise ValueError(f"Invalid segment_length: {segment_length}. Must be '1s' or '4s'")
    
    return DataConfig(
        pickle_dir=DATA_PATHS[segment_length],
        segment_length=SEGMENT_LENGTHS[segment_length],
        batch_size=batch_size,
        random_seed=random_seed
    )


def create_all_experiments_for_segment_length(segment_length: str, epochs: int = 50, 
                                            learning_rate: float = 0.001, batch_size: int = 128):
    """
    Create all experiment types for a specific segment length.
    
    Args:
        segment_length: '1s' or '4s'
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        
    Returns:
        List of all experiment configurations
    """
    experiments = []

    # Create base data config for this segment length
    base_data_config = create_data_config_for_segment_length(segment_length, batch_size)
    
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
    gender_cv_config = create_data_config_for_segment_length(segment_length, batch_size)
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
    
    age_cv_config = create_data_config_for_segment_length(segment_length, batch_size)
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
        gender_cross_config = create_data_config_for_segment_length(segment_length, batch_size)
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
        age_cross_config = create_data_config_for_segment_length(segment_length, batch_size)
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
        gender_cross_cv_config = create_data_config_for_segment_length(segment_length, batch_size)
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
        age_cross_cv_config = create_data_config_for_segment_length(segment_length, batch_size)
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
    combined_baseline_config = create_data_config_for_segment_length(segment_length, batch_size)
    
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
    combined_cv_config = create_data_config_for_segment_length(segment_length, batch_size)
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
        combined_cross_config = create_data_config_for_segment_length(segment_length, batch_size)
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
        combined_cross_cv_config = create_data_config_for_segment_length(segment_length, batch_size)
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
    multi_output_baseline_config = create_data_config_for_segment_length(segment_length, batch_size)
    
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
    multi_output_cv_config = create_data_config_for_segment_length(segment_length, batch_size)
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
        multi_output_cross_config = create_data_config_for_segment_length(segment_length, batch_size)
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
        multi_output_cross_cv_config = create_data_config_for_segment_length(segment_length, batch_size)
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
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--results_dir', type=str, default='experiment_results', 
                       help='Directory to save results')
    parser.add_argument('--reports_dir', type=str, default='reports', 
                       help='Directory to save reports')
    
    args = parser.parse_args()
    
    # Create system configuration
    system_config = SystemConfig(
        results_dir=args.results_dir,
        reports_dir=args.reports_dir
    )
    
    print("="*80)
    print("EEG CLASSIFICATION EXPERIMENT RUNNER")
    print("="*80)
    print(f"Mode: {args.mode} (EEG segment length)")
    print(f"Results directory: {args.results_dir}")
    print(f"Reports directory: {args.reports_dir}")
    print(f"Epochs: {args.epochs}, Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}, Random seed: {args.random_seed}")
    print("="*80)
    
    # Create all experiments for the specified segment length
    print(f"Creating all experiments for {args.mode} EEG segments...")
    experiments = create_all_experiments_for_segment_length(
        segment_length=args.mode,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )
    
    print(f"Running {len(experiments)} experiments for {args.mode} segments...")
    
    # Run all experiments
    results = run_experiments(experiments, system_config, args.random_seed, generate_reports=True)
    
    print(f"All {args.mode} experiments completed successfully!")
    print(f"Results saved to: {args.results_dir}")
    print(f"Reports generated in: {args.reports_dir}")

if __name__ == "__main__":
    main()
