"""
Systematic Main Script for EEG Classification Experiments
Provides a comprehensive interface for running multiple experiments and generating reports.
"""

import sys
import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import torch
import numpy as np

# Add the data_processing directory to the path FIRST
sys.path.append('/home/mojtabam/projects/def-aghodsib/mojtabam/EEG/data_processing')

from config import TrainingConfig, ModelConfig, SystemConfig
from experiment import Experiment, run_multiple_experiments, create_default_experiments, create_custom_experiments, ExperimentResult
from config import ExperimentConfig, DataConfig
from report_generator import ReportGenerator
from eeg_dataset import EEGDataLoader
from target_transforms import gender_classification_transform, age_classification_transform

def create_comprehensive_experiments() -> List[ExperimentConfig]:
    """
    Create comprehensive experiment configurations using different data loader methods.
    
    Returns:
        List of ExperimentConfig objects for all 12 experiment types
    """
    experiments = []
    
    # ============================================================================
    # 1-2. BASIC TRAIN/VAL/TEST EXPERIMENTS
    # ============================================================================
    
    # 1. Gender classification using create_train_val_test_loaders
    experiments.append(ExperimentConfig(
        name="gender_baseline_train_val_test",
        model_type="gender_cnn",
        target_type="gender",
        data_config=DataConfig(),
        model_config=ModelConfig(num_channels=60),
        training_config=TrainingConfig(epochs=50, learning_rate=0.001, target_key="gender")
    ))
    
    # 2. Age classification using create_train_val_test_loaders
    experiments.append(ExperimentConfig(
        name="age_baseline_train_val_test",
        model_type="age_cnn",
        target_type="age",
        data_config=DataConfig(),
        model_config=ModelConfig(num_channels=60),
        training_config=TrainingConfig(epochs=50, learning_rate=0.001, target_key="age")
    ))
    
    # ============================================================================
    # 3-4. CROSS-VALIDATION EXPERIMENTS WITH STRATIFICATION
    # ============================================================================
    
    # 3. Gender classification using create_cross_validation_loaders with gender stratification
    experiments.append(ExperimentConfig(
        name="gender_cv_gender_stratified",
        model_type="gender_cnn",
        target_type="gender",
        data_config=DataConfig(use_cross_validation=True, n_folds=5, cv_strategy="gender"),
        model_config=ModelConfig(num_channels=60),
        training_config=TrainingConfig(epochs=50, learning_rate=0.001, target_key="gender")
    ))
    
    # 4. Age classification using create_cross_validation_loaders with age stratification
    experiments.append(ExperimentConfig(
        name="age_cv_age_stratified",
        model_type="age_cnn",
        target_type="age",
        data_config=DataConfig(use_cross_validation=True, n_folds=5, cv_strategy="age"),
        model_config=ModelConfig(num_channels=60),
        training_config=TrainingConfig(epochs=50, learning_rate=0.001, target_key="age")
    ))
    
    # ============================================================================
    # 5-8. CROSS-TASK EXPERIMENTS
    # ============================================================================
    
    # 5. Gender classification: active training -> passive validation
    experiments.append(ExperimentConfig(
        name="gender_cross_task_active_to_passive",
        model_type="gender_cnn",
        target_type="gender",
        data_config=DataConfig(train_task_type="active", val_test_task_type="passive"),
        model_config=ModelConfig(num_channels=60),
        training_config=TrainingConfig(epochs=50, learning_rate=0.001, target_key="gender")
    ))
    
    # 6. Gender classification: passive training -> active validation
    experiments.append(ExperimentConfig(
        name="gender_cross_task_passive_to_active",
        model_type="gender_cnn",
        target_type="gender",
        data_config=DataConfig(train_task_type="passive", val_test_task_type="active"),
        model_config=ModelConfig(num_channels=60),
        training_config=TrainingConfig(epochs=50, learning_rate=0.001, target_key="gender")
    ))
    
    # 7. Age classification: active training -> passive validation
    experiments.append(ExperimentConfig(
        name="age_cross_task_active_to_passive",
        model_type="age_cnn",
        target_type="age",
        data_config=DataConfig(train_task_type="active", val_test_task_type="passive"),
        model_config=ModelConfig(num_channels=60),
        training_config=TrainingConfig(epochs=50, learning_rate=0.001, target_key="age")
    ))
    
    # 8. Age classification: passive training -> active validation
    experiments.append(ExperimentConfig(
        name="age_cross_task_passive_to_active",
        model_type="age_cnn",
        target_type="age",
        data_config=DataConfig(train_task_type="passive", val_test_task_type="active"),
        model_config=ModelConfig(num_channels=60),
        training_config=TrainingConfig(epochs=50, learning_rate=0.001, target_key="age")
    ))
    
    # ============================================================================
    # 9-12. CROSS-TASK CROSS-VALIDATION EXPERIMENTS
    # ============================================================================
    
    # 9. Gender classification: active training -> passive validation with gender stratification
    experiments.append(ExperimentConfig(
        name="gender_cross_task_cv_active_to_passive_gender_stratified",
        model_type="gender_cnn",
        target_type="gender",
        data_config=DataConfig(
            use_cross_validation=True, 
            n_folds=5, 
            cv_strategy="gender",
            train_task_type="active", 
            val_test_task_type="passive"
        ),
        model_config=ModelConfig(num_channels=60),
        training_config=TrainingConfig(epochs=50, learning_rate=0.001, target_key="gender")
    ))
    
    # 10. Gender classification: passive training -> active validation with gender stratification
    experiments.append(ExperimentConfig(
        name="gender_cross_task_cv_passive_to_active_gender_stratified",
        model_type="gender_cnn",
        target_type="gender",
        data_config=DataConfig(
            use_cross_validation=True, 
            n_folds=5, 
            cv_strategy="gender",
            train_task_type="passive", 
            val_test_task_type="active"
        ),
        model_config=ModelConfig(num_channels=60),
        training_config=TrainingConfig(epochs=50, learning_rate=0.001, target_key="gender")
    ))
    
    # 11. Age classification: active training -> passive validation with age stratification
    experiments.append(ExperimentConfig(
        name="age_cross_task_cv_active_to_passive_age_stratified",
        model_type="age_cnn",
        target_type="age",
        data_config=DataConfig(
            use_cross_validation=True, 
            n_folds=5, 
            cv_strategy="age",
            train_task_type="active", 
            val_test_task_type="passive"
        ),
        model_config=ModelConfig(num_channels=60),
        training_config=TrainingConfig(epochs=50, learning_rate=0.001, target_key="age")
    ))
    
    # 12. Age classification: passive training -> active validation with age stratification
    experiments.append(ExperimentConfig(
        name="age_cross_task_cv_passive_to_active_age_stratified",
        model_type="age_cnn",
        target_type="age",
        data_config=DataConfig(
            use_cross_validation=True, 
            n_folds=5, 
            cv_strategy="age",
            train_task_type="passive", 
            val_test_task_type="active"
        ),
        model_config=ModelConfig(num_channels=60),
        training_config=TrainingConfig(epochs=50, learning_rate=0.001, target_key="age")
    ))
    
    return experiments


def create_data_loaders(data_config: DataConfig) -> Dict[str, Tuple]:
    """
    Create data loaders for all target types using eeg_dataset.py directly.
    
    Args:
        data_config: DataConfig instance containing data loading parameters
        
    Returns:
        Dictionary mapping target_type to (train_loader, val_loader, test_loader)
    """
    print("Creating data loaders for all target types...")
    
    data_loaders = {}
    
    for target_type in ['gender', 'age']:
        print(f"Creating data loaders for {target_type} classification...")
        
        # Select appropriate transform based on target type
        if target_type == 'gender':
            gender_transform = gender_classification_transform
            age_transform = None
        else:  # age
            gender_transform = None
            age_transform = age_classification_transform
        
        # Create data loaders using eeg_dataset.py directly
        train_loader, val_loader, test_loader = EEGDataLoader.create_train_val_test_loaders(
            pickle_dir=data_config.pickle_dir,
            batch_size=data_config.batch_size,
            num_workers=data_config.num_workers,
            random_seed=data_config.random_seed,
            task_type=data_config.task_type,
            target_type=target_type,
            gender_transform=gender_transform,
            age_transform=age_transform,
            transform=None
        )
        
        data_loaders[target_type] = (train_loader, val_loader, test_loader)
        print(f"{target_type.capitalize()} data: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
    
    return data_loaders


def create_specialized_data_loaders(experiment_config: ExperimentConfig) -> Tuple:
    """
    Create specialized data loaders based on experiment configuration.
    
    Args:
        experiment_config: ExperimentConfig instance containing experiment parameters
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader) or list of fold tuples
    """
    data_config = experiment_config.data_config
    target_type = experiment_config.target_type
    
    # Select appropriate transforms
    if target_type == 'gender':
        gender_transform = gender_classification_transform
        age_transform = None
    else:  # age
        gender_transform = None
        age_transform = age_classification_transform
    
    # Handle different experiment types
    if data_config.use_cross_validation and data_config.train_task_type and data_config.val_test_task_type:
        # Cross-task cross-validation experiments (9-12)
        print(f"Creating cross-task cross-validation loaders: {data_config.train_task_type} -> {data_config.val_test_task_type}")
        return EEGDataLoader.create_cross_task_cross_validation_loaders(
            pickle_dir=data_config.pickle_dir,
            train_task_type=data_config.train_task_type,
            val_test_task_type=data_config.val_test_task_type,
            n_folds=data_config.n_folds,
            batch_size=data_config.batch_size,
            num_workers=data_config.num_workers,
            random_seed=data_config.random_seed,
            stratify_by=data_config.cv_strategy,
            target_type=target_type,
            gender_transform=gender_transform,
            age_transform=age_transform,
            transform=None
        )
    
    elif data_config.use_cross_validation:
        # Cross-validation experiments (3-4)
        print(f"Creating cross-validation loaders with {data_config.cv_strategy} stratification")
        return EEGDataLoader.create_cross_validation_loaders(
            pickle_dir=data_config.pickle_dir,
            n_folds=data_config.n_folds,
            batch_size=data_config.batch_size,
            num_workers=data_config.num_workers,
            random_seed=data_config.random_seed,
            stratify_by=data_config.cv_strategy,
            target_type=target_type,
            gender_transform=gender_transform,
            age_transform=age_transform,
            transform=None
        )
    
    elif data_config.train_task_type and data_config.val_test_task_type:
        # Cross-task experiments (5-8)
        print(f"Creating cross-task loaders: {data_config.train_task_type} -> {data_config.val_test_task_type}")
        return EEGDataLoader.create_cross_task_loaders(
            pickle_dir=data_config.pickle_dir,
            train_task_type=data_config.train_task_type,
            val_test_task_type=data_config.val_test_task_type,
            batch_size=data_config.batch_size,
            num_workers=data_config.num_workers,
            random_seed=data_config.random_seed,
            target_type=target_type,
            gender_transform=gender_transform,
            age_transform=age_transform,
            transform=None
        )
    
    else:
        # Standard train/val/test experiments (1-2)
        print("Creating standard train/val/test loaders")
        return EEGDataLoader.create_train_val_test_loaders(
            pickle_dir=data_config.pickle_dir,
            batch_size=data_config.batch_size,
            num_workers=data_config.num_workers,
            random_seed=data_config.random_seed,
            task_type=data_config.task_type,
            target_type=target_type,
            gender_transform=gender_transform,
            age_transform=age_transform,
            transform=None
        )

def run_batch_experiments(experiments, data_config, system_config, random_seed, experiment_type):
    """Run a batch of experiments and generate reports."""
    print(f"Running {experiment_type} experiments...")
    
    # Create data loaders for all target types
    data_loaders = create_data_loaders(data_config)
    
    # Run all experiments
    results = run_multiple_experiments(experiments, data_loaders, system_config, random_seed)
    
    # Generate reports
    report_generator = ReportGenerator(system_config.results_dir, system_config.reports_dir)
    report_generator.generate_all_reports(results)
    
    print(f"{experiment_type.title()} experiments completed. Results in: {system_config.results_dir}")
    print(f"Reports generated in: {system_config.reports_dir}")


def run_comprehensive_experiments(experiments, system_config, random_seed):
    """Run comprehensive experiments with specialized data loaders for each experiment."""
    print(f"Running comprehensive experiments...")
    
    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    print(f"\n{'='*80}")
    print(f"STARTING COMPREHENSIVE EXPERIMENTS")
    print(f"Number of experiments: {len(experiments)}")
    print(f"Random seed: {random_seed}")
    print(f"Results directory: {system_config.results_dir}")
    print(f"{'='*80}")
    
    results = []
    
    for i, config in enumerate(experiments):
        print(f"\nProgress: {i+1}/{len(experiments)}")
        print(f"Experiment: {config.name}")
        
        try:
            # Create specialized data loaders for this experiment
            data_loaders = create_specialized_data_loaders(config)
            
            # Handle different data loader types
            if isinstance(data_loaders, list):
                # Cross-validation experiments - run each fold
                print(f"Running {len(data_loaders)} folds for cross-validation...")
                fold_results = []
                
                for fold_idx, (train_loader, val_loader, test_loader) in enumerate(data_loaders):
                    print(f"  Fold {fold_idx + 1}/{len(data_loaders)}")
                    
                    # Create experiment for this fold
                    fold_config = ExperimentConfig(
                        name=f"{config.name}_fold_{fold_idx + 1}",
                        model_type=config.model_type,
                        target_type=config.target_type,
                        data_config=config.data_config,
                        model_config=config.model_config,
                        training_config=config.training_config
                    )
                    
                    experiment = Experiment(fold_config, system_config)
                    experiment.setup(train_loader, val_loader, test_loader)
                    result = experiment.run()
                    fold_results.append(result)
                
                # Aggregate fold results
                if fold_results and all(r.success for r in fold_results):
                    # Calculate average metrics across folds
                    avg_metrics = {}
                    for key in fold_results[0].metrics.keys():
                        if isinstance(fold_results[0].metrics[key], (int, float)):
                            # Numeric metrics - calculate average
                            avg_metrics[key] = sum(r.metrics[key] for r in fold_results) / len(fold_results)
                        else:
                            # Non-numeric metrics (like lists) - use first fold's value
                            avg_metrics[key] = fold_results[0].metrics[key]
                    
                    # Create aggregated result
                    aggregated_result = ExperimentResult(
                        experiment_name=config.name,
                        model_type=config.model_type,
                        target_type=config.target_type,
                        success=True,
                        total_time=sum(r.total_time for r in fold_results),
                        training_time=sum(r.training_time for r in fold_results),
                        evaluation_time=sum(r.evaluation_time for r in fold_results),
                        metrics=avg_metrics,
                        training_metrics=fold_results[0].training_metrics,  # Use first fold's training metrics
                        error=None
                    )
                    results.append(aggregated_result)
                    print(f"  ✅ Cross-validation completed: {len(fold_results)} folds")
                else:
                    print(f"  ❌ Cross-validation failed")
                    results.append(ExperimentResult(
                        experiment_name=config.name,
                        model_type=config.model_type,
                        target_type=config.target_type,
                        success=False,
                        total_time=0.0,
                        training_time=0.0,
                        evaluation_time=0.0,
                        metrics=None,
                        training_metrics=None,
                        error="Cross-validation failed"
                    ))
            
            else:
                # Single experiment (train_loader, val_loader, test_loader)
                train_loader, val_loader, test_loader = data_loaders
                
                experiment = Experiment(config, system_config)
                experiment.setup(train_loader, val_loader, test_loader)
                result = experiment.run()
                results.append(result)
                
                if result.success:
                    print(f"  ✅ Experiment completed")
                else:
                    print(f"  ❌ Experiment failed: {result.error}")
        
        except Exception as e:
            print(f"  ❌ Experiment failed with error: {e}")
            results.append(ExperimentResult(
                experiment_name=config.name,
                model_type=config.model_type,
                target_type=config.target_type,
                success=False,
                total_time=0.0,
                training_time=0.0,
                evaluation_time=0.0,
                metrics=None,
                training_metrics=None,
                error=str(e)
            ))
    
    # Save summary of all results
    _save_experiment_summary(results, system_config.results_dir)
    
    print(f"\n{'='*80}")
    print("ALL COMPREHENSIVE EXPERIMENTS COMPLETED")
    print(f"Results saved to: {system_config.results_dir}")
    print(f"{'='*80}")
    
    return results


def _save_experiment_summary(results, results_dir):
    """Save experiment summary (imported from experiment.py)."""
    from experiment import _save_experiment_summary as _save_summary
    _save_summary(results, results_dir)


def main():
    """Main function for running experiments."""
    parser = argparse.ArgumentParser(description='EEG Classification Experiment Runner')
    parser.add_argument('--mode', choices=['single', 'batch', 'default', 'comprehensive'], default='default',
                       help='Experiment mode: single (one experiment), default (predefined experiments), batch (custom experiments), or comprehensive (all 12 experiment types)')
    parser.add_argument('--experiment', type=str, help='Experiment name for single mode')
    parser.add_argument('--target', choices=['gender', 'age'], help='Target type for single mode')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--results_dir', type=str, default='experiment_results', 
                       help='Directory to save results')
    parser.add_argument('--reports_dir', type=str, default='reports', 
                       help='Directory to save reports')
    
    args = parser.parse_args()
    
    # Create configurations
    data_config = DataConfig(
        batch_size=args.batch_size,
        random_seed=args.random_seed
    )
    
    system_config = SystemConfig(
        results_dir=args.results_dir,
        reports_dir=args.reports_dir
    )
    
    print("="*80)
    print("EEG CLASSIFICATION EXPERIMENT RUNNER")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Results directory: {args.results_dir}")
    print(f"Reports directory: {args.reports_dir}")
    print(f"Data config: batch_size={data_config.batch_size}, random_seed={data_config.random_seed}")
    print("="*80)
    
    if args.mode == 'single':
        # Single experiment mode
        if not args.experiment or not args.target:
            print("Error: --experiment and --target are required for single mode")
            return
        
        # Create data loaders for the specific target type using eeg_dataset.py directly
        if args.target == 'gender':
            gender_transform = gender_classification_transform
            age_transform = None
        else:  # age
            gender_transform = None
            age_transform = age_classification_transform
        
        train_loader, val_loader, test_loader = EEGDataLoader.create_train_val_test_loaders(
            pickle_dir=data_config.pickle_dir,
            batch_size=data_config.batch_size,
            num_workers=data_config.num_workers,
            random_seed=data_config.random_seed,
            task_type=data_config.task_type,
            target_type=args.target,
            gender_transform=gender_transform,
            age_transform=age_transform,
            transform=None
        )
        
        # Create experiment configuration
        model_type = "gender_cnn" if args.target == "gender" else "age_cnn"
        config = ExperimentConfig(
            name=args.experiment,
            model_type=model_type,
            target_type=args.target,
            data_config=data_config,
            model_config=ModelConfig(num_channels=60),
            training_config=TrainingConfig(epochs=args.epochs, learning_rate=args.learning_rate, target_key=args.target)
        )
        
        # Create and run experiment using the Experiment class
        experiment = Experiment(config, system_config)
        experiment.setup(train_loader, val_loader, test_loader)
        result = experiment.run()
        
        if result.success:
            print(f"Single experiment completed: {args.experiment}")
            if result.metrics:
                print(f"Accuracy: {result.metrics.get('accuracy', 'N/A'):.4f}")
        else:
            print(f"Single experiment failed: {result.error}")
    
    elif args.mode == 'default':
        # Default experiments mode - run predefined experiments
        experiments = create_default_experiments()
        run_batch_experiments(experiments, data_config, system_config, args.random_seed, "default")
    
    elif args.mode == 'batch':
        # Batch experiments mode - run custom experiments
        experiments = create_custom_experiments()
        run_batch_experiments(experiments, data_config, system_config, args.random_seed, "custom batch")
    
    elif args.mode == 'comprehensive':
        # Comprehensive experiments mode - run all 12 experiment types
        experiments = create_comprehensive_experiments()
        results = run_comprehensive_experiments(experiments, system_config, args.random_seed)
        
        # Generate reports
        report_generator = ReportGenerator(system_config.results_dir, system_config.reports_dir)
        report_generator.generate_all_reports(results)
        
        print(f"Comprehensive experiments completed. Results in: {system_config.results_dir}")
        print(f"Reports generated in: {system_config.reports_dir}")
    
    else:
        print(f"Error: Unknown mode '{args.mode}'. Available modes: single, default, batch, comprehensive")
        return

if __name__ == "__main__":
    main()
