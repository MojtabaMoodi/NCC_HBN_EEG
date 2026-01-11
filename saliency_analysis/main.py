#!/usr/bin/env python3
"""
Main script for running saliency analysis on EEG models.

Usage (Single Model):
    python main.py --checkpoint <path> --target_type <gender|age> --segment_length <1s|2s|4s> [options]

Usage (Batch Mode - All Best Models):
    python main.py --batch [options]
    
Examples:
    # Process all best models with 4 GPUs
    python main.py --batch --device cuda --num_gpus 4
    
    # Process single model
    python main.py --checkpoint path/to/checkpoint.pth --target_type gender --segment_length 2s
    
    # Process specific models only
    python main.py --batch --models gender_baseline_1s age_classification_2s
"""

import argparse
import sys
from pathlib import Path
import torch

# Add parent directory and CNN directory to path for imports
# Parent directory allows: from CNN.trainer import ...
# CNN directory allows relative imports within CNN module (from models import ...)
parent_dir = Path(__file__).parent.parent
cnn_dir = parent_dir / 'CNN'
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
if str(cnn_dir) not in sys.path:
    sys.path.insert(0, str(cnn_dir))

from analyzer import SaliencyAnalyzer
from visualization import plot_method_comparison, save_method_comparison_report
from CNN.models import ModelFactory
from CNN.gpu_utils import detect_available_gpus, setup_model_for_gpus, print_gpu_info, get_device
from data_processing.eeg_dataset import EEGDataset, EEGDataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute saliency maps for EEG channel importance analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument('--batch', action='store_true',
                       help='Batch mode: process all best models automatically')
    
    # Required arguments (for single model mode)
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint file (required for single model mode)')
    parser.add_argument('--target_type', type=str, default=None,
                       choices=['gender', 'age', 'combined'],
                       help='Type of prediction target (required for single model mode)')
    parser.add_argument('--segment_length', type=str, default=None,
                       choices=['1s', '2s', '4s'],
                       help='EEG segment length (required for single model mode)')
    
    # Data arguments
    parser.add_argument('--hdf5_dir', type=str,
                       default='data_processing/processed_eeg_data_hdf5_no_compression',
                       help='Directory containing HDF5 files')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Data split to analyze')
    parser.add_argument('--task_type', type=str, default='both',
                       choices=['active', 'passive', 'both'],
                       help='Task type to analyze')
    
    # Model arguments
    parser.add_argument('--model_type', type=str,
                       choices=['gender_cnn', 'age_cnn', 'age_regression_cnn', 
                               'combined_cnn', 'multi_output_cnn'],
                       help='Model type (auto-detected from checkpoint if not specified)')
    parser.add_argument('--num_channels', type=int, default=60,
                       help='Number of EEG channels')
    parser.add_argument('--prediction_type', type=str, default='classification',
                       choices=['classification', 'regression'],
                       help='Prediction type (for age models)')
    
    # GPU/Device arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device preference')
    parser.add_argument('--num_gpus', type=int, default=1,
                       help='Number of GPUs to use (1 = single GPU/CPU, >1 = multi-GPU)')
    
    # Saliency method arguments
    parser.add_argument('--method', type=str, default='integrated_gradients',
                       choices=['vanilla_gradients', 'integrated_gradients', 'both'],
                       help='Saliency computation method. Use "both" to compute and compare both methods.')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of steps for integrated gradients')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (None = all)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for data loading')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='saliency_results',
                       help='Output directory for results and visualizations')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of top channels to highlight in visualizations')
    
    # Visualization arguments
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating visualization plots')
    
    # Batch mode arguments
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Specific models to process in batch mode (by name, e.g., gender_baseline_1s). If not specified, processes all models.')
    
    return parser.parse_args()


def get_model_type_from_checkpoint(checkpoint_path: str) -> str:
    """
    Try to infer model type from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
    
    Returns:
        Model type string
    """
    import torch
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Check checkpoint metadata
    if 'model_info' in checkpoint:
        model_name = checkpoint['model_info'].get('model_name', '')
        if 'Gender' in model_name or 'gender' in model_name.lower():
            return 'gender_cnn'
        elif 'Age' in model_name or 'age' in model_name.lower():
            if 'Regression' in model_name or 'regression' in model_name.lower():
                return 'age_regression_cnn'
            else:
                return 'age_cnn'
        elif 'Combined' in model_name or 'combined' in model_name.lower():
            return 'combined_cnn'
        elif 'MultiOutput' in model_name or 'multi_output' in model_name.lower():
            return 'multi_output_cnn'
    
    # Check config
    if 'config' in checkpoint:
        config = checkpoint['config']
        if isinstance(config, dict):
            target_key = config.get('target_key', '')
            prediction_type = config.get('prediction_type', 'classification')
            
            if target_key == 'gender':
                return 'gender_cnn'
            elif target_key == 'age':
                if prediction_type == 'regression':
                    return 'age_regression_cnn'
                else:
                    return 'age_cnn'
            elif target_key == 'combined':
                return 'combined_cnn'
            elif target_key == 'multi_output':
                return 'multi_output_cnn'
    
    raise ValueError(f"Could not determine model type from checkpoint: {checkpoint_path}")


def create_model(model_type: str, num_channels: int, num_classes: int = None,
                 prediction_type: str = 'classification') -> tuple:
    """
    Create model instance.
    
    Args:
        model_type: Model type string
        num_channels: Number of channels
        num_classes: Number of classes (auto-determined if None)
        prediction_type: Prediction type
    
    Returns:
        Tuple of (model, num_classes)
    """
    if num_classes is None:
        if model_type == 'gender_cnn':
            num_classes = 2
        elif model_type == 'age_cnn':
            num_classes = 3
        elif model_type == 'age_regression_cnn':
            num_classes = 1
        elif model_type == 'combined_cnn':
            num_classes = 6
        elif model_type == 'multi_output_cnn':
            num_classes = 2  # Will be overridden by separate heads
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # Create model using ModelFactory
    # ModelFactory.create_model expects keyword arguments matching model __init__
    model = ModelFactory.create_model(
        model_type,
        num_channels=num_channels,
        num_classes=num_classes,
        dropout_rate=0.5,
        use_layer_norm=True
    )
    
    return model, num_classes


# Configuration for all best models (for batch mode)
# TODO: Add user identification model configuration and resnet models configurations.
BEST_MODELS_CONFIG = [
    {
        'checkpoint': 'CNN/report_2025-12-16/results_1s/checkpoints/gender_baseline_1s_best.pth',
        'model_type': 'gender_cnn',
        'target_type': 'gender',
        'segment_length': '1s',
        'num_classes': 2,
        'prediction_type': 'classification',
        'class_names': ['Female', 'Male']
    },
    {
        'checkpoint': 'CNN/report_2025-12-16/results_1s/checkpoints/age_classification_1s_best.pth',
        'model_type': 'age_cnn',
        'target_type': 'age',
        'segment_length': '1s',
        'num_classes': 3,
        'prediction_type': 'classification',
        'class_names': ['<8.5 years', '8.5-12.5 years', '>12.5 years']
    },
    {
        'checkpoint': 'CNN/report_2025-12-16/results_2s/checkpoints/gender_baseline_2s_best.pth',
        'model_type': 'gender_cnn',
        'target_type': 'gender',
        'segment_length': '2s',
        'num_classes': 2,
        'prediction_type': 'classification',
        'class_names': ['Female', 'Male']
    },
    {
        'checkpoint': 'CNN/report_2025-12-16/results_2s/checkpoints/age_classification_2s_best.pth',
        'model_type': 'age_cnn',
        'target_type': 'age',
        'segment_length': '2s',
        'num_classes': 3,
        'prediction_type': 'classification',
        'class_names': ['<8.5 years', '8.5-12.5 years', '>12.5 years']
    },
    {
        'checkpoint': 'CNN/report_2025-12-16/results_2s/checkpoints/age_regression_2s_best.pth',
        'model_type': 'age_regression_cnn',
        'target_type': 'age',
        'segment_length': '2s',
        'num_classes': 1,
        'prediction_type': 'regression',
        'class_names': None
    },
    {
        'checkpoint': 'CNN/report_2025-12-16/results_4s/checkpoints/gender_baseline_4s_best.pth',
        'model_type': 'gender_cnn',
        'target_type': 'gender',
        'segment_length': '4s',
        'num_classes': 2,
        'prediction_type': 'classification',
        'class_names': ['Female', 'Male']
    },
    {
        'checkpoint': 'CNN/report_2025-12-16/results_4s/checkpoints/age_classification_4s_best.pth',
        'model_type': 'age_cnn',
        'target_type': 'age',
        'segment_length': '4s',
        'num_classes': 3,
        'prediction_type': 'classification',
        'class_names': ['<8.5 years', '8.5-12.5 years', '>12.5 years']
    },
    {
        'checkpoint': 'CNN/report_2025-12-16/results_4s/checkpoints/age_regression_4s_best.pth',
        'model_type': 'age_regression_cnn',
        'target_type': 'age',
        'segment_length': '4s',
        'num_classes': 1,
        'prediction_type': 'regression',
        'class_names': None
    },
]


def get_model_name_from_checkpoint(checkpoint_path: str) -> str:
    """Extract model name from checkpoint path."""
    return Path(checkpoint_path).stem.replace('_best', '')


def process_single_model(args):
    """Process a single model (original functionality)."""
    print("=" * 80)
    print("EEG Channel Importance Analysis via Saliency Maps")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Target type: {args.target_type}")
    print(f"Segment length: {args.segment_length}")
    print(f"Method: {args.method}")
    print("=" * 80)
    
    # Determine model type
    if args.model_type:
        model_type = args.model_type
    else:
        print("Auto-detecting model type from checkpoint...")
        model_type = get_model_type_from_checkpoint(args.checkpoint)
        print(f"Detected model type: {model_type}")
    
    # Create model
    print(f"\nCreating model: {model_type}")
    model, num_classes = create_model(model_type, args.num_channels, 
                                      prediction_type=args.prediction_type)
    
    # Setup device and multi-GPU if needed
    device = get_device(args.device)
    available_gpus = detect_available_gpus()
    
    if args.num_gpus > 1 and available_gpus > 0:
        print(f"\nSetting up multi-GPU ({args.num_gpus} GPUs)...")
        model = setup_model_for_gpus(model, args.num_gpus, device)
        print_gpu_info(args.num_gpus, args.num_gpus, device)
    else:
        if args.num_gpus > 1:
            print(f"\n⚠️  Requested {args.num_gpus} GPUs but only {available_gpus} available. Using single GPU/CPU.")
        model = model.to(device)
    
    # Create analyzer
    analyzer = SaliencyAnalyzer(
        model=model,
        target_type=args.target_type,
        num_channels=args.num_channels,
        device=device,
        device_preference=args.device,
        prediction_type=args.prediction_type
    )
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    analyzer.load_checkpoint(args.checkpoint)
    
    # Create data loader
    print(f"\nLoading data from: {args.hdf5_dir}")
    hdf5_dir = Path(args.hdf5_dir)
    
    # Get HDF5 file path(s) - handles both single files (2s/4s) and multiple files (1s)
    # Note: For 1s segments, this returns lists of files. For 2s/4s, returns single files.
    train_files, val_files, test_files = EEGDataLoader._get_hdf5_file_paths(
        hdf5_dir, args.segment_length
    )
    
    # Select the appropriate split
    if args.split == 'train':
        split_files = train_files
    elif args.split == 'val':
        split_files = val_files
    else:  # test
        split_files = test_files
    
    # Verify files exist (verify all splits to ensure data is available)
    EEGDataLoader._verify_hdf5_files(train_files, val_files, test_files)
    
    # Create dataset - handles both single files and lists of files automatically
    dataset = EEGDataLoader._create_dataset_from_hdf5(
        hdf5_file_or_files=split_files,
        task_type=args.task_type,
        target_type=args.target_type,
        transform=None,
        gender_transform=None,
        age_transform=None,
        combined_transform=None,
        user_identification_transform=None,
        shuffle=False  # Don't shuffle for consistent analysis
    )
    
    data_loader = EEGDataLoader.create_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=(device.type == 'cuda')
    )
    
    # Determine which methods to run
    if args.method == 'both':
        methods_to_run = ['vanilla_gradients', 'integrated_gradients']
    else:
        methods_to_run = [args.method]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine class names based on target type
    # Use stored class_names from batch mode if available, otherwise determine from target_type
    if hasattr(args, '_class_names') and args._class_names is not None:
        class_names = args._class_names
    elif args.target_type == 'gender':
        class_names = ['Female', 'Male']
    elif args.target_type == 'age':
        class_names = ['<8.5 years', '8.5-12.5 years', '>12.5 years']
    elif args.target_type == 'combined':
        class_names = [
            '(Female, <8.5)', '(Female, 8.5-12.5)', '(Female, >12.5)',
            '(Male, <8.5)', '(Male, 8.5-12.5)', '(Male, >12.5)'
        ]
    else:
        class_names = None
    
    # Process each method
    all_analyses = {}
    for method in methods_to_run:
        print(f"\n{'='*80}")
        print(f"Processing method: {method.upper()}")
        print(f"{'='*80}")
        
        # Compute saliency
        print(f"\nComputing saliency maps...")
        method_kwargs = {}
        if method == 'integrated_gradients':
            method_kwargs['num_steps'] = args.num_steps
        
        results = analyzer.compute_saliency(
            data_loader=data_loader,
            method=method,
            max_samples=args.max_samples,
            **method_kwargs
        )
        
        # Analyze channel importance
        print(f"\nAnalyzing channel importance...")
        analysis = analyzer.analyze_channel_importance(method=method)
        all_analyses[method] = analysis
        
        # Save results
        print(f"\nSaving results to: {output_dir}")
        analyzer.save_results(str(output_dir), method=method)
        
        # Generate visualizations
        if not args.no_plots:
            print(f"\nGenerating visualizations for {method}...")
            analyzer.visualize_results(
                output_dir=str(output_dir),
                method=method,
                channel_names=None,  # Can be extended with actual channel names
                class_names=class_names,
                top_k=args.top_k
            )
        
        # Print summary for this method
        print(f"\n{'-'*80}")
        print(f"Summary for {method.upper()}:")
        print(f"{'-'*80}")
        print(f"Top {args.top_k} most important channels:")
        top_channels = analyzer.get_top_channels(method=method, top_k=args.top_k)
        for rank, ch_idx in enumerate(top_channels, 1):
            avg_imp = analysis['average_importance'][ch_idx]
            print(f"  {rank:2d}. Channel {ch_idx+1:2d} (importance: {avg_imp:.6f})")
    
    # If both methods were run, generate comparison
    if args.method == 'both' and len(methods_to_run) == 2:
        print(f"\n{'='*80}")
        print("Generating Method Comparison")
        print(f"{'='*80}")
        
        # Get results for both methods
        vg_results = analyzer.results.get('vanilla_gradients')
        ig_results = analyzer.results.get('integrated_gradients')
        
        if vg_results is not None and ig_results is not None:
            # Generate comparison visualization
            if not args.no_plots:
                print("\nGenerating comparison visualizations...")
                plot_method_comparison(
                    channel_importance_vg=vg_results['channel_importance'],
                    channel_importance_ig=ig_results['channel_importance'],
                    channel_names=None,
                    title=f"Saliency Method Comparison ({args.target_type.capitalize()})",
                    save_path=str(output_dir / "method_comparison.png"),
                    top_k=args.top_k
                )
            
            # Generate comparison report
            print("\nGenerating comparison report...")
            save_method_comparison_report(
                channel_importance_vg=vg_results['channel_importance'],
                channel_importance_ig=ig_results['channel_importance'],
                channel_names=None,
                save_path=str(output_dir / "method_comparison_report.csv")
            )
        else:
            print("⚠️  Warning: Could not generate comparison - missing results for one or both methods")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    
    if args.method == 'both':
        print(f"\nMethods processed: {', '.join(methods_to_run)}")
        print(f"Comparison report: {output_dir / 'method_comparison_report.csv'}")
        if not args.no_plots:
            print(f"Comparison visualization: {output_dir / 'method_comparison.png'}")
    
    print("=" * 80)


def process_batch_models(args):
    """Process all best models in batch mode."""
    print("=" * 80)
    print("SALIENCY MAP COMPUTATION FOR ALL BEST MODELS")
    print("=" * 80)
    print(f"HDF5 directory: {args.hdf5_dir}")
    print(f"Device preference: {args.device}")
    print(f"Number of GPUs: {args.num_gpus}")
    print(f"Method: {args.method}")
    print(f"Max samples per model: {args.max_samples if args.max_samples else 'all'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")
    
    # Check GPU availability
    available_gpus = detect_available_gpus()
    print(f"\nAvailable GPUs: {available_gpus}")
    if args.num_gpus > 1 and available_gpus == 0:
        print(f"⚠️  Warning: Requested {args.num_gpus} GPUs but CUDA is not available.")
        print("   Will use CPU instead.")
        args.num_gpus = 1
        args.device = 'cpu'
    elif args.num_gpus > available_gpus:
        print(f"⚠️  Warning: Requested {args.num_gpus} GPUs but only {available_gpus} available.")
        print(f"   Will use {available_gpus} GPUs instead.")
        args.num_gpus = available_gpus
    
    # Filter models if specified
    models_to_process = BEST_MODELS_CONFIG
    if args.models:
        models_to_process = [
            config for config in BEST_MODELS_CONFIG
            if get_model_name_from_checkpoint(config['checkpoint']) in args.models
        ]
        if not models_to_process:
            print(f"\n❌ No matching models found for: {args.models}")
            print(f"   Available models: {[get_model_name_from_checkpoint(c['checkpoint']) for c in BEST_MODELS_CONFIG]}")
            return
    
    print(f"\nProcessing {len(models_to_process)} model(s)...")
    print("=" * 80)
    
    # Process each model
    successful = 0
    failed = 0
    
    for i, config in enumerate(models_to_process, 1):
        model_name = get_model_name_from_checkpoint(config['checkpoint'])
        print(f"\n[{i}/{len(models_to_process)}] Processing: {model_name}")
        print("-" * 80)
        
        if not Path(config['checkpoint']).exists():
            print(f"⚠️  Checkpoint not found: {config['checkpoint']}")
            print("   Skipping this model...")
            failed += 1
            continue
        
        try:
            # Create temporary args for this model
            import argparse
            model_args = argparse.Namespace(**vars(args))
            model_args.checkpoint = config['checkpoint']
            model_args.target_type = config['target_type']
            model_args.segment_length = config['segment_length']
            model_args.model_type = config['model_type']
            model_args.prediction_type = config['prediction_type']
            model_args.output_dir = str(Path(args.output_dir) / model_name)
            
            # Store class_names for visualization (will be used in process_single_model)
            model_args._class_names = config.get('class_names')
            
            # Process this model
            process_single_model(model_args)
            successful += 1
            
            # Clear CUDA cache after each model to prevent state corruption
            if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
        except Exception as e:
            print(f"\n❌ Error processing {model_name}:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            
            # Clear CUDA cache on error to reset state
            if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    # Summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total models processed: {len(models_to_process)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


def main():
    """Main function."""
    args = parse_args()
    
    # Validate arguments based on mode
    if args.batch:
        # Batch mode: process all best models
        process_batch_models(args)
    else:
        # Single model mode: validate required arguments
        if not args.checkpoint or not args.target_type or not args.segment_length:
            print("❌ Error: Single model mode requires --checkpoint, --target_type, and --segment_length")
            print("   Or use --batch to process all best models automatically")
            return
        process_single_model(args)


if __name__ == '__main__':
    main()

