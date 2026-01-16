"""
Training Script for LaBraM User Identification with ArcFace Loss

This script uses LaBraM's native training infrastructure (layer decay LR, etc.)
but with ArcFace loss for large-scale user identification.
"""

import argparse
import sys
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import numpy as np

# Add paths
_project_root = Path(__file__).parent.parent
_labram_dir = _project_root / "LaBraM"

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_labram_dir) not in sys.path:
    sys.path.insert(0, str(_labram_dir))

# Import LaBraM utilities
from timm.utils import ModelEma
import utils as labram_utils
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

# Import our modified trainer
from user_identification.labram_trainer import train_one_epoch, evaluate

# Import ArcFace loss
from CNN.models.arcface_loss import ArcFaceLoss

# Import model and data
from user_identification.labram_model import LaBraMUserIdentificationWrapper
from data_processing.eeg_dataset import EEGDataLoader
from data_processing.target_transforms import create_user_identification_transform_from_hdf5


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('LaBraM User Identification with ArcFace', add_help=False)
    
    # Data parameters
    parser.add_argument('--hdf5_dir', type=str, required=True,
                       help='Directory containing HDF5 files')
    parser.add_argument('--segment_length', type=str, default='4s',
                       choices=['1s', '2s', '4s'],
                       help='EEG segment length')
    parser.add_argument('--batch_size', type=int, default=192,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loader workers')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='labram_base_patch200_200',
                       help='LaBraM model name')
    parser.add_argument('--pretrained_path', type=str, default=None,
                       help='Path to pre-trained LaBraM checkpoint')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--layer_decay', type=float, default=0.9,
                       help='Layer decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                       help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                       help='Warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                       help='Minimum learning rate')
    parser.add_argument('--clip_grad', type=float, default=0.2,
                       help='Gradient clipping norm')
    parser.add_argument('--update_freq', type=int, default=1,
                       help='Gradient accumulation frequency')
    
    # ArcFace parameters
    parser.add_argument('--arcface_margin', type=float, default=0.5,
                       help='ArcFace angular margin')
    parser.add_argument('--arcface_scale', type=float, default=256.0,
                       help='ArcFace scale parameter')
    parser.add_argument('--arcface_easy_margin', action='store_true',
                       help='Use easy margin for ArcFace')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./results_labram_arcface',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--save_ckpt_freq', type=int, default=5,
                       help='Save checkpoint frequency')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Distributed training parameters
    parser.add_argument('--distributed', action='store_true',
                       help='Enable distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                       help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int,
                       help='Local rank for distributed training')
    parser.add_argument('--dist_url', default='env://',
                       help='URL used to set up distributed training')
    parser.add_argument('--dist_on_itp', action='store_true',
                       help='Use ITP for distributed training')
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Initialize distributed training if enabled
    labram_utils.init_distributed_mode(args)
    
    # Set random seed (adjust for distributed training)
    seed = args.seed + labram_utils.get_rank() if args.distributed else args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    # Set device
    if args.distributed:
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device(args.device)
    
    # Create output directory (only on rank 0 for distributed training)
    if args.distributed:
        if labram_utils.is_main_process():
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()  # Wait for rank 0 to create directory
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load user identification transform and get number of classes
    # Only print on main process for distributed training
    if not args.distributed or labram_utils.is_main_process():
        transform, num_classes = create_user_identification_transform_from_hdf5(args.hdf5_dir)
        print(f"📊 User identification: {num_classes} user classes detected")
    else:
        # For distributed training, other ranks need to load transform too
        transform, num_classes = create_user_identification_transform_from_hdf5(args.hdf5_dir)
    if args.distributed:
        torch.distributed.barrier()  # Synchronize all processes
    
    # Create data loaders using LaBraM's dataset format
    if not args.distributed or labram_utils.is_main_process():
        print("Creating data loaders...")
    from LaBraM.labram_dataset import prepare_labram_dataset
    from torch.utils.data import DataLoader
    
    # Create LaBraM-compatible datasets (returns (X, Y) tuples)
    train_dataset, val_dataset, test_dataset = prepare_labram_dataset(
        dataset_type="user_identification",
        hdf5_dir=args.hdf5_dir,
        segment_length=args.segment_length,
        task_type="both",
        random_seed=args.seed,
        user_identification_transform=transform,
        sampling_rate=200
    )
    
    # For distributed training, IterableDataset handles sharding internally
    # No need for DistributedSampler (IterableDataset doesn't support it)
    if args.distributed:
        num_tasks = labram_utils.get_world_size()
        global_rank = labram_utils.get_rank()
        if labram_utils.is_main_process():
            print(f"Distributed training: {num_tasks} GPUs, rank {global_rank}")
    else:
        num_tasks = 1
        global_rank = 0
    
    # Create DataLoaders (LaBraM datasets return (X, Y) tuples, so no custom collate_fn needed)
    # Note: IterableDataset doesn't support DistributedSampler, sharding is handled in dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(1.5 * args.batch_size),  # Larger batch for validation
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(1.5 * args.batch_size),  # Larger batch for testing
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    if not args.distributed or labram_utils.is_main_process():
        print(f"✅ Data loaders created using LaBraM's dataset format (X, Y) tuples")
        if args.distributed:
            total_batch_size = args.batch_size * num_tasks * args.update_freq
            print(f"   Effective batch size: {total_batch_size} (batch_size={args.batch_size} × {num_tasks} GPUs × update_freq={args.update_freq})")
    
    # Create model
    if not args.distributed or labram_utils.is_main_process():
        print("Creating model...")
    model = LaBraMUserIdentificationWrapper(
        num_channels=60,
        num_classes=num_classes,
        dropout_rate=args.dropout_rate,
        pretrained_path=args.pretrained_path,
        model_name=args.model_name
    )
    model.to(device)
    
    # Wrap model with DistributedDataParallel for multi-GPU training
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu], 
            find_unused_parameters=True
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    # Get embedding dimension for ArcFace
    # Use model_without_ddp to access the underlying model's attributes
    if hasattr(model_without_ddp, 'fc2_intermediate'):
        embedding_dim = model_without_ddp.fc2_intermediate.out_features
    else:
        # Fallback: use embed_dim
        embedding_dim = model_without_ddp.embed_dim
    
    if not args.distributed or labram_utils.is_main_process():
        print(f"✅ Model created with {num_classes} classes")
        print(f"   Embedding dimension: {embedding_dim}")
    
    # Create ArcFace loss
    criterion = ArcFaceLoss(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        margin=args.arcface_margin,
        scale=args.arcface_scale,
        easy_margin=args.arcface_easy_margin
    )
    criterion = criterion.to(device)
    if not args.distributed or labram_utils.is_main_process():
        print(f"✅ ArcFace loss created: margin={args.arcface_margin}, scale={args.arcface_scale}")
    
    # Create optimizer with layer decay
    # Use model_without_ddp for optimizer creation (already set above)
    num_layers = model_without_ddp.labram_model.get_num_layers() if hasattr(model_without_ddp.labram_model, 'get_num_layers') else 12
    
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
        )
        if not args.distributed or labram_utils.is_main_process():
            print(f"✅ Using layer decay: {args.layer_decay}")
            print(f"   Assigned values: {assigner.values[:5]}... (showing first 5)")
    else:
        assigner = None
        if not args.distributed or labram_utils.is_main_process():
            print("✅ No layer decay (layer_decay=1.0)")
    
    skip_weight_decay_list = set()
    if hasattr(model_without_ddp.labram_model, 'no_weight_decay'):
        skip_weight_decay_list = model_without_ddp.labram_model.no_weight_decay()
    
    # Create optimizer
    class OptimArgs:
        def __init__(self):
            self.lr = args.lr
            self.weight_decay = args.weight_decay
            self.opt = 'adamw'
            self.opt_eps = 1e-8
            self.opt_betas = None
            self.momentum = 0.9
    
    optim_args = OptimArgs()
    optimizer = create_optimizer(
        optim_args, model_without_ddp,
        skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None
    )
    
    # Create loss scaler
    loss_scaler = labram_utils.NativeScalerWithGradNormCount()
    
    # Create LR and WD schedules
    # Estimate steps per epoch (use placeholder for IterableDataset)
    num_training_steps_per_epoch = 10000  # Placeholder for IterableDataset
    
    lr_schedule_values = labram_utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=-1
    )
    
    wd_schedule_values = labram_utils.cosine_scheduler(
        args.weight_decay, args.weight_decay, args.epochs, num_training_steps_per_epoch
    )
    
    if not args.distributed or labram_utils.is_main_process():
        print(f"✅ Optimizer and schedules created")
        print(f"   Max LR: {max(lr_schedule_values):.6f}, Min LR: {min(lr_schedule_values):.6f}")
    
    # Training loop
    if not args.distributed or labram_utils.is_main_process():
        print(f"\n{'='*80}")
        print(f"Starting training for {args.epochs} epochs")
        if args.distributed:
            print(f"Using {num_tasks} GPUs (distributed training)")
        print(f"{'='*80}\n")
    
    max_accuracy = 0.0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer,
            device, epoch, loss_scaler, args.clip_grad,
            model_ema=None, log_writer=None,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq,
            ch_names=None,  # Will be determined from model
            use_arcface=True
        )
        
        # Validate
        val_stats = evaluate(
            val_loader, model, device,
            header=f'Val Epoch {epoch+1}:',
            ch_names=None,
            metrics=['accuracy'],
            use_arcface=True,
            criterion=criterion
        )
        
        # Test
        test_stats = evaluate(
            test_loader, model, device,
            header=f'Test Epoch {epoch+1}:',
            ch_names=None,
            metrics=['accuracy'],
            use_arcface=True,
            criterion=criterion
        )
        
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        train_loss = train_stats.get('loss', 0.0)
        train_acc = train_stats.get('class_acc', 0.0) * 100
        val_loss = val_stats.get('loss', 0.0)
        val_acc = val_stats.get('accuracy', 0.0) * 100
        test_acc = test_stats.get('accuracy', 0.0) * 100
        
        # Only print and save on main process for distributed training
        if not args.distributed or labram_utils.is_main_process():
            print(f"  → Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f}% | Val Acc: {val_acc:.4f}% | "
                  f"Test Acc: {test_acc:.4f}% | LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")
        
        # Save checkpoint (only on main process)
        if not args.distributed or labram_utils.is_main_process():
            if val_acc > max_accuracy:
                max_accuracy = val_acc
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model_without_ddp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_scaler_state_dict': loss_scaler.state_dict(),
                    'arcface_state_dict': criterion.state_dict(),
                    'val_accuracy': val_acc,
                    'test_accuracy': test_acc,
                    'num_classes': num_classes,
                    'embedding_dim': embedding_dim
                }
                checkpoint_path = output_dir / 'best_model.pth'
                torch.save(checkpoint, checkpoint_path)
                print(f"  ✅ Best model saved (val_acc: {val_acc:.4f}%)")
            
            # Periodic checkpoint
            if (epoch + 1) % args.save_ckpt_freq == 0:
                checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pth'
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model_without_ddp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_scaler_state_dict': loss_scaler.state_dict(),
                    'arcface_state_dict': criterion.state_dict(),
                }
                torch.save(checkpoint, checkpoint_path)
    
    total_time = time.time() - start_time
    if not args.distributed or labram_utils.is_main_process():
        print(f"\n{'='*80}")
        print(f"Training completed!")
        print(f"   Total time: {total_time:.2f}s ({total_time/3600:.2f} hours)")
        print(f"   Best validation accuracy: {max_accuracy:.4f}%")
        print(f"{'='*80}")


if __name__ == '__main__':
    main()
