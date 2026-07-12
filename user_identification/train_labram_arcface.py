"""
Training Script for LaBraM User Identification with ArcFace or FaceNet Triplet Loss

This script uses LaBraM's native training infrastructure (layer decay LR, etc.)
with either ArcFace (default) or FaceNet-style triplet loss for user identification.
"""

import argparse
import json
import math
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

from user_identification.embedding_criterion_utils import (
    build_training_criterion,
    checkpoint_extra_state,
)
from user_identification.triplet_diagnostics import diagnose_embedding_separation
from CNN.models.triplet_loss import TripletTrainingCriterion

# Import model and data
from user_identification.labram_model import LaBraMUserIdentificationWrapper, iter_trainable_parameter_names
from data_processing.eeg_dataset import EEGDataLoader, EEGMapDataset, get_train_sample_list_user_identification
from data_processing.target_transforms import create_user_identification_transform_from_hdf5
from einops import rearrange
from torch.utils.data import DataLoader


def _serialize_training_tracker(
    *,
    loss_type: str,
    best_val_loss: float,
    patience_counter: int,
    max_accuracy: float,
    best_test_accuracy: float,
) -> dict:
    """Persist early-stopping tracker state in checkpoints for resume."""
    return {
        "patience_counter": patience_counter,
        "val_accuracy": max_accuracy,
        "test_accuracy": best_test_accuracy,
        **({"best_val_loss": best_val_loss} if loss_type == "triplet" else {}),
    }


def _restore_training_tracker_from_checkpoints(
    checkpoint: dict,
    output_dir: Path,
    loss_type: str,
) -> tuple[float, float, float, int]:
    """Restore best metrics and patience counter when resuming training."""
    max_accuracy = float(checkpoint.get("val_accuracy") or 0.0)
    best_test_accuracy = float(checkpoint.get("test_accuracy") or 0.0)
    patience_counter = int(checkpoint.get("patience_counter") or 0)

    if loss_type == "triplet":
        best_val_loss = checkpoint.get("best_val_loss")
        if best_val_loss is None and checkpoint.get("val_loss") is not None:
            best_val_loss = checkpoint["val_loss"]
        if best_val_loss is None:
            best_path = output_dir / "best_model.pth"
            if best_path.exists():
                best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
                if best_ckpt.get("best_val_loss") is not None:
                    best_val_loss = best_ckpt["best_val_loss"]
                elif best_ckpt.get("val_loss") is not None:
                    best_val_loss = best_ckpt["val_loss"]
                if best_ckpt.get("val_accuracy") is not None:
                    max_accuracy = float(best_ckpt["val_accuracy"])
                if best_ckpt.get("test_accuracy") is not None:
                    best_test_accuracy = float(best_ckpt["test_accuracy"])
                if best_ckpt.get("patience_counter") is not None:
                    patience_counter = int(best_ckpt["patience_counter"])
        best_val_loss = float(best_val_loss if best_val_loss is not None else float("inf"))
    else:
        best_val_loss = float("inf")

    return max_accuracy, best_test_accuracy, best_val_loss, patience_counter


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        'LaBraM User Identification (ArcFace or FaceNet Triplet)', add_help=False
    )
    
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
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze LaBraM (labram_model) and train only feature_projection '
                            '(embedding MLP before ArcFace). Recommended with --pretrained_path.')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
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
    
    # Loss selection
    parser.add_argument(
        '--loss_type',
        type=str,
        default='arcface',
        choices=['arcface', 'triplet'],
        help='Training loss: arcface (default) or facenet triplet',
    )

    # ArcFace parameters
    parser.add_argument('--arcface_margin', type=float, default=0.5,
                       help='ArcFace angular margin')
    parser.add_argument('--arcface_scale', type=float, default=256.0,
                       help='ArcFace scale parameter')
    parser.add_argument('--arcface_easy_margin', action='store_true',
                       help='Use easy margin for ArcFace')

    # FaceNet triplet parameters (used when --loss_type triplet)
    parser.add_argument('--triplet_margin', type=float, default=0.2,
                       help='Triplet margin alpha (FaceNet default: 0.2; try 0.5 if loss saturates)')
    parser.add_argument('--triplet_mining', type=str, default='semi_hard',
                       choices=['batch_hard', 'semi_hard', 'all'],
                       help='Triplet mining: semi_hard (FaceNet paper), batch_hard, or all')
    parser.add_argument('--triplet_classes_per_batch', type=int, default=32,
                       help='P in P×K triplet batching (identities per batch; FaceNet ~45)')
    parser.add_argument('--triplet_samples_per_class', type=int, default=12,
                       help='K in P×K triplet batching (segments per identity; FaceNet ~40)')
    parser.add_argument('--triplet_extra_negatives', type=int, default=0,
                       help='Random out-of-batch negative segments per batch (FaceNet paper; try 128 with batch P×K+128)')
    parser.add_argument('--prototype_momentum', type=float, default=0.9,
                       help='EMA momentum for online prototype updates during training')
    parser.add_argument('--triplet_logit_scale', type=float, default=10.0,
                       help='Scale for -distance prototype logits at eval')
    parser.add_argument('--triplet_batches_per_epoch', type=int, default=None,
                       help='PK mini-batches per epoch (default: train_samples//batch_size, max 10000)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./results_labram_arcface',
                       help='Directory to save all results: best_model.pth, checkpoint_epoch_*.pth, and logs')
    parser.add_argument('--save_ckpt_freq', type=int, default=5,
                       help='Save checkpoint frequency')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (e.g., best_model.pth or checkpoint_epoch_X.pth)')
    
    # Early stopping parameters
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Epochs without improvement before early stopping (0 to disable)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=1e-4,
                       help='Minimum improvement to reset patience: val accuracy (ArcFace) or '
                            'val loss decrease (triplet)')
    
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


@torch.no_grad()
def refresh_prototypes_from_train_hdf5(
    model: torch.nn.Module,
    hdf5_dir: str,
    segment_length: str,
    participant_id_to_class: dict,
    criterion: TripletTrainingCriterion,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> None:
    """
    FaceNet-pure eval: set class prototypes to mean L2-normalized train embeddings.

    Matches analyze_confidence.compute_centroids_from_embeddings used for open-set.
    """
    train_files, _, _ = EEGDataLoader._get_hdf5_file_paths(Path(hdf5_dir), segment_length)
    samples, class_indices, _ = get_train_sample_list_user_identification(
        train_files,
        participant_id_to_class=participant_id_to_class,
        task_type="both",
    )
    num_classes = criterion.prototype_classifier.num_classes
    embedding_dim = criterion.prototype_classifier.embedding_dim
    dataset = EEGMapDataset(
        samples=samples,
        target_type="user_identification",
        class_indices=class_indices,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=EEGDataLoader.collate_fn,
    )
    model.eval()
    model_to_use = model.module if hasattr(model, "module") else model
    sums = torch.zeros(num_classes, embedding_dim, device=device)
    counts = torch.zeros(num_classes, device=device, dtype=torch.long)

    for batch in loader:
        eeg = batch["eeg_data"].float().to(device, non_blocking=True) / 100
        eeg = rearrange(eeg, "B N (A T) -> B N A T", T=200)
        labels = batch["user_identification"].to(device, non_blocking=True).long()
        features = model_to_use.extract_features(eeg)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        for c in labels.unique():
            c_int = int(c.item())
            mask = labels == c_int
            sums[c_int] += features[mask].sum(dim=0)
            counts[c_int] += int(mask.sum().item())

    missing = (counts == 0).nonzero(as_tuple=True)[0].cpu().tolist()
    if missing:
        raise RuntimeError(
            f"Cannot compute train prototypes: {len(missing)} classes have zero "
            f"training samples in HDF5 index (e.g. {missing[:10]})."
        )
    centroids = sums / counts.unsqueeze(1).clamp(min=1).to(sums.dtype)
    centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)
    criterion.prototype_classifier.set_prototypes_from_numpy(centroids)
    model.train()


def main():
    args = get_args()

    if args.loss_type == "triplet":
        core_bs = args.triplet_classes_per_batch * args.triplet_samples_per_class
        expected_bs = core_bs + args.triplet_extra_negatives
        if args.batch_size != expected_bs:
            raise ValueError(
                f"For triplet loss, batch_size must equal P×K + extra_negatives = "
                f"{args.triplet_classes_per_batch}×{args.triplet_samples_per_class}"
                f"+{args.triplet_extra_negatives}={expected_bs}, "
                f"got batch_size={args.batch_size}. Adjust --batch_size, P/K, or extra_negatives."
            )
        if args.triplet_samples_per_class <= 1:
            raise ValueError(
                f"triplet_samples_per_class must be > 1, got {args.triplet_samples_per_class}"
            )
        if args.triplet_extra_negatives < 0:
            raise ValueError(
                f"triplet_extra_negatives must be >= 0, got {args.triplet_extra_negatives}"
            )
    
    # Initialize distributed training only when --distributed was passed (multi-GPU / SLURM).
    # When not passed, skip init so single-GPU runs work even if SLURM_PROCID etc. are set (e.g. salloc).
    if args.distributed:
        labram_utils.init_distributed_mode(args)
    else:
        # Ensure we do not enter distributed mode (LaBraM would otherwise use SLURM_PROCID / RANK)
        args.distributed = False
    
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
    
    participant_id_to_class = transform.participant_id_to_class_idx

    if args.loss_type == "triplet":
        train_loader = EEGDataLoader.create_pk_user_identification_train_loader(
            hdf5_dir=args.hdf5_dir,
            segment_length=args.segment_length,
            participant_id_to_class=participant_id_to_class,
            classes_per_batch=args.triplet_classes_per_batch,
            samples_per_class=args.triplet_samples_per_class,
            task_type="both",
            num_workers=args.num_workers,
            random_seed=args.seed,
            batches_per_epoch=args.triplet_batches_per_epoch,
            extra_negatives=args.triplet_extra_negatives,
        )
        _, val_dataset, test_dataset = prepare_labram_dataset(
            dataset_type="user_identification",
            hdf5_dir=args.hdf5_dir,
            segment_length=args.segment_length,
            task_type="both",
            random_seed=args.seed,
            user_identification_transform=transform,
            sampling_rate=200,
        )
    else:
        train_dataset, val_dataset, test_dataset = prepare_labram_dataset(
            dataset_type="user_identification",
            hdf5_dir=args.hdf5_dir,
            segment_length=args.segment_length,
            task_type="both",
            random_seed=args.seed,
            user_identification_transform=transform,
            sampling_rate=200,
        )
        train_loader = None
    
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
    if args.loss_type != "triplet":
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
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
        print(f"   Loss type: {args.loss_type}")
        if args.loss_type == "triplet":
            print(
                f"   Triplet train loader: P={args.triplet_classes_per_batch}, "
                f"K={args.triplet_samples_per_class}, extra_neg={args.triplet_extra_negatives}, "
                f"batch_size={args.batch_size}, batches/epoch={len(train_loader)}"
            )
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
        model_name=args.model_name,
        freeze_backbone=args.freeze_backbone,
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
        if args.freeze_backbone:
            n_trainable = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in model_without_ddp.parameters())
            print(f"   freeze_backbone=True: trainable params {n_trainable:,} / {n_total:,}")
            if args.pretrained_path is None:
                print(
                    "   Warning: --freeze_backbone without --pretrained_path: "
                    "the backbone stays at random init while only the projection trains."
                )
            trainable_names = list(iter_trainable_parameter_names(model_without_ddp))
            if len(trainable_names) <= 20:
                for n in trainable_names:
                    print(f"      trainable: {n}")
            else:
                print(f"      ({len(trainable_names)} trainable tensors; names omitted)")
    
    # Create training criterion (ArcFace or FaceNet triplet)
    criterion = build_training_criterion(
        args.loss_type,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale,
        arcface_easy_margin=args.arcface_easy_margin,
        triplet_margin=args.triplet_margin,
        triplet_mining=args.triplet_mining,
        prototype_momentum=args.prototype_momentum,
        triplet_logit_scale=args.triplet_logit_scale,
        device=device,
    )
    if not args.distributed or labram_utils.is_main_process():
        if args.loss_type == "arcface":
            print(f"✅ ArcFace loss created: margin={args.arcface_margin}, scale={args.arcface_scale}")
        else:
            print(
                f"✅ FaceNet triplet loss: margin={args.triplet_margin}, mining={args.triplet_mining}, "
                f"P×K={args.triplet_classes_per_batch}×{args.triplet_samples_per_class}"
            )
    
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
    
    # Steps per epoch: ArcFace IterableDataset is capped at 10k; triplet PK must match
    # len(train_loader) so the cosine LR schedule stays aligned with actual updates.
    if args.loss_type == "triplet":
        num_training_steps_per_epoch = len(train_loader)
        if num_training_steps_per_epoch <= 0:
            raise RuntimeError(
                "Triplet PK train loader has zero batches per epoch. "
                "Check P×K settings and training data."
            )
    else:
        num_training_steps_per_epoch = 10000  # Placeholder cap for IterableDataset
    
    lr_schedule_values = labram_utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=-1
    )
    
    wd_schedule_values = labram_utils.cosine_scheduler(
        args.weight_decay, args.weight_decay, args.epochs, num_training_steps_per_epoch
    )
    
    if not args.distributed or labram_utils.is_main_process():
        print(f"✅ Optimizer and schedules created")
        print(f"   Steps per epoch: {num_training_steps_per_epoch}")
        print(f"   Max LR: {max(lr_schedule_values):.6f}, Min LR: {min(lr_schedule_values):.6f}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    max_accuracy = 0.0
    best_test_accuracy = 0.0
    best_val_loss = float("inf")
    last_separation_ratio = float("nan")
    patience_counter = 0  # For early stopping (tracks epochs without improvement)
    if args.resume:
        if not args.distributed or labram_utils.is_main_process():
            print(f"🔍 Resume argument received: {args.resume}")
            print(f"🔍 Output directory: {output_dir}")
            print(f"🔍 Checking for checkpoint in: {output_dir / args.resume}")
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            # If relative path, check multiple locations:
            # 1. First try: in output_dir (most common case - just filename like "best_model.pth")
            if (output_dir / resume_path).exists():
                resume_path = output_dir / resume_path
            # 2. Second try: as-is in current directory
            elif Path(resume_path).exists():
                resume_path = Path(resume_path)
            # 3. Third try: if path contains output_dir name, extract just filename
            elif (output_dir / resume_path.name).exists():
                resume_path = output_dir / resume_path.name
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found: {args.resume}\n"
                    f"  Checked: {output_dir / resume_path}\n"
                    f"  Checked: {Path(resume_path)}\n"
                    f"  Checked: {output_dir / resume_path.name}"
                )
        
        if not args.distributed or labram_utils.is_main_process():
            print(f"📂 Resuming from checkpoint: {resume_path}")
        
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        
        # Load model state
        model_without_ddp.load_state_dict(checkpoint['model_state_dict'])
        if args.freeze_backbone:
            model_without_ddp._freeze_labram_backbone()
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load loss scaler state if available
        if 'loss_scaler_state_dict' in checkpoint and loss_scaler is not None:
            loss_scaler.load_state_dict(checkpoint['loss_scaler_state_dict'])
        
        # Load loss-specific state if available
        ckpt_loss_type = checkpoint.get("loss_type", "arcface")
        if ckpt_loss_type != args.loss_type:
            raise ValueError(
                f"Checkpoint loss_type={ckpt_loss_type!r} does not match --loss_type={args.loss_type!r}"
            )
        if args.loss_type == "arcface":
            if "arcface_state_dict" not in checkpoint:
                raise KeyError("ArcFace resume checkpoint missing 'arcface_state_dict'")
            criterion.load_state_dict(checkpoint["arcface_state_dict"])
        else:
            if "triplet_criterion_state_dict" in checkpoint:
                criterion.load_state_dict(checkpoint["triplet_criterion_state_dict"])
            elif "prototype_state_dict" in checkpoint:
                criterion.prototype_classifier.load_state_dict(
                    checkpoint["prototype_state_dict"]
                )
            else:
                raise KeyError(
                    "Triplet resume checkpoint missing 'triplet_criterion_state_dict' "
                    "or 'prototype_state_dict'"
                )
        
        # Get starting epoch and max accuracy
        # Note: checkpoint saves 'epoch' as 1-indexed (epoch + 1 after completing the epoch)
        # Training loop uses 0-indexed epochs (range(0, args.epochs))
        # If checkpoint has epoch 50 (1-indexed), it means we completed epoch 49 (0-indexed)
        # So we should start from epoch 50 (0-indexed) = epoch 51 (1-indexed)
        checkpoint_epoch_1indexed = start_epoch
        if 'epoch' in checkpoint:
            checkpoint_epoch_1indexed = checkpoint['epoch']  # 1-indexed: last completed epoch + 1
            # Convert to 0-indexed: if checkpoint says 50 (completed epoch 49), start from 50
            start_epoch = checkpoint_epoch_1indexed  # Keep as-is: checkpoint epoch 50 means start from 50 (0-indexed)
        max_accuracy, best_test_accuracy, best_val_loss, patience_counter = (
            _restore_training_tracker_from_checkpoints(checkpoint, output_dir, args.loss_type)
        )
        
        # Warn if resuming from last epoch (no more training will happen)
        if start_epoch >= args.epochs:
            if not args.distributed or labram_utils.is_main_process():
                print(f"   ⚠️  WARNING: Checkpoint is from epoch {start_epoch}, but --epochs={args.epochs}")
                print(f"   ⚠️  No training will occur. Increase --epochs to continue training.")
        
        if not args.distributed or labram_utils.is_main_process():
            print(f"   ✅ Loaded checkpoint from epoch {checkpoint_epoch_1indexed} (completed)")
            print(f"   ✅ Best validation accuracy so far: {max_accuracy:.4f}%")
            if args.loss_type == "triplet" and math.isfinite(best_val_loss):
                print(f"   ✅ Best validation loss so far: {best_val_loss:.4f}")
            if args.early_stopping_patience > 0:
                print(
                    f"   ✅ Early stopping patience counter: {patience_counter}/"
                    f"{args.early_stopping_patience}"
                )
            if start_epoch < args.epochs:
                print(f"   ✅ Continuing training from epoch {start_epoch + 1} to {args.epochs}")
            else:
                print(f"   ⚠️  Already completed {args.epochs} epochs. Increase --epochs to continue.")
    
    # Training loop
    if not args.distributed or labram_utils.is_main_process():
        print(f"\n{'='*80}")
        if args.resume:
            print(f"Continuing training from epoch {start_epoch + 1} to {args.epochs}")
        else:
            print(f"Starting training for {args.epochs} epochs")
        if args.distributed:
            print(f"Using {num_tasks} GPUs (distributed training)")
        print(f"{'='*80}\n")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        if args.loss_type == "triplet":
            pk_sampler = getattr(train_loader, "batch_sampler", None)
            if pk_sampler is not None and hasattr(pk_sampler, "set_epoch"):
                pk_sampler.set_epoch(epoch)
        
        # Train (epoch is 0-indexed in the loop, but we display as 1-indexed)
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
            loss_type=args.loss_type,
        )

        if args.loss_type == "triplet":
            if not args.distributed or labram_utils.is_main_process():
                print("  Refreshing FaceNet prototypes from full train split...")
            refresh_prototypes_from_train_hdf5(
                model,
                args.hdf5_dir,
                args.segment_length,
                participant_id_to_class,
                criterion,
                device,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                seed=args.seed,
            )
        
        # Validate
        val_stats = evaluate(
            val_loader, model, device,
            header=f'Val Epoch {epoch+1}:',
            ch_names=None,
            metrics=['accuracy'],
            criterion=criterion,
            loss_type=args.loss_type,
        )
        
        # Test
        test_stats = evaluate(
            test_loader, model, device,
            header=f'Test Epoch {epoch+1}:',
            ch_names=None,
            metrics=['accuracy'],
            criterion=criterion,
            loss_type=args.loss_type,
        )
        
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        train_loss = train_stats.get('loss', 0.0)
        train_acc = train_stats.get('class_acc', 0.0) * 100
        val_loss = val_stats.get('loss', 0.0)
        val_acc = val_stats.get('accuracy', 0.0) * 100
        test_acc = test_stats.get('accuracy', 0.0) * 100
        tri_active = train_stats.get('tri_active', None)
        tri_d_ap = train_stats.get('tri_d_ap', None)
        tri_d_an = train_stats.get('tri_d_an', None)

        separation_stats = None
        if args.loss_type == "triplet" and (
            not args.distributed or labram_utils.is_main_process()
        ):
            separation_stats = diagnose_embedding_separation(
                model, val_loader, device,
                max_batches=20,
                header="Val embedding separation",
            )
            last_separation_ratio = separation_stats["inter_over_intra_ratio"]
        
        # Only print and save on main process for distributed training
        if not args.distributed or labram_utils.is_main_process():
            line = (
                f"  → Epoch {epoch+1:3d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.4f}% | Val Acc: {val_acc:.4f}% | "
                f"Test Acc: {test_acc:.4f}% | LR: {current_lr:.6f} | Time: {epoch_time:.2f}s"
            )
            if tri_active is not None:
                line += f" | tri_active: {tri_active:.3f}"
            if tri_d_ap is not None and tri_d_an is not None:
                line += f" | d_ap: {tri_d_ap:.3f} d_an: {tri_d_an:.3f}"
            if separation_stats is not None:
                line += f" | inter/intra: {separation_stats['inter_over_intra_ratio']:.3f}"
            print(line)
        
        # Save checkpoint and check for early stopping (only on main process)
        if not args.distributed or labram_utils.is_main_process():
            if args.loss_type == "triplet":
                # Triplet: val loss (prototype CE) is a more stable early-stop signal than ~0% val acc
                improved = val_loss < (best_val_loss - args.early_stopping_min_delta)
                if improved:
                    best_val_loss = val_loss
                    max_accuracy = val_acc
                    best_test_accuracy = test_acc
                    patience_counter = 0
                    checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': model_without_ddp.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_scaler_state_dict': loss_scaler.state_dict(),
                        'val_accuracy': val_acc,
                        'test_accuracy': test_acc,
                        'val_loss': val_loss,
                        'num_classes': num_classes,
                        'embedding_dim': embedding_dim,
                        'freeze_backbone': bool(args.freeze_backbone),
                    }
                    if separation_stats is not None:
                        checkpoint['embedding_separation'] = separation_stats
                    checkpoint.update(
                        _serialize_training_tracker(
                            loss_type=args.loss_type,
                            best_val_loss=best_val_loss,
                            patience_counter=patience_counter,
                            max_accuracy=max_accuracy,
                            best_test_accuracy=best_test_accuracy,
                        )
                    )
                    checkpoint.update(checkpoint_extra_state(args.loss_type, criterion))
                    checkpoint_path = output_dir / 'best_model.pth'
                    torch.save(checkpoint, checkpoint_path)
                    print(
                        f"  ✅ Best model saved (val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}%)"
                    )
                else:
                    patience_counter += 1
                    if args.early_stopping_patience > 0:
                        print(
                            f"  ⏳ No val_loss improvement for {patience_counter}/"
                            f"{args.early_stopping_patience} epochs "
                            f"(val_loss: {val_loss:.4f}, best: {best_val_loss:.4f})"
                        )
            else:
                improvement_threshold = max_accuracy + args.early_stopping_min_delta
                if val_acc > improvement_threshold:
                    improvement = val_acc - max_accuracy
                    max_accuracy = val_acc
                    best_test_accuracy = test_acc
                    patience_counter = 0
                    checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': model_without_ddp.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_scaler_state_dict': loss_scaler.state_dict(),
                        'val_accuracy': val_acc,
                        'test_accuracy': test_acc,
                        'num_classes': num_classes,
                        'embedding_dim': embedding_dim,
                        'freeze_backbone': bool(args.freeze_backbone),
                    }
                    checkpoint.update(checkpoint_extra_state(args.loss_type, criterion))
                    checkpoint_path = output_dir / 'best_model.pth'
                    torch.save(checkpoint, checkpoint_path)
                    print(f"  ✅ Best model saved (val_acc: {val_acc:.4f}%, improvement: {improvement:.4f}%)")
                else:
                    patience_counter += 1
                    if args.early_stopping_patience > 0:
                        print(f"  ⏳ No improvement for {patience_counter}/{args.early_stopping_patience} epochs (val_acc: {val_acc:.4f}%, best: {max_accuracy:.4f}%)")
            
            # Early stopping check
            if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                print(f"\n{'='*80}")
                print(f"⏹️  Early stopping triggered at epoch {epoch+1}")
                print(f"   No improvement for {patience_counter} epochs")
                if args.loss_type == "triplet":
                    print(f"   Best validation loss: {best_val_loss:.4f}")
                    print(f"   Best validation accuracy: {max_accuracy:.4f}%")
                    if not math.isnan(last_separation_ratio):
                        print(f"   Last inter/intra ratio: {last_separation_ratio:.4f}")
                else:
                    print(f"   Best validation accuracy: {max_accuracy:.4f}%")
                print(f"{'='*80}\n")
                break
            
            # Periodic checkpoint
            if (epoch + 1) % args.save_ckpt_freq == 0:
                checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pth'
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model_without_ddp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_scaler_state_dict': loss_scaler.state_dict(),
                    'freeze_backbone': bool(args.freeze_backbone),
                }
                checkpoint.update(
                    _serialize_training_tracker(
                        loss_type=args.loss_type,
                        best_val_loss=best_val_loss,
                        patience_counter=patience_counter,
                        max_accuracy=max_accuracy,
                        best_test_accuracy=best_test_accuracy,
                    )
                )
                checkpoint.update(checkpoint_extra_state(args.loss_type, criterion))
                torch.save(checkpoint, checkpoint_path)
    
    total_time = time.time() - start_time
    if not args.distributed or labram_utils.is_main_process():
        results = {
            'val_accuracy': float(max_accuracy),
            'test_accuracy': float(best_test_accuracy),
            'freeze_backbone': bool(args.freeze_backbone),
            'loss_type': args.loss_type,
        }
        if args.loss_type == "triplet":
            results['best_val_loss'] = float(best_val_loss)
            results['triplet_mining'] = args.triplet_mining
            results['triplet_margin'] = float(args.triplet_margin)
            if not math.isnan(last_separation_ratio):
                results['last_inter_over_intra_ratio'] = float(last_separation_ratio)
        results_path = output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n{'='*80}")
        print(f"Training completed!")
        print(f"   Total time: {total_time:.2f}s ({total_time/3600:.2f} hours)")
        print(f"   Best validation accuracy: {max_accuracy:.4f}%")
        print(f"   Test accuracy (at best val): {best_test_accuracy:.4f}%")
        print(f"   Results written to: {results_path}")
        print(f"{'='*80}")


if __name__ == '__main__':
    main()
