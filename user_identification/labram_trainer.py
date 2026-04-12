"""
Modified LaBraM Training Engine with ArcFace Loss Support

This module extends LaBraM's native training code to support ArcFace loss
for large-scale user identification tasks. It inherits and overrides key
functions from LaBraM's engine_for_finetuning.py while maintaining
LaBraM-specific optimizations like layer decay learning rate scheduling.
"""

import contextlib
import math
import sys
import warnings
from pathlib import Path
from typing import Iterable, Optional
import torch
import torch.nn as nn

# Suppress FutureWarning for autocast deprecation
warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")
# NOTE: We do NOT suppress the sklearn warning about many classes - we investigate it instead

# Add paths for imports
_project_root = Path(__file__).parent.parent
_labram_dir = _project_root / "LaBraM"

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_labram_dir) not in sys.path:
    sys.path.insert(0, str(_labram_dir))

# Import LaBraM modules
from timm.utils import ModelEma
import utils as labram_utils
from einops import rearrange

# Import LaBraM's original engine functions (we'll override specific functions)
# Import directly from LaBraM module
try:
    from LaBraM.engine_for_finetuning import (
        get_loss_scale_for_deepspeed,
        evaluate_by_task_type
    )
except ImportError:
    # Fallback: try importing from engine_for_finetuning directly
    import importlib.util
    engine_path = _labram_dir / "engine_for_finetuning.py"
    if engine_path.exists():
        spec = importlib.util.spec_from_file_location("engine_for_finetuning", engine_path)
        engine_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine_module)
        get_loss_scale_for_deepspeed = engine_module.get_loss_scale_for_deepspeed
        evaluate_by_task_type = getattr(engine_module, 'evaluate_by_task_type', None)
    else:
        # Minimal fallback - define what we need
        def get_loss_scale_for_deepspeed(model):
            optimizer = model.optimizer
            return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale
        evaluate_by_task_type = None

# Import ArcFace loss
from CNN.models.arcface_loss import ArcFaceLoss


def train_class_batch_with_arcface(model, samples, target, criterion, input_chans, use_arcface=False):
    """
    Modified train_class_batch that supports ArcFace loss.
    
    For ArcFace:
    - Extracts features from model (before classification head)
    - Computes loss using ArcFace (which handles logits internally)
    - Returns both loss and logits for accuracy calculation
    
    For standard loss:
    - Uses original LaBraM behavior (direct model output)
    
    Args:
        model: Model instance (should have extract_features method if use_arcface=True)
        samples: Input samples
        target: Target labels
        criterion: Loss function (ArcFaceLoss or standard)
        input_chans: Input channel indices
        use_arcface: Whether to use ArcFace loss
        
    Returns:
        loss: Computed loss
        outputs: Model outputs (logits for ArcFace, direct outputs for standard)
    """
    if use_arcface:
        # Extract features for ArcFace
        # Note: samples is already in LaBraM format [B, N, A, T] from rearrange
        # Handle DistributedDataParallel wrapper
        model_to_use = model.module if hasattr(model, 'module') else model
        
        if hasattr(model_to_use, 'extract_features'):
            # extract_features can handle both [B, N, T] and [B, N, A, T] formats
            features = model_to_use.extract_features(samples, input_chans=input_chans)
        else:
            # Fallback: try to get features from forward_features
            if hasattr(model_to_use, 'forward_features'):
                features = model_to_use.forward_features(samples, input_chans=input_chans)
            else:
                raise AttributeError(
                    "Model must have 'extract_features' or 'forward_features' method for ArcFace loss"
                )
        
        # ArcFace loss computes logits internally from normalized features
        loss = criterion(features, target)
        
        # For accuracy calculation, compute logits from ArcFace
        with torch.no_grad():
            outputs = criterion.compute_logits(features)
            # Convert to float32 for accuracy calculation
            outputs = outputs.float() if outputs.dtype == torch.float16 else outputs
    else:
        # Standard LaBraM behavior (for user identification, single output)
        outputs = model(samples, input_chans=input_chans)
        loss = criterion(outputs, target)
    
    return loss, outputs


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, ch_names=None, 
                    use_arcface=False):
    """
    Modified train_one_epoch that supports ArcFace loss for user identification.
    
    This function is based on LaBraM's train_one_epoch but modified to:
    1. Use ArcFace loss when use_arcface=True
    2. Extract features before computing loss for ArcFace
    3. Maintain all LaBraM-specific optimizations (layer decay LR, etc.)
    4. Focus solely on user identification (multi-class classification)
    
    Args:
        model: Model to train
        criterion: Loss function (ArcFaceLoss or standard)
        data_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        loss_scaler: Loss scaler for mixed precision
        max_norm: Maximum gradient norm for clipping
        model_ema: Optional EMA model
        log_writer: Optional log writer
        start_steps: Starting step number
        lr_schedule_values: Learning rate schedule values
        wd_schedule_values: Weight decay schedule values
        num_training_steps_per_epoch: Number of training steps per epoch
        update_freq: Gradient accumulation frequency
        ch_names: Channel names
        use_arcface: Whether to use ArcFace loss
    """
    skipped_batches = 0
    input_chans = None
    if ch_names is not None:
        input_chans = labram_utils.get_input_chans(ch_names)
    model.train(True)
    metric_logger = labram_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', labram_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', labram_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch + 1)  # Display as 1-indexed for user clarity
    print_freq = 500
    
    # Diagnostic: Count samples and batches in first epoch
    sample_count = 0
    batch_count = 0
    count_samples = (epoch == 0)

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    actual_steps = 0
    
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Handle both dictionary and tuple formats
        if isinstance(batch, dict):
            # Dictionary format from EEGDataLoader
            samples = batch['eeg_data']
            if 'user_identification' in batch:
                targets = batch['user_identification']
            else:
                raise ValueError("user_identification target not found in batch dictionary. "
                               "Ensure target_type='user_identification' when creating data loaders.")
        else:
            # Tuple format from LaBraM dataset: (samples, targets) or (samples, targets, participant_ids)
            samples = batch[0]
            targets = batch[1]
        
        if count_samples:
            sample_count += samples.shape[0]
            batch_count = data_iter_step + 1
        step = data_iter_step // update_freq
        if num_training_steps_per_epoch is not None and num_training_steps_per_epoch < 100000:
            if step >= num_training_steps_per_epoch:
                break
        actual_steps = step + 1
        it = start_steps + step
        
        # Update LR & WD
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # Check for NaN/Inf in input
        raw_samples = samples.float()
        if torch.isnan(raw_samples).any() or torch.isinf(raw_samples).any():
            print(f"WARNING: NaN/Inf found in RAW input samples at epoch {epoch}, step {data_iter_step}")
            skipped_batches += 1
            continue
        
        samples = raw_samples.to(device, non_blocking=True) / 100
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        
        if torch.isnan(samples).any() or torch.isinf(samples).any():
            print(f"WARNING: NaN/Inf introduced during preprocessing at epoch {epoch}, step {data_iter_step}")
            skipped_batches += 1
            continue
        
        # Handle targets (user identification - single class labels)
        targets = targets.to(device, non_blocking=True).long()

        # Compute loss and outputs
        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch_with_arcface(
                model, samples, targets, criterion, input_chans, use_arcface=use_arcface)
        else:
            with torch.amp.autocast(device_type='cuda'):
                loss, output = train_class_batch_with_arcface(
                    model, samples, targets, criterion, input_chans, use_arcface=use_arcface)

        # Check for NaN/Inf in output
        output_has_nan = False
        if isinstance(output, dict):
            for key, val in output.items():
                if torch.isnan(val).any() or torch.isinf(val).any():
                    output_has_nan = True
                    break
        elif torch.is_tensor(output) and (torch.isnan(output).any() or torch.isinf(output).any()):
            output_has_nan = True

        loss_value = loss.item()

        if not math.isfinite(loss_value) or output_has_nan:
            skipped_batches += 1
            if skipped_batches > 5:
                print(f"ERROR: Skipped {skipped_batches} batches in a row. Training unstable, stopping.")
                sys.exit(1)
            continue

        skipped_batches = 0

        # Backward pass
        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()
            if (data_iter_step + 1) % update_freq == 0:
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        # Compute accuracy (user identification - multi-class classification)
        # For ArcFace, output is already logits
        # For standard, output is logits
        class_acc = (output.max(-1)[-1] == targets.squeeze()).float().mean()
            
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
        
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    if count_samples:
        actual_batch_size = data_loader.batch_size
        print(f"\n{'='*60}")
        print(f"  DIAGNOSTIC: Epoch {epoch} Summary")
        print(f"{'='*60}")
        print(f"  Processed: {sample_count:,} samples in {batch_count:,} batches")
        print(f"  Batch size: {actual_batch_size}")
        print(f"{'='*60}\n")
    
    stats['skipped_batches'] = skipped_batches
    if skipped_batches > 0:
        print(f"Epoch {epoch} summary: Skipped {skipped_batches} batches due to NaN/Inf")
    
    return stats


@torch.no_grad()
def evaluate(data_loader, model, device, header='Test:', ch_names=None, metrics=['accuracy'], 
             use_arcface=False, criterion=None):
    """
    Modified evaluate function that supports ArcFace loss for user identification.
    
    Args:
        data_loader: Validation/test data loader
        model: Model to evaluate
        device: Device to evaluate on
        header: Header string for logging
        ch_names: Channel names
        metrics: Metrics to compute
        use_arcface: Whether using ArcFace loss
        criterion: Loss function (needed for ArcFace to compute logits)
    """
    input_chans = None
    if ch_names is not None:
        input_chans = labram_utils.get_input_chans(ch_names)
    
    # For user identification, we use CrossEntropyLoss if not using ArcFace
    criterion_eval = torch.nn.CrossEntropyLoss() if not use_arcface else None

    metric_logger = labram_utils.MetricLogger(delimiter="  ")
    model.eval()
    pred = []
    true = []
    
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # Handle both dictionary and tuple formats
        if isinstance(batch, dict):
            # Dictionary format from EEGDataLoader
            EEG = batch['eeg_data']
            if 'user_identification' in batch:
                target = batch['user_identification']
            else:
                raise ValueError("user_identification target not found in batch dictionary. "
                               "Ensure target_type='user_identification' when creating data loaders.")
        else:
            # Tuple format from LaBraM dataset: (EEG, target) or (EEG, target, participant_ids)
            EEG = batch[0]
            target = batch[1]
        
        EEG = EEG.float().to(device, non_blocking=True) / 100
        EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)
        
        # Handle targets (user identification - single class labels)
        target = target.to(device, non_blocking=True).long()
        
        # Compute output (autocast only on CUDA; CPU eval uses full precision)
        amp_ctx = (
            torch.amp.autocast(device_type="cuda")
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with amp_ctx:
            if use_arcface:
                # Extract features and compute logits via ArcFace
                # Handle DistributedDataParallel wrapper
                model_to_use = model.module if hasattr(model, 'module') else model
                
                if hasattr(model_to_use, 'extract_features'):
                    features = model_to_use.extract_features(EEG, input_chans=input_chans)
                else:
                    if hasattr(model_to_use, 'forward_features'):
                        features = model_to_use.forward_features(EEG, input_chans=input_chans)
                    else:
                        raise AttributeError("Model must have extract_features or forward_features for ArcFace")
                
                output = criterion.compute_logits(features)
                output = output.float() if output.dtype == torch.float16 else output
                loss = criterion(features, target)
            else:
                output = model(EEG, input_chans=input_chans)
                loss = criterion_eval(output, target)
        
        # Process outputs for metrics (user identification - multi-class)
        output = output.cpu()
        target = target.cpu()
        pred.append(output)
        true.append(target)

        batch_size = EEG.shape[0]
        metric_logger.update(loss=loss.item())
        
        # DIAGNOSTIC: Check batch class distribution (only log first few batches to avoid spam)
        if step < 5 and labram_utils.is_main_process():
            unique_classes = len(torch.unique(target))
            class_ratio = unique_classes / batch_size
            # Get class distribution statistics
            class_counts = torch.bincount(target.squeeze())
            class_counts = class_counts[class_counts > 0]  # Remove zeros
            max_samples_per_class = class_counts.max().item() if len(class_counts) > 0 else 0
            min_samples_per_class = class_counts.min().item() if len(class_counts) > 0 else 0
            mean_samples_per_class = class_counts.float().mean().item() if len(class_counts) > 0 else 0
            
            print(f"\n📊 Batch {step} class distribution diagnostic:")
            print(f"   Batch size: {batch_size}")
            print(f"   Unique classes in batch: {unique_classes} (ratio: {class_ratio:.2%})")
            print(f"   Samples per class: min={min_samples_per_class}, max={max_samples_per_class}, mean={mean_samples_per_class:.2f}")
            if class_ratio > 0.5:
                print(f"   ✅ High class diversity (ratio > 50%) - this is EXPECTED and CORRECT")
                print(f"   ✅ Indicates proper shuffling: samples from different users are mixed")
                print(f"   ✅ With 3145 classes and batch_size={batch_size}, high diversity is normal")
                print(f"   ⚠️  sklearn warning is harmless - metrics are computed correctly")
            else:
                print(f"   ⚠️  Low class diversity (ratio <= 50%) - may indicate clustering issue")
                print(f"   ⚠️  If this persists, check preprocessing shuffling")
        
        # Compute metrics per batch (user identification - multi-class classification)
        results = labram_utils.get_metrics(output.numpy(), target.numpy(), metrics, is_binary=False)
        for key, value in results.items():
            metric_logger.meters[key].update(value, n=batch_size)
    
    metric_logger.synchronize_between_processes()
    
    # Aggregate predictions
    pred = torch.cat(pred, dim=0)
    true = torch.cat(true, dim=0)
    
    # Compute final metrics (user identification - multi-class classification)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    results = labram_utils.get_metrics(pred.numpy(), true.numpy(), metrics, is_binary=False)
    for key, value in results.items():
        stats[key] = value
    
    return stats
