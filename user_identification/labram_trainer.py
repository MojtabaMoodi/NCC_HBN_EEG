"""
Modified LaBraM Training Engine with ArcFace Loss Support

This module extends LaBraM's native training code to support ArcFace loss
for large-scale user identification tasks. It inherits and overrides key
functions from LaBraM's engine_for_finetuning.py while maintaining
LaBraM-specific optimizations like layer decay learning rate scheduling.
"""

import contextlib
import inspect
import math
import sys
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
import torch.nn as nn


def multiclass_macro_fpr_fnr(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Macro-averaged one-vs-rest FPR and FNR over classes that appear in y_true.

    For each class k: FPR_k = FP / (FP + TN), FNR_k = FN / (FN + TP) with the usual
    OvR confusion counts for that class.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.size == 0:
        return float("nan"), float("nan")
    classes = np.unique(y_true)
    fprs: List[float] = []
    fnrs: List[float] = []
    for k in classes:
        y_t = y_true == k
        y_p = y_pred == k
        tp = int(np.sum(y_t & y_p))
        fp = int(np.sum(~y_t & y_p))
        fn = int(np.sum(y_t & ~y_p))
        tn = int(np.sum(~y_t & ~y_p))
        d_fpr = fp + tn
        d_fnr = fn + tp
        fprs.append(float(fp / d_fpr) if d_fpr > 0 else 0.0)
        fnrs.append(float(fn / d_fnr) if d_fnr > 0 else 0.0)
    return float(np.mean(fprs)), float(np.mean(fnrs))


def _tensor_has_invalid_values(t: torch.Tensor) -> bool:
    """True for NaN or +inf. -inf is allowed (triplet masked prototype logits)."""
    return bool(torch.isnan(t).any() or torch.isposinf(t).any())


def _call_optional_input_chans(module: nn.Module, method_name: str, EEG: torch.Tensor, input_chans) -> torch.Tensor:
    """Call ``module.method_name(EEG, input_chans=...)`` only if the method accepts ``input_chans`` (LaBraM); else ``(EEG)`` (CNN/ResNet)."""
    fn = getattr(module, method_name)
    if "input_chans" in inspect.signature(fn).parameters:
        return fn(EEG, input_chans=input_chans)
    return fn(EEG)


def _forward_logits_optional_input_chans(module: nn.Module, EEG: torch.Tensor, input_chans) -> torch.Tensor:
    fn = module.forward
    if "input_chans" in inspect.signature(fn).parameters:
        return fn(EEG, input_chans=input_chans)
    return fn(EEG)


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

# Import embedding loss helpers
from user_identification.embedding_criterion_utils import resolve_loss_type, uses_embedding_logits
from CNN.models.triplet_loss import batch_local_nearest_prototype_accuracy


def train_class_batch_with_embedding_loss(
    model,
    samples,
    target,
    criterion,
    input_chans,
    loss_type=None,
    use_arcface=False,
):
    """
    Compute loss and logits for user-identification embedding training.

    For ArcFace / FaceNet triplet:
    - Extracts features from model (before classification head)
    - Computes metric-learning loss via criterion
    - Returns logits from criterion.compute_logits for accuracy

    For standard CE:
    - Uses direct model output logits
    """
    loss_type = resolve_loss_type(loss_type=loss_type, use_arcface=use_arcface)

    if uses_embedding_logits(loss_type):
        model_to_use = model.module if hasattr(model, "module") else model

        if hasattr(model_to_use, "extract_features"):
            features = model_to_use.extract_features(samples, input_chans=input_chans)
        elif hasattr(model_to_use, "forward_features"):
            features = model_to_use.forward_features(samples, input_chans=input_chans)
        else:
            raise AttributeError(
                "Model must have 'extract_features' or 'forward_features' for "
                f"loss_type={loss_type!r}"
            )

        if not hasattr(criterion, "compute_logits"):
            raise AttributeError(
                f"criterion for loss_type={loss_type!r} must implement compute_logits(); "
                f"got {type(criterion).__name__}"
            )

        loss = criterion(features, target)

        with torch.no_grad():
            if loss_type == "triplet":
                # FaceNet-pure: triplet loss only during training; prototypes are built
                # offline before val/test (refresh_prototypes_from_train_hdf5).
                triplet_batch_acc = batch_local_nearest_prototype_accuracy(features, target)
                outputs = None
            else:
                triplet_batch_acc = None
                outputs = criterion.compute_logits(features)
                outputs = outputs.float() if outputs.dtype == torch.float16 else outputs
    else:
        outputs = model(samples, input_chans=input_chans)
        loss = criterion(outputs, target)
        triplet_batch_acc = None

    return loss, outputs, triplet_batch_acc


def train_class_batch_with_arcface(model, samples, target, criterion, input_chans, use_arcface=False):
    """Backward-compatible alias for ArcFace / embedding batch training."""
    loss, output, _ = train_class_batch_with_embedding_loss(
        model, samples, target, criterion, input_chans,
        loss_type="arcface" if use_arcface else "ce",
        use_arcface=use_arcface,
    )
    return loss, output


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, ch_names=None, 
                    use_arcface=False, loss_type=None):
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
        use_arcface: Deprecated; use loss_type='arcface'. Kept for backward compatibility.
        loss_type: 'arcface', 'triplet', or 'ce'
    """
    resolved_loss_type = resolve_loss_type(loss_type=loss_type, use_arcface=use_arcface)
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
            loss, output, triplet_batch_acc = train_class_batch_with_embedding_loss(
                model, samples, targets, criterion, input_chans,
                loss_type=resolved_loss_type)
        else:
            with torch.amp.autocast(device_type='cuda'):
                loss, output, triplet_batch_acc = train_class_batch_with_embedding_loss(
                    model, samples, targets, criterion, input_chans,
                    loss_type=resolved_loss_type)

        # Check for NaN/+inf in output (-inf is valid for triplet masked logits)
        output_has_nan = False
        if output is not None:
            if isinstance(output, dict):
                for key, val in output.items():
                    if torch.is_tensor(val) and _tensor_has_invalid_values(val):
                        output_has_nan = True
                        break
            elif torch.is_tensor(output) and _tensor_has_invalid_values(output):
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
        if triplet_batch_acc is not None:
            class_acc = triplet_batch_acc
        else:
            class_acc = (output.max(-1)[-1] == targets.squeeze()).float().mean()
            
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        if resolved_loss_type == "triplet" and hasattr(criterion, "last_batch_stats"):
            tri_stats = criterion.last_batch_stats
            if tri_stats:
                metric_logger.update(
                    tri_active=tri_stats.get("active_triplet_fraction", 0.0)
                )
                if "mean_d_ap" in tri_stats:
                    metric_logger.update(tri_d_ap=tri_stats["mean_d_ap"])
                    metric_logger.update(tri_d_an=tri_stats["mean_d_an"])
        
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
    loss_meter = metric_logger.meters.get("loss")
    if loss_meter is not None and loss_meter.count == 0:
        raise RuntimeError(
            f"Epoch {epoch + 1}: no training batches completed successfully. "
            f"The DataLoader yielded {batch_count} batch(es) but metrics were never updated. "
            f"Check triplet P×K settings, data paths, and skipped-batch warnings above."
        )
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    if count_samples:
        actual_batch_size = getattr(data_loader, "batch_size", None)
        if actual_batch_size is None and hasattr(data_loader, "batch_sampler"):
            actual_batch_size = getattr(data_loader.batch_sampler, "batch_size", "P×K")
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
def evaluate(
    data_loader,
    model,
    device,
    header='Test:',
    ch_names=None,
    metrics=['accuracy'],
    use_arcface=False,
    criterion=None,
    collect_predictions: bool = False,
    eeg_input_mode: str = "labram",
    loss_type=None,
):
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
        collect_predictions: If True, include ``y_true`` and ``y_pred`` (NumPy int64) in the
            returned stats for the full loader (for saving per-sample labels). Callers should
            remove these before JSON serialization.
        eeg_input_mode: ``"labram"`` (default) scales by ``/100`` and rearranges to patch layout;
            ``"cnn"`` uses raw ``(B, C, T)`` tensors from ``EEGDataLoader`` (no LaBraM preprocessing).
        loss_type: ``'arcface'``, ``'triplet'``, or ``'ce'``. Overrides ``use_arcface`` when set.
    """
    resolved_loss_type = resolve_loss_type(loss_type=loss_type, use_arcface=use_arcface)
    if eeg_input_mode not in ("labram", "cnn"):
        raise ValueError(f"eeg_input_mode must be 'labram' or 'cnn', got {eeg_input_mode!r}")
    input_chans = None
    if ch_names is not None:
        input_chans = labram_utils.get_input_chans(ch_names)
    
    # CrossEntropyLoss for direct logits; embedding losses use criterion forward for eval loss.
    criterion_eval = torch.nn.CrossEntropyLoss() if not uses_embedding_logits(resolved_loss_type) else None

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
        
        EEG = EEG.float().to(device, non_blocking=True)
        if eeg_input_mode == "labram":
            EEG = EEG / 100
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
            model_to_use = model.module if hasattr(model, 'module') else model
            if uses_embedding_logits(resolved_loss_type):
                if criterion is None:
                    raise ValueError(
                        f"criterion is required when loss_type={resolved_loss_type!r}"
                    )
                if hasattr(model_to_use, 'extract_features'):
                    features = _call_optional_input_chans(
                        model_to_use, 'extract_features', EEG, input_chans
                    )
                elif hasattr(model_to_use, 'forward_features'):
                    features = _call_optional_input_chans(
                        model_to_use, 'forward_features', EEG, input_chans
                    )
                else:
                    raise AttributeError(
                        f"Model must have extract_features or forward_features for "
                        f"loss_type={resolved_loss_type!r}"
                    )

                if not hasattr(criterion, "compute_logits"):
                    raise AttributeError(
                        f"criterion must implement compute_logits for "
                        f"loss_type={resolved_loss_type!r}"
                    )
                output = criterion.compute_logits(features)
                output = output.float() if output.dtype == torch.float16 else output
                if resolved_loss_type == "triplet":
                    if hasattr(criterion, "classification_loss"):
                        loss = criterion.classification_loss(features, target)
                    else:
                        loss = torch.nn.functional.cross_entropy(
                            output, target.view(-1).long()
                        )
                else:
                    loss = criterion(features, target)
            else:
                output = _forward_logits_optional_input_chans(model_to_use, EEG, input_chans)
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
            if resolved_loss_type == "triplet":
                print(f"   ✅ Triplet (P×K) training expects low class diversity per batch")
            elif class_ratio > 0.5 and resolved_loss_type == "arcface":
                print(f"   ✅ High class diversity (ratio > 50%) - this is EXPECTED and CORRECT")
                print(f"   ✅ Indicates proper shuffling: samples from different users are mixed")
                if eeg_input_mode == "labram":
                    print(f"   ✅ With many user classes and large batch_size, high diversity is normal")
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

    # Full-set multiclass F1 (batch-wise F1 is not meaningful); complements accuracy from get_metrics.
    y_true = true.detach().cpu().numpy().ravel().astype("int64", copy=False)
    y_pred = torch.argmax(pred.detach().cpu(), dim=-1).numpy().ravel().astype("int64", copy=False)
    if y_true.size == 0:
        stats["f1_macro"] = float("nan")
        stats["f1_weighted"] = float("nan")
        stats["fpr_macro"] = float("nan")
        stats["fnr_macro"] = float("nan")
    else:
        stats["f1_macro"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0.0)
        )
        stats["f1_weighted"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0.0)
        )
        fpr_m, fnr_m = multiclass_macro_fpr_fnr(y_true, y_pred)
        stats["fpr_macro"] = fpr_m
        stats["fnr_macro"] = fnr_m

    if collect_predictions:
        stats["y_true"] = y_true.copy()
        stats["y_pred"] = y_pred.copy()

    return stats
