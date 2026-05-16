# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable, Optional
import numpy as np
import torch

# Suppress FutureWarning for autocast deprecation (may come from PyTorch internals)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")
from timm.utils import ModelEma
import utils
from einops import rearrange

# Set DEBUG_EVALUATE=1 to print batch-structure debug in evaluate()
_DEBUG_EVALUATE = os.environ.get("DEBUG_EVALUATE", "").lower() in ("1", "true", "yes")

# Add LaBraM directory and project root for imports (labram_dataset, data_processing)
_labram_dir = Path(__file__).parent
_project_root = _labram_dir.parent
if str(_labram_dir) not in sys.path:
    sys.path.insert(0, str(_labram_dir))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import LaBraM-specific modules
from labram_dataset import prepare_labram_dataset, collate_labram_with_participant_ids

# Reuse participant-level aggregation from data_processing (same as CNN)
try:
    from data_processing.aggregation import (
        aggregate_participant_level_classification,
        aggregate_predictions_by_group,
        AGGREGATION_MAJORITY_VOTE,
        AGGREGATION_MEAN_PROB,
        AGGREGATION_MEDIAN,
        AGGREGATION_MEAN,
    )
except ImportError:
    aggregate_participant_level_classification = None
    aggregate_predictions_by_group = None
    AGGREGATION_MAJORITY_VOTE = None
    AGGREGATION_MEAN_PROB = None
    AGGREGATION_MEDIAN = None
    AGGREGATION_MEAN = None

def train_class_batch(model, samples, target, criterion, ch_names):
    outputs = model(samples, ch_names)
    
    # Handle multi-output models
    if isinstance(outputs, dict):
        # Multi-output: target is (gender_labels, age_labels) tuple
        gender_labels, age_labels = target
        gender_loss = criterion(outputs['gender'], gender_labels)
        age_loss = criterion(outputs['age'], age_labels)
        loss = gender_loss + age_loss  # Combined loss
    else:
        # Single-output model
        loss = criterion(outputs, target)
    
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, ch_names=None, is_binary=True,
                    is_regression=False):
    skipped_batches = 0  # Track batches skipped due to NaN/Inf
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500
    
    # Diagnostic: Count samples and batches in first epoch to verify dataset size
    sample_count = 0
    batch_count = 0
    count_samples = (epoch == 0)  # Only count in first epoch
    task_type_counts = {'active': 0, 'passive': 0} if count_samples else None

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    # Track actual steps for IterableDataset (unknown size)
    # For IterableDataset, we let the iterator naturally exhaust
    # The step limit check is only for safety if num_training_steps_per_epoch is set to a reasonable value
    actual_steps = 0
    
    # For first epoch, count task types by iterating through underlying dataset separately
    # This creates a new iterator, so it won't interfere with the training loop
    if count_samples and hasattr(data_loader.dataset, 'eeg_dataset'):
        print(f"  Counting task types in training dataset...")
        eeg_dataset = data_loader.dataset.eeg_dataset
        # Create a new iterator (IterableDataset creates new iterator each time)
        for sample in eeg_dataset:
            task_type = sample.get('task_type', 'unknown')
            if task_type in task_type_counts:
                task_type_counts[task_type] += 1
        print(f"  Training dataset: {task_type_counts['active']:,} active, {task_type_counts['passive']:,} passive samples")
    
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = batch[0]
        targets = batch[1]  # batch may be (eeg, label) or (eeg, label, participant_ids)
        # Count samples and batches in first epoch for diagnostic
        if count_samples:
            sample_count += samples.shape[0]
            batch_count = data_iter_step + 1  # Current batch index (0-indexed, so +1)
        step = data_iter_step // update_freq
        # Only apply limit if num_training_steps_per_epoch is set to a reasonable value (< 100000)
        # For IterableDataset with placeholder value (10000), we ignore the limit and let iterator exhaust
        if num_training_steps_per_epoch is not None and num_training_steps_per_epoch < 100000:
            if step >= num_training_steps_per_epoch:
                break  # Use break instead of continue to exit the loop
        actual_steps = step + 1  # Track actual steps taken
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # Check for NaN/Inf in raw input BEFORE any preprocessing
        # This is a safety net - dataset should catch NaN/Inf and raise ValueError
        # If we reach here, it means either:
        # 1. NaN/Inf was introduced during tensor conversion (unlikely but possible)
        # 2. Dataset check was bypassed somehow
        # Best practice: Skip the batch and log warning, but this should rarely happen
        raw_samples = samples.float()
        if torch.isnan(raw_samples).any() or torch.isinf(raw_samples).any():
            print(f"WARNING: NaN/Inf found in RAW input samples (from dataset) at epoch {epoch}, step {data_iter_step}")
            print(f"  This should have been caught by dataset validation. Skipping batch.")
            print(f"  NaN count: {torch.isnan(raw_samples).sum().item()}, Inf count: {torch.isinf(raw_samples).sum().item()}")
            print(f"  Raw sample shape: {raw_samples.shape}")
            print(f"  Raw sample stats: min={raw_samples.min().item():.6f}, max={raw_samples.max().item():.6f}, mean={raw_samples.mean().item():.6f}")
            # Find which samples in the batch have NaN/Inf
            nan_mask = torch.isnan(raw_samples).any(dim=(1, 2)) | torch.isinf(raw_samples).any(dim=(1, 2))
            nan_indices = torch.where(nan_mask)[0].cpu().tolist()
            print(f"  Samples with NaN/Inf in batch: {nan_indices} (out of {raw_samples.shape[0]} samples)")
            print(f"  ⚠️  This indicates a data quality issue. Please check preprocessing pipeline.")
            # Skip this batch (safety net - dataset should have caught this)
            skipped_batches += 1
            continue
        
        samples = raw_samples.to(device, non_blocking=True) / 100
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        
        # Check for NaN/Inf after preprocessing (division and reshape)
        if torch.isnan(samples).any() or torch.isinf(samples).any():
            print(f"WARNING: NaN/Inf introduced during preprocessing at epoch {epoch}, step {data_iter_step}")
            print(f"  NaN count: {torch.isnan(samples).sum().item()}, Inf count: {torch.isinf(samples).sum().item()}")
            print(f"  Sample stats after preprocessing: min={samples.min().item():.6f}, max={samples.max().item():.6f}, mean={samples.mean().item():.6f}")
            # Skip this batch
            continue
        
        # Handle multi-output targets (tuple of (gender, age))
        is_multi_output = isinstance(targets, (list, tuple)) and len(targets) == 2
        if is_multi_output:
            gender_targets = targets[0].to(device, non_blocking=True).long()
            age_targets = targets[1].to(device, non_blocking=True).long()
            targets = (gender_targets, age_targets)
        else:
            targets = targets.to(device, non_blocking=True)
            if is_regression:
                targets = targets.float()
                if targets.dim() == 1:
                    targets = targets.unsqueeze(-1)
            elif is_binary:
                targets = targets.float().unsqueeze(-1)
        
        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion, input_chans)
        else:
            if is_regression:
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    loss, output = train_class_batch(
                        model, samples, targets, criterion, input_chans)
            else:
                with torch.amp.autocast(device_type='cuda'):
                    loss, output = train_class_batch(
                        model, samples, targets, criterion, input_chans)

        # Check for NaN/Inf in model output before computing loss
        output_has_nan = False
        if isinstance(output, dict):
            for key, val in output.items():
                if torch.isnan(val).any() or torch.isinf(val).any():
                    print(f"WARNING: NaN/Inf in model output '{key}' at epoch {epoch}, step {data_iter_step} - skipping batch")
                    print(f"  Output stats: min={val.min().item():.6f}, max={val.max().item():.6f}, mean={val.mean().item():.6f}")
                    output_has_nan = True
                    break
        elif torch.is_tensor(output) and (torch.isnan(output).any() or torch.isinf(output).any()):
            print(f"WARNING: NaN/Inf in model output at epoch {epoch}, step {data_iter_step} - skipping batch")
            print(f"  Output stats: min={output.min().item():.6f}, max={output.max().item():.6f}, mean={output.mean().item():.6f}")
            output_has_nan = True

        loss_value = loss.item()

        if not math.isfinite(loss_value) or output_has_nan:
            skipped_batches += 1
            current_lr = optimizer.param_groups[0]['lr']
            print(f"WARNING: Skipping batch (loss={loss_value}, output_has_nan={output_has_nan}) at epoch {epoch}, step {data_iter_step}")
            print(f"  Loss scale: {loss_scaler.state_dict()['scale'] if loss_scaler else 'N/A'}")
            print(f"  Learning rate: {current_lr}")
            print(f"  Total skipped batches this epoch: {skipped_batches}")
            
            # Check model parameters for NaN FIRST - if params are NaN, we must stop immediately
            nan_params = []
            for name, param in model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    nan_params.append(name)
            if nan_params:
                print(f"ERROR: Parameters with NaN/Inf detected: {nan_params[:5]}...")
                print("  Model is corrupted, stopping training immediately")
                raise RuntimeError("Training unstable: model parameters contain NaN/Inf")
            
            # If learning rate is extremely small, it may cause numerical instability
            # Note: With layer decay, some layers can have very small effective LR
            # Check the base learning rate (before layer decay) instead of effective LR
            # Get the maximum LR across all param groups to check the base schedule
            if lr_schedule_values is not None and it < len(lr_schedule_values):
                base_lr = lr_schedule_values[it]
            else:
                # If we can't get base LR from schedule, use the maximum LR across all param groups
                # (this should be close to the base LR since layer decay only reduces it)
                base_lr = max(pg['lr'] for pg in optimizer.param_groups)
            
            # Stop if base LR is below 1e-6 (10x smaller than typical min_lr of 1e-5)
            # This prevents stopping due to layer decay making some layers have very small LR
            if base_lr < 1e-6:
                print(f"ERROR: Base learning rate ({base_lr:.2e}) is too small, causing numerical instability.")
                print(f"  Effective LR for this layer: {current_lr:.2e}")
                print("  This may be due to learning rate schedule being too aggressive.")
                print("  Stopping training to prevent further corruption.")
                raise RuntimeError("Training unstable: learning rate too small")
            
            # If we've skipped too many batches in a row, stop training
            if skipped_batches > 5:
                print(f"ERROR: Skipped {skipped_batches} batches in a row. Training unstable, stopping.")
                print("  This suggests the model may be corrupted or there's a systematic issue.")
                raise RuntimeError("Training unstable: too many consecutive NaN/Inf batches")
            
            # Skip this batch and continue
            continue

        # Reset skipped_batches counter on successful batch
        skipped_batches = 0

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
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

        # Handle multi-output accuracy calculation
        if isinstance(output, dict):
            # Multi-output: compute average accuracy
            gender_pred = output['gender'].max(-1)[-1].detach().cpu()
            age_pred = output['age'].max(-1)[-1].detach().cpu()
            gender_labels, age_labels = targets
            gender_acc = (gender_pred == gender_labels.cpu()).float().mean()
            age_acc = (age_pred == age_labels.cpu()).float().mean()
            class_acc = (gender_acc + age_acc) / 2.0
        elif is_regression:
            mae_b = (output.detach() - targets.detach()).abs().mean()
            class_acc = 1.0 / (1.0 + mae_b)
        elif is_binary:
            class_acc = utils.get_metrics(torch.sigmoid(output).detach().cpu().numpy(), targets.detach().cpu().numpy(), ["accuracy"], is_binary)["accuracy"]
        else:
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

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # Note: Detailed stats are logged during training via log_every
    # The epoch summary is printed in run_class_finetuning.py for better readability
    
    # Prepare return dictionary
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    # Add task type counts to stats if available
    if task_type_counts is not None:
        stats['task_type_counts'] = task_type_counts.copy()
    
    # Print sample and batch count diagnostic in first epoch
    if count_samples:
        actual_batch_size = data_loader.batch_size
        expected_batches = sample_count // actual_batch_size if actual_batch_size > 0 else 0
        print(f"\n{'='*60}")
        print(f"  DIAGNOSTIC: Epoch {epoch} Summary")
        print(f"{'='*60}")
        print(f"  Processed: {sample_count:,} samples in {batch_count:,} batches")
        print(f"  Batch size: {actual_batch_size}")
        print(f"  Expected batches: ~{expected_batches:,} (samples ÷ batch_size)")
        if batch_count > 0:
            avg_samples = sample_count / batch_count
            print(f"  Average samples per batch: {avg_samples:.2f}")
            if abs(avg_samples - actual_batch_size) > 1:
                print(f"  ⚠️  Warning: Average samples per batch ({avg_samples:.2f}) differs from batch_size ({actual_batch_size})")
        
        # Compare with CNN baseline
        # CNN uses batch_size=256, shows progress every 1000 batches
        # If CNN shows "Processed 4000 batches", it might have more (just progress updates)
        # But based on typical dataset size, CNN should have ~7000 batches with batch_size=256 = ~1.79M samples
        cnn_samples = 1_792_000  # Approximate: 7000 batches × 256 batch_size
        cnn_batches = 7000
        if sample_count > 0:
            ratio = sample_count / cnn_samples
            print(f"\n  Comparison with CNN (baseline):")
            print(f"    CNN: ~{cnn_samples:,} samples (~{cnn_batches:,} batches, batch_size=256)")
            print(f"    LaBraM: {sample_count:,} samples ({batch_count:,} batches, batch_size={actual_batch_size})")
            print(f"    Ratio: {ratio:.2f}x samples ({'more' if ratio > 1 else 'fewer'} than CNN)")
            if ratio > 1.1 or ratio < 0.9:
                print(f"    ⚠️  WARNING: Significant difference from CNN dataset size!")
                print(f"    This suggests a serious issue with data loading or participant splits!")
                print(f"    Possible causes:")
                print(f"      1. Different participant splits (check random_seed and split ratios)")
                print(f"      2. Data duplication in IterableDataset (samples yielded multiple times)")
                print(f"      3. Different task_type usage (CNN might use 'active' or 'passive' instead of 'both')")
                print(f"      4. Different number of participants in train split")
                print(f"      5. LaBraM iterating through dataset multiple times")
                print(f"    Action: Check participant counts and verify no duplication in __iter__ methods")
        print(f"{'='*60}\n")
    
    # Add skipped batches count to stats
    stats['skipped_batches'] = skipped_batches
    if skipped_batches > 0:
        print(f"Epoch {epoch} summary: Skipped {skipped_batches} batches due to NaN/Inf")
    
    return stats


@torch.no_grad()
def evaluate(data_loader, model, device, header='Test:', ch_names=None, metrics=['acc'], is_binary=True,
             aggregate_by_participant=None, is_regression=False, age_min=None, age_max=None):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    
    # Check if model is multi-output (handle DDP wrapping)
    model_to_check = model.module if hasattr(model, 'module') else model
    is_multi_output = hasattr(model_to_check, 'multi_output') and model_to_check.multi_output

    if is_regression and is_binary:
        raise ValueError("evaluate: use is_binary=False for regression (is_regression=True).")
    if is_multi_output:
        criterion_gender = torch.nn.CrossEntropyLoss()
        criterion_age = torch.nn.CrossEntropyLoss()
    elif is_regression:
        criterion = torch.nn.MSELoss()
    elif is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    model.eval()
    pred = []
    true = []
    pred_gender = []
    pred_age = []
    true_gender = []
    true_age = []
    all_participant_ids = []

    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        EEG = batch[0]
        if len(batch) == 3:
            target = batch[1]
            participant_ids_batch = batch[2]
            if participant_ids_batch is not None:
                all_participant_ids.extend(participant_ids_batch)
        else:
            target = batch[-1]
            participant_ids_batch = None
        # If first element is not valid EEG (0-dim or 1-dim), find EEG-shaped tensor in batch.
        # Happens when dataset/collate yields wrong order: (label, pids, eeg) or (label, eeg, pids).
        if EEG.dim() < 2:
            if _DEBUG_EVALUATE and step == 0:
                for i, elem in enumerate(batch):
                    if torch.is_tensor(elem):
                        print(f"[DEBUG evaluate] batch[{i}]: tensor shape={elem.shape}, dim={elem.dim()}, dtype={elem.dtype}")
                    else:
                        t = type(elem).__name__
                        l = len(elem) if isinstance(elem, (list, tuple)) else "n/a"
                        print(f"[DEBUG evaluate] batch[{i}]: type={t}, len={l}")
            eeg_candidate = None
            for i, elem in enumerate(batch):
                if torch.is_tensor(elem) and elem.dim() >= 2 and elem.shape[-2] == 60:
                    eeg_candidate = (i, elem)
                    break
            if eeg_candidate is not None:
                idx, eeg_candidate = eeg_candidate
                if _DEBUG_EVALUATE and step == 0:
                    print(f"[DEBUG evaluate] Using batch[{idx}] as EEG (shape={eeg_candidate.shape})")
                warnings.warn(
                    "Batch first element not valid EEG; using EEG-shaped tensor from batch. Fix dataset yield order (eeg, label, pid).",
                    UserWarning,
                    stacklevel=2,
                )
                EEG = eeg_candidate
                # Target is the scalar/small tensor; pids are the list of strings.
                if idx == 1:
                    target = batch[0]
                elif idx == 2 and len(batch) >= 2:
                    target = batch[0]
                    participant_ids_batch = batch[1]
            if EEG.dim() < 2:
                if _DEBUG_EVALUATE:
                    for i, elem in enumerate(batch):
                        if torch.is_tensor(elem):
                            print(f"[DEBUG evaluate] batch[{i}]: tensor shape={elem.shape}, dim={elem.dim()}, shape[-2]={elem.shape[-2] if elem.dim() >= 2 else 'n/a'}")
                raise ValueError(
                    f"EEG batch has wrong shape: dim={EEG.dim()}, shape={EEG.shape}. "
                    "Expected (B, n_channels, n_timepoints). Check dataset yield order (eeg_data, label, participant_id)."
                )
        EEG = EEG.float().to(device, non_blocking=True) / 100
        if EEG.dim() == 2:
            EEG = EEG.unsqueeze(0)  # batch_size=1: (N, T) -> (1, N, T)
        EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)
        
        # Handle multi-output targets
        if isinstance(target, (list, tuple)) and len(target) == 2:
            gender_target = target[0].to(device, non_blocking=True).long()
            age_target = target[1].to(device, non_blocking=True).long()
        else:
            target = target.to(device, non_blocking=True)
            B = EEG.shape[0]
            if is_regression:
                target = target.float()
                if target.dim() == 1:
                    target = target.unsqueeze(1)
                if target.shape[0] != B:
                    raise ValueError(
                        f"Regression target batch size {target.shape[0]} does not match EEG batch size {B}."
                    )
                if target.shape != (B, 1):
                    raise ValueError(
                        f"Regression target must have shape ({B}, 1); got {tuple(target.shape)}. "
                        "Use collate_labram_with_participant_ids(..., label_mode='regression')."
                    )
            elif is_binary:
                target = target.float().unsqueeze(-1)
                # BCE and metrics expect target (B, 1) with binary 0/1. If target has wrong shape, normalize.
                if target.dim() > 2 or target.shape != (B, 1):
                    warnings.warn(
                        f"Binary target had shape {target.shape}, expected ({B}, 1). "
                        "Using first element per sample clamped to 0/1. Fix dataset/collate.",
                        UserWarning,
                        stacklevel=2,
                    )
                    flat = target.reshape(-1)[:B]
                    target = torch.clamp(flat.round().long(), 0, 1).float().unsqueeze(1)
            else:
                # Multi-class (e.g. age): CrossEntropyLoss expects target (B,) with class indices.
                if target.dim() == 0:
                    target = target.unsqueeze(0)
                if target.dim() > 1 or target.size(0) != B:
                    warnings.warn(
                        f"Multi-class target had shape {target.shape}, expected ({B},). "
                        "Using first B elements as class indices. Fix dataset/collate.",
                        UserWarning,
                        stacklevel=2,
                    )
                    target = target.reshape(-1)[:B].long()
        
        # compute output (fp32 for regression — matches CNN stability practice for MSE on bounded targets)
        if is_regression:
            if is_multi_output:
                raise ValueError("evaluate(is_regression=True) does not support multi_output models.")
            with torch.amp.autocast(device_type='cuda', enabled=False):
                output = model(EEG, input_chans=input_chans)
                loss = criterion(output, target)
        else:
            with torch.amp.autocast(device_type='cuda'):
                output = model(EEG, input_chans=input_chans)
                
                # Compute loss
                if is_multi_output:
                    gender_loss = criterion_gender(output['gender'], gender_target)
                    age_loss = criterion_age(output['age'], age_target)
                    loss = gender_loss + age_loss
                else:
                    loss = criterion(output, target)
        
        # Process outputs and targets for metrics
        if is_multi_output:
            # Process gender head
            gender_output = output['gender'].cpu()
            gender_target_cpu = gender_target.cpu()
            pred_gender.append(gender_output)
            true_gender.append(gender_target_cpu)
            
            # Process age head
            age_output = output['age'].cpu()
            age_target_cpu = age_target.cpu()
            pred_age.append(age_output)
            true_age.append(age_target_cpu)
            
            # For backward compatibility, use gender predictions
            pred.append(gender_output)
            true.append(gender_target_cpu)
        else:
            if is_binary:
                output = torch.sigmoid(output).cpu()
            elif is_regression:
                output = output.detach().cpu()
            else:
                output = output.cpu()
            target = target.cpu()
            pred.append(output)
            true.append(target)

        batch_size = EEG.shape[0]
        metric_logger.update(loss=loss.item())
        
        # Compute metrics per batch
        # Note: For binary classification, only compute accuracy per-batch
        # Other metrics (balanced_accuracy, pr_auc, roc_auc) require both classes
        # and will be computed correctly at the end when all predictions are aggregated
        if is_multi_output:
            # Compute metrics for gender head
            gender_results = utils.get_metrics(gender_output.numpy(), gender_target_cpu.numpy(), metrics, is_binary=False)
            for key, value in gender_results.items():
                metric_logger.meters[f'gender_{key}'].update(value, n=batch_size)
            
            # Compute metrics for age head
            age_results = utils.get_metrics(age_output.numpy(), age_target_cpu.numpy(), metrics, is_binary=False)
            for key, value in age_results.items():
                metric_logger.meters[f'age_{key}'].update(value, n=batch_size)
            
            # Also update main metrics (using gender for backward compatibility)
            for key, value in gender_results.items():
                metric_logger.meters[key].update(value, n=batch_size)
        else:
            # For binary classification, only update accuracy per-batch
            # Other metrics will be computed at the end from aggregated predictions
            if is_binary:
                # Compute only accuracy per-batch (works even with single-class batches)
                batch_target = target.numpy()
                batch_output = output.numpy()
                batch_pred = (batch_output > 0.5).astype(float)
                batch_acc = (batch_pred == batch_target).mean()
                metric_logger.meters['accuracy'].update(batch_acc, n=batch_size)
                
                # Update balanced_accuracy only when batch has both classes.
                # Note: If the model predicts only one class in a batch (all 0 or all 1),
                # balanced_accuracy = (recall_0 + recall_1)/2 is exactly 0.5 (one recall 0, one 1).
                # So progress-bar balanced_accuracy staying at 0.5 while accuracy ~65% usually
                # means the model is collapsed to the majority class; final metrics use full-set pred/true.
                has_both_classes = (batch_target.sum() > 0) and (batch_target.sum() < len(batch_target))
                if has_both_classes:
                    results = utils.get_metrics(batch_output, batch_target, metrics, is_binary)
                    for key, value in results.items():
                        if key != 'accuracy':  # Already updated above
                            metric_logger.meters[key].update(value, n=batch_size)
            elif is_regression:
                pass  # Final MAE/MSE/RMSE/R² computed on full val/test set (denormalized when age bounds are set).
            else:
                # Multi-class: compute all metrics per-batch
                results = utils.get_metrics(output.numpy(), target.numpy(), metrics, is_binary)
                for key, value in results.items():
                    metric_logger.meters[key].update(value, n=batch_size)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))
    
    # Compute final metrics
    if is_multi_output:
            pred_gender_all = torch.cat(pred_gender, dim=0).numpy()
            true_gender_all = torch.cat(true_gender, dim=0).numpy()
            pred_age_all = torch.cat(pred_age, dim=0).numpy()
            true_age_all = torch.cat(true_age, dim=0).numpy()
            
            # Compute metrics for each head
            gender_metrics = utils.get_metrics(pred_gender_all, true_gender_all, metrics, is_binary=False, threshold=0.5)
            age_metrics = utils.get_metrics(pred_age_all, true_age_all, metrics, is_binary=False, threshold=0.5)
            
            # Compute average metrics to match training's calculation
            # Training computes: (gender_acc + age_acc) / 2
            avg_metrics = {}
            for key in gender_metrics:
                if key in age_metrics:
                    gender_val = gender_metrics.get(key)
                    age_val = age_metrics.get(key)
                    # Handle None values (e.g., for ROC-AUC if not computed)
                    if gender_val is not None and age_val is not None:
                        avg_metrics[key] = (gender_val + age_val) / 2.0
                    elif gender_val is not None:
                        avg_metrics[key] = gender_val
                    elif age_val is not None:
                        avg_metrics[key] = age_val
                    else:
                        avg_metrics[key] = None
        
            # Combine results - use average metrics as primary (matching training behavior)
            ret = {
                'loss': metric_logger.loss.global_avg,
                'gender_head': {k: v for k, v in gender_metrics.items()},
                'age_head': {k: v for k, v in age_metrics.items()},
                **avg_metrics  # Unpack average metrics as primary (matching training behavior)
            }
    else:
        pred = torch.cat(pred, dim=0).numpy()
        true = torch.cat(true, dim=0).numpy()
        ret = utils.get_metrics(
            pred, true, metrics, is_binary, 0.5,
            is_regression=is_regression, age_min=age_min, age_max=age_max,
        )
        ret['loss'] = metric_logger.loss.global_avg

    # Participant-level aggregation (segment-level vs majority vote / mean prob), reusing data_processing.aggregation (same as CNN)
    do_agg = (
        aggregate_by_participant == 'majority_vote'
        and aggregate_predictions_by_group is not None
        and len(all_participant_ids) > 0
        and None not in all_participant_ids
    )
    if do_agg and is_multi_output:
        pred_gender_all = torch.cat(pred_gender, dim=0).numpy()
        true_gender_all = torch.cat(true_gender, dim=0).numpy()
        pred_age_all = torch.cat(pred_age, dim=0).numpy()
        true_age_all = torch.cat(true_age, dim=0).numpy()
        n_seg = len(pred_gender_all)
        if len(all_participant_ids) != n_seg:
            raise ValueError(
                f"Participant-level aggregation requires one participant_id per segment. "
                f"Got {len(all_participant_ids)} participant_ids for {n_seg} segments."
            )
        probs_gender = torch.softmax(torch.from_numpy(pred_gender_all), dim=1).numpy()
        probs_age = torch.softmax(torch.from_numpy(pred_age_all), dim=1).numpy()
        pred_gender_cls = np.argmax(probs_gender, axis=1)
        pred_age_cls = np.argmax(probs_age, axis=1)
        pid_list = all_participant_ids
        if aggregate_participant_level_classification is not None:
            _, g_true_mp, g_pred_mp, g_prob_mp, g_true_mv, g_pred_mv, g_prob_mv = aggregate_participant_level_classification(
                pred_gender_cls, true_gender_all, pid_list, probs_gender
            )
            _, a_true_mp, a_pred_mp, a_prob_mp, a_true_mv, a_pred_mv, a_prob_mv = aggregate_participant_level_classification(
                pred_age_cls, true_age_all, pid_list, probs_age
            )
        else:
            _, g_true_mp, g_pred_mp, g_prob_mp = aggregate_predictions_by_group(
                pred_gender_cls, true_gender_all, pid_list,
                prediction_type='classification', aggregation=AGGREGATION_MEAN_PROB,
                probabilities=probs_gender)
            _, a_true_mp, a_pred_mp, a_prob_mp = aggregate_predictions_by_group(
                pred_age_cls, true_age_all, pid_list,
                prediction_type='classification', aggregation=AGGREGATION_MEAN_PROB,
                probabilities=probs_age)
            _, g_true_mv, g_pred_mv, g_prob_mv = aggregate_predictions_by_group(
                pred_gender_cls, true_gender_all, pid_list,
                prediction_type='classification', aggregation=AGGREGATION_MAJORITY_VOTE,
                probabilities=probs_gender)
            _, a_true_mv, a_pred_mv, a_prob_mv = aggregate_predictions_by_group(
                pred_age_cls, true_age_all, pid_list,
                prediction_type='classification', aggregation=AGGREGATION_MAJORITY_VOTE,
                probabilities=probs_age)
        # Metrics from predictions; use probabilities when available (for ROC etc.), else predictions (accuracy only)
        g_mp = utils.get_metrics(
            g_prob_mp if g_prob_mp is not None else g_pred_mp,
            g_true_mp, metrics, is_binary=False
        )
        a_mp = utils.get_metrics(
            a_prob_mp if a_prob_mp is not None else a_pred_mp,
            a_true_mp, metrics, is_binary=False
        )
        g_mv = utils.get_metrics(
            g_prob_mv if g_prob_mv is not None else g_pred_mv,
            g_true_mv, metrics, is_binary=False
        )
        a_mv = utils.get_metrics(
            a_prob_mv if a_prob_mv is not None else a_pred_mv,
            a_true_mv, metrics, is_binary=False
        )
        ret['segment_metrics'] = {k: v for k, v in ret.items() if k not in ('loss',)}
        ret['participant_mean_prob_metrics'] = {'gender_head': g_mp, 'age_head': a_mp,
            'accuracy': (g_mp['accuracy'] + a_mp['accuracy']) / 2.0}
        ret['participant_metrics'] = {'gender_head': g_mv, 'age_head': a_mv,
            'accuracy': (g_mv['accuracy'] + a_mv['accuracy']) / 2.0}
    elif do_agg and not is_multi_output and is_regression:
        if aggregate_predictions_by_group is None or AGGREGATION_MEDIAN is None or AGGREGATION_MEAN is None:
            raise RuntimeError(
                "Participant-level regression aggregation requires data_processing.aggregation "
                "(aggregate_predictions_by_group, AGGREGATION_MEDIAN, AGGREGATION_MEAN)."
            )
        pred_all = np.asarray(pred).reshape(-1)
        true_all = np.asarray(true).reshape(-1)
        if len(all_participant_ids) != len(pred_all):
            raise ValueError(
                f"Participant-level aggregation requires one participant_id per segment. "
                f"Got {len(all_participant_ids)} participant_ids for {len(pred_all)} segments."
            )
        _, true_med, pred_med, _ = aggregate_predictions_by_group(
            pred_all, true_all, all_participant_ids,
            prediction_type='regression', aggregation=AGGREGATION_MEAN_PROB,
            regression_aggregation=AGGREGATION_MEDIAN,
        )
        _, true_mean, pred_mean, _ = aggregate_predictions_by_group(
            pred_all, true_all, all_participant_ids,
            prediction_type='regression', aggregation=AGGREGATION_MEAN_PROB,
            regression_aggregation=AGGREGATION_MEAN,
        )
        ret['segment_metrics'] = {k: v for k, v in ret.items() if k != 'loss'}
        ret['participant_median_metrics'] = utils.get_metrics(
            pred_med, true_med, metrics, is_binary=False, threshold=0.5,
            is_regression=True, age_min=age_min, age_max=age_max,
        )
        ret['participant_mean_metrics'] = utils.get_metrics(
            pred_mean, true_mean, metrics, is_binary=False, threshold=0.5,
            is_regression=True, age_min=age_min, age_max=age_max,
        )
    elif do_agg and not is_multi_output and not is_regression:
        # pred/true are already numpy after the single-output branch above
        pred_all = pred
        true_all = true
        if len(all_participant_ids) != len(pred_all):
            raise ValueError(
                f"Participant-level aggregation requires one participant_id per segment. "
                f"Got {len(all_participant_ids)} participant_ids for {len(pred_all)} segments."
            )
        if is_binary:
            probs = pred_all
            pred_cls = (probs > 0.5).astype(np.int64)
        else:
            probs = torch.softmax(torch.from_numpy(pred_all), dim=1).numpy()
            pred_cls = np.argmax(probs, axis=1)
        _, true_mp, pred_mp, prob_mp = aggregate_predictions_by_group(
            pred_cls, true_all, all_participant_ids,
            prediction_type='classification', aggregation=AGGREGATION_MEAN_PROB,
            probabilities=probs)
        _, true_mv, pred_mv, prob_mv = aggregate_predictions_by_group(
            pred_cls, true_all, all_participant_ids,
            prediction_type='classification', aggregation=AGGREGATION_MAJORITY_VOTE,
            probabilities=probs)
        ret['segment_metrics'] = {k: v for k, v in ret.items() if k not in ('loss',)}
        # Binary: use probabilities when available (for ROC etc.), else predictions.
        # Multiclass: mean_prob has 2D prob_mp — pass it so get_metrics/multiclass_metrics_fn get (n, n_classes). Majority vote gives 1D class indices; get_metrics converts to 2D when needed.
        if is_binary:
            out_mp = prob_mp if prob_mp is not None else pred_mp
            out_mv = prob_mv if prob_mv is not None else pred_mv
        else:
            out_mp = prob_mp if prob_mp is not None else pred_mp
            out_mv = pred_mv
        ret['participant_mean_prob_metrics'] = utils.get_metrics(out_mp, true_mp, metrics, is_binary, 0.5)
        ret['participant_metrics'] = utils.get_metrics(out_mv, true_mv, metrics, is_binary, 0.5)

    return ret


def count_task_type_samples(data_loader):
    """
    Count active and passive samples in a data loader by iterating through it.
    
    Args:
        data_loader: DataLoader to count samples from
        
    Returns:
        Dictionary with 'active' and 'passive' sample counts
    """
    counts = {'active': 0, 'passive': 0}
    
    # Access underlying dataset to get task_type from samples
    dataset = data_loader.dataset
    if hasattr(dataset, 'eeg_dataset'):
        # LaBraMEEGDataset wrapper - access underlying EEGDataset
        eeg_dataset = dataset.eeg_dataset
        # Iterate through dataset to count task types (creates new iterator)
        for sample in eeg_dataset:
            task_type = sample.get('task_type', 'unknown')
            if task_type in counts:
                counts[task_type] += 1
    else:
        # Direct EEGDataset - iterate to count (creates new iterator)
        for sample in dataset:
            task_type = sample.get('task_type', 'unknown')
            if task_type in counts:
                counts[task_type] += 1
    
    return counts


def evaluate_by_task_type(model, device, dataset_type, hdf5_dir, segment_length, 
                          ch_names=None, metrics=['acc'], is_binary=True, random_seed=42,
                          is_regression=False, age_min=None, age_max=None):
    """
    Evaluate model separately on active and passive tasks.
    
    Args:
        model: Model to evaluate
        device: Device to run evaluation on
        dataset_type: Type of dataset ("gender", "age", "age_regression", "combined", "multi_output")
        hdf5_dir: Directory containing HDF5 files
        segment_length: Length of segments ('1s', '2s', or '4s')
        ch_names: Channel names
        metrics: List of metrics to compute
        is_binary: Whether this is binary classification
        random_seed: Random seed for shuffling
        is_regression: Age regression (continuous targets)
        age_min, age_max: Denormalization bounds from training data (years)
    Returns:
        Dictionary mapping task type to evaluation results with sample counts
        {'active': {...}, 'passive': {...}}
    """
    # Create datasets and loaders for each task type
    from functools import partial

    def _create_task_type_loader(task_type: str):
        """Helper to create dataset and loader for a specific task type."""
        test_dataset = prepare_labram_dataset(
            dataset_type=dataset_type,
            hdf5_dir=hdf5_dir,
            segment_length=segment_length,
            task_type=task_type,
            random_seed=random_seed
        )[2]  # Get test dataset (index 2)
        collate_fn = (
            partial(collate_labram_with_participant_ids, label_mode='regression')
            if is_regression
            else collate_labram_with_participant_ids
        )
        
        # drop_last=True so every batch has full size; avoids DataParallel scatter
        # giving empty chunks when batch size < num GPUs (TypeError: forward() missing 1 required positional argument: 'x')
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=64,  # Use reasonable batch size for evaluation
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
    
    active_loader = _create_task_type_loader("active")
    passive_loader = _create_task_type_loader("passive")
    
    # Count samples for each task type
    print(f"\n{'='*60}")
    print(f"Counting samples by task type...")
    print(f"{'='*60}")
    active_counts = count_task_type_samples(active_loader)
    passive_counts = count_task_type_samples(passive_loader)
    
    # Evaluate separately
    print(f"\n{'='*60}")
    print(f"Evaluating on active tasks ({active_counts['active']:,} samples)...")
    print(f"{'='*60}")
    active_results = evaluate(active_loader, model, device, header='Active Test:', 
                              ch_names=ch_names, metrics=metrics, is_binary=is_binary,
                              is_regression=is_regression, age_min=age_min, age_max=age_max)
    active_results['num_samples'] = active_counts['active']
    
    print(f"\n{'='*60}")
    print(f"Evaluating on passive tasks ({passive_counts['passive']:,} samples)...")
    print(f"{'='*60}")
    passive_results = evaluate(passive_loader, model, device, header='Passive Test:', 
                               ch_names=ch_names, metrics=metrics, is_binary=is_binary,
                               is_regression=is_regression, age_min=age_min, age_max=age_max)
    passive_results['num_samples'] = passive_counts['passive']
    
    return {'active': active_results, 'passive': passive_results}
