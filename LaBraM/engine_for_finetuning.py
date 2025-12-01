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
import sys
from typing import Iterable, Optional
import torch
from timm.utils import ModelEma
import utils
from einops import rearrange

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
                    num_training_steps_per_epoch=None, update_freq=None, ch_names=None, is_binary=True):
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

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    # Track actual steps for IterableDataset (unknown size)
    # For IterableDataset, we let the iterator naturally exhaust
    # The step limit check is only for safety if num_training_steps_per_epoch is set to a reasonable value
    actual_steps = 0
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
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

        samples = samples.float().to(device, non_blocking=True) / 100
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        
        # Handle multi-output targets (tuple of (gender, age))
        is_multi_output = isinstance(targets, (list, tuple)) and len(targets) == 2
        if is_multi_output:
            gender_targets = targets[0].to(device, non_blocking=True).long()
            age_targets = targets[1].to(device, non_blocking=True).long()
            targets = (gender_targets, age_targets)
        else:
            targets = targets.to(device, non_blocking=True)
            if is_binary:
                targets = targets.float().unsqueeze(-1)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion, input_chans)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion, input_chans)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

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
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, header='Test:', ch_names=None, metrics=['acc'], is_binary=True):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    
    # Check if model is multi-output (handle DDP wrapping)
    model_to_check = model.module if hasattr(model, 'module') else model
    is_multi_output = hasattr(model_to_check, 'multi_output') and model_to_check.multi_output
    
    if is_multi_output:
        criterion_gender = torch.nn.CrossEntropyLoss()
        criterion_age = torch.nn.CrossEntropyLoss()
    elif is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #header = 'Test:'

    # switch to evaluation mode
    model.eval()
    pred = []
    true = []
    pred_gender = []
    pred_age = []
    true_gender = []
    true_age = []
    
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        EEG = batch[0]
        target = batch[-1]
        EEG = EEG.float().to(device, non_blocking=True) / 100
        EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)
        
        # Handle multi-output targets
        if isinstance(target, (list, tuple)) and len(target) == 2:
            gender_target = target[0].to(device, non_blocking=True).long()
            age_target = target[1].to(device, non_blocking=True).long()
        else:
            target = target.to(device, non_blocking=True)
            if is_binary:
                target = target.float().unsqueeze(-1)
        
        # compute output
        with torch.cuda.amp.autocast():
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
            else:
                output = output.cpu()
            target = target.cpu()
            pred.append(output)
            true.append(target)

        batch_size = EEG.shape[0]
        metric_logger.update(loss=loss.item())
        
        # Compute metrics per batch
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
        ret = utils.get_metrics(pred, true, metrics, is_binary, 0.5)
        ret['loss'] = metric_logger.loss.global_avg
    
    return ret
