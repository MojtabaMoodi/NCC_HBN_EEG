"""
Shared segment- and participant-level classification metrics for LaBraM evaluation.

Used by ``engine_for_finetuning.evaluate`` and ``eval_random_classifier`` so both paths
apply the same ``get_metrics`` + ``aggregate_predictions_by_group`` logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import importlib.util

import numpy as np
import torch
from einops import rearrange
from sklearn.metrics import confusion_matrix

_labram_dir = Path(__file__).resolve().parent
_utils_path = _labram_dir / "utils.py"
_spec = importlib.util.spec_from_file_location("labram_internal_utils", _utils_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load LaBraM utils from {_utils_path}")
labram_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(labram_utils)

from data_processing.aggregation import (
    AGGREGATION_MAJORITY_VOTE,
    AGGREGATION_MEAN_PROB,
    aggregate_predictions_by_group,
)

# Standard 10–20 montage used by all LaBraM custom HDF5 datasets (see run_class_finetuning.CHANNEL_NAMES).
LABRAM_CHANNEL_NAMES: Tuple[str, ...] = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "F1", "F2", "F5", "F6", "F9", "F10",
    "AF3", "AF4", "AF7", "AF8", "AFZ", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6",
    "FT7", "FT8", "T7", "T8", "T9", "T10", "P7", "P3", "PZ", "P4", "P8", "P1", "P2",
    "P5", "P6", "PO3", "PO4", "PO7", "PO8", "POZ", "OZ", "O1", "O2", "C3", "C4",
    "C1", "C2", "C5", "C6", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPZ",
)


def collect_test_labels_and_participant_ids(
    data_loader,
    *,
    is_binary: bool,
) -> Tuple[np.ndarray, List[str]]:
    """
    Iterate a LaBraM test DataLoader (with ``collate_labram_with_participant_ids``)
    and collect segment labels and participant ids without running a model.
    """
    true_chunks: List[np.ndarray] = []
    participant_ids: List[str] = []

    for batch in data_loader:
        if len(batch) == 3:
            target = batch[1]
            participant_ids_batch = batch[2]
            if participant_ids_batch is not None:
                participant_ids.extend(participant_ids_batch)
        else:
            target = batch[-1]

        if isinstance(target, (list, tuple)):
            raise ValueError(
                "collect_test_labels_and_participant_ids does not support multi-output batches."
            )

        target_cpu = target.detach().cpu()
        if is_binary:
            if target_cpu.dim() == 0:
                target_cpu = target_cpu.unsqueeze(0)
            if target_cpu.dim() > 1:
                flat = target_cpu.reshape(-1)
                target_cpu = torch.clamp(flat.round().long(), 0, 1).float().unsqueeze(1)
            else:
                target_cpu = target_cpu.float().unsqueeze(-1)
        else:
            if target_cpu.dim() == 0:
                target_cpu = target_cpu.unsqueeze(0)
            if target_cpu.dim() > 1:
                target_cpu = target_cpu.reshape(-1).long()
        true_chunks.append(target_cpu.numpy())

    if not true_chunks:
        raise ValueError("Test DataLoader produced no batches.")

    true_all = np.concatenate(true_chunks, axis=0)
    return true_all, participant_ids


def _pred_probs_from_segment_outputs(
    pred_all: np.ndarray,
    *,
    is_binary: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (class_indices, probabilities) for participant aggregation."""
    if is_binary:
        probs = np.asarray(pred_all, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs.reshape(-1, 1)
        pred_cls = (probs > 0.5).astype(np.int64).reshape(-1)
        return pred_cls, probs

    logits = np.asarray(pred_all, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred_cls = np.argmax(probs, axis=1).astype(np.int64)
    return pred_cls, probs


def compute_classification_eval_stats(
    pred_all: np.ndarray,
    true_all: np.ndarray,
    participant_ids: Sequence[str],
    *,
    metrics: Sequence[str],
    is_binary: bool,
    aggregate_by_participant: Optional[str],
    loss: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute segment-level metrics and optional participant-level aggregates.

    Mirrors the single-output tail of ``engine_for_finetuning.evaluate``.
    """
    true_all = np.asarray(true_all)
    pred_all = np.asarray(pred_all)

    if is_binary:
        if true_all.ndim == 1:
            true_for_metrics = true_all.reshape(-1, 1)
        else:
            true_for_metrics = true_all
    else:
        true_for_metrics = true_all.reshape(-1).astype(np.int64, copy=False)

    ret = labram_utils.get_metrics(
        pred_all,
        true_for_metrics,
        list(metrics),
        is_binary,
        threshold=0.5,
    )
    if loss is not None:
        ret["loss"] = float(loss)

    do_agg = (
        aggregate_by_participant == AGGREGATION_MAJORITY_VOTE
        and aggregate_predictions_by_group is not None
        and len(participant_ids) > 0
        and None not in participant_ids
    )
    if not do_agg:
        return ret

    if len(participant_ids) != len(pred_all):
        raise ValueError(
            "Participant-level aggregation requires one participant_id per segment. "
            f"Got {len(participant_ids)} participant_ids for {len(pred_all)} segments."
        )

    pred_cls, probs = _pred_probs_from_segment_outputs(pred_all, is_binary=is_binary)
    if is_binary:
        true_flat = true_for_metrics.reshape(-1)
    else:
        true_flat = true_for_metrics

    _, true_mp, pred_mp, prob_mp = aggregate_predictions_by_group(
        pred_cls,
        true_flat,
        list(participant_ids),
        prediction_type="classification",
        aggregation=AGGREGATION_MEAN_PROB,
        probabilities=probs,
    )
    _, true_mv, pred_mv, prob_mv = aggregate_predictions_by_group(
        pred_cls,
        true_flat,
        list(participant_ids),
        prediction_type="classification",
        aggregation=AGGREGATION_MAJORITY_VOTE,
        probabilities=probs,
    )

    ret["segment_metrics"] = {k: v for k, v in ret.items() if k != "loss"}
    if is_binary:
        out_mp = prob_mp if prob_mp is not None else pred_mp
        out_mv = prob_mv if prob_mv is not None else pred_mv
    else:
        out_mp = prob_mp if prob_mp is not None else pred_mp
        out_mv = pred_mv

    ret["participant_mean_prob_metrics"] = labram_utils.get_metrics(
        out_mp, true_mp, list(metrics), is_binary, 0.5
    )
    ret["participant_metrics"] = labram_utils.get_metrics(
        out_mv, true_mv, list(metrics), is_binary, 0.5
    )
    return ret


def classification_label_indices(*, is_binary: bool, nb_classes_config: int) -> List[int]:
    """Fixed class indices for confusion matrices (gender: 0/1; age: 0..C-1)."""
    if is_binary:
        return [0, 1]
    if nb_classes_config < 2:
        raise ValueError(
            f"Multi-class confusion matrix requires nb_classes_config >= 2; got {nb_classes_config}."
        )
    return list(range(nb_classes_config))


def flatten_classification_true_labels(true_all: np.ndarray, *, is_binary: bool) -> np.ndarray:
    """Segment-level integer class labels aligned with ``_pred_probs_from_segment_outputs``."""
    true_all = np.asarray(true_all)
    if is_binary:
        if true_all.ndim == 1:
            flat = true_all.reshape(-1)
        else:
            flat = true_all.reshape(-1)
        return np.clip(np.rint(flat).astype(np.int64), 0, 1)
    return true_all.reshape(-1).astype(np.int64, copy=False)


def confusion_matrix_payload(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Sequence[int],
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Confusion matrix requires equal-length y_true and y_pred; "
            f"got {len(y_true)} vs {len(y_pred)}."
        )
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    return {
        "labels": [int(x) for x in labels],
        "matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
    }


def compute_classification_confusion_matrices(
    pred_all: np.ndarray,
    true_all: np.ndarray,
    participant_ids: Sequence[str],
    *,
    is_binary: bool,
    nb_classes_config: int,
    aggregate_by_participant: Optional[str],
) -> Dict[str, Any]:
    """
    Segment- and participant-level confusion matrices for LaBraM age/gender classification.

    Participant aggregation mirrors ``compute_classification_eval_stats`` (mean prob + majority vote).
    """
    labels = classification_label_indices(is_binary=is_binary, nb_classes_config=nb_classes_config)
    true_flat = flatten_classification_true_labels(true_all, is_binary=is_binary)
    pred_cls, probs = _pred_probs_from_segment_outputs(pred_all, is_binary=is_binary)

    out: Dict[str, Any] = {
        "segment": confusion_matrix_payload(true_flat, pred_cls, labels=labels),
    }

    if aggregate_by_participant != AGGREGATION_MAJORITY_VOTE:
        return out

    if len(participant_ids) == 0:
        raise ValueError(
            "Participant-level confusion matrices require participant_ids from the test loader."
        )
    if None in participant_ids:
        raise ValueError("Participant ids must not contain None for participant-level confusion matrices.")
    if len(participant_ids) != len(pred_all):
        raise ValueError(
            "Participant-level confusion matrices require one participant_id per segment. "
            f"Got {len(participant_ids)} participant_ids for {len(pred_all)} segments."
        )

    _, true_mp, pred_mp, _ = aggregate_predictions_by_group(
        pred_cls,
        true_flat,
        list(participant_ids),
        prediction_type="classification",
        aggregation=AGGREGATION_MEAN_PROB,
        probabilities=probs,
    )
    _, true_mv, pred_mv, _ = aggregate_predictions_by_group(
        pred_cls,
        true_flat,
        list(participant_ids),
        prediction_type="classification",
        aggregation=AGGREGATION_MAJORITY_VOTE,
        probabilities=probs,
    )
    out["participant_mean_prob"] = confusion_matrix_payload(true_mp, pred_mp, labels=labels)
    out["participant_majority_vote"] = confusion_matrix_payload(true_mv, pred_mv, labels=labels)
    return out


@torch.no_grad()
def collect_model_segment_predictions(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    *,
    ch_names: Sequence[str],
    is_binary: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Run LaBraM classification inference on a test loader.

    Returns raw model outputs (sigmoid probs for binary gender, logits for multi-class age),
    segment labels, and participant ids — same tensors as ``engine_for_finetuning.evaluate``.
    """
    input_chans = labram_utils.get_input_chans(list(ch_names))
    model.eval()
    pred_chunks: List[np.ndarray] = []
    true_chunks: List[np.ndarray] = []
    participant_ids: List[str] = []

    for batch in data_loader:
        eeg = batch[0]
        if len(batch) == 3:
            target = batch[1]
            participant_ids_batch = batch[2]
            if participant_ids_batch is not None:
                participant_ids.extend(participant_ids_batch)
        else:
            target = batch[-1]

        if isinstance(target, (list, tuple)):
            raise ValueError(
                "collect_model_segment_predictions does not support multi-output batches."
            )

        if eeg.dim() < 2:
            eeg_candidate = None
            for elem in batch:
                if torch.is_tensor(elem) and elem.dim() >= 2 and elem.shape[-2] == 60:
                    eeg_candidate = elem
                    break
            if eeg_candidate is None:
                raise ValueError(
                    f"EEG batch has wrong shape: dim={eeg.dim()}, shape={tuple(eeg.shape)}. "
                    "Expected (B, n_channels, n_timepoints)."
                )
            eeg = eeg_candidate

        eeg = eeg.float().to(device, non_blocking=True) / 100
        if eeg.dim() == 2:
            eeg = eeg.unsqueeze(0)
        eeg = rearrange(eeg, "B N (A T) -> B N A T", T=200)

        target = target.to(device, non_blocking=True)
        batch_size = eeg.shape[0]
        if is_binary:
            target = target.float().unsqueeze(-1)
            if target.dim() > 2 or target.shape != (batch_size, 1):
                flat = target.reshape(-1)[:batch_size]
                target = torch.clamp(flat.round().long(), 0, 1).float().unsqueeze(1)
        else:
            if target.dim() == 0:
                target = target.unsqueeze(0)
            if target.dim() > 1 or target.size(0) != batch_size:
                target = target.reshape(-1)[:batch_size].long()

        with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
            output = model(eeg, input_chans=input_chans)

        model_to_check = model.module if hasattr(model, "module") else model
        if hasattr(model_to_check, "multi_output") and model_to_check.multi_output:
            raise ValueError(
                "collect_model_segment_predictions does not support multi_output models."
            )

        if is_binary:
            output = torch.sigmoid(output).cpu()
        else:
            output = output.cpu()
        target = target.cpu()
        pred_chunks.append(output.numpy())
        true_chunks.append(target.numpy())

    if not pred_chunks:
        raise ValueError("Test DataLoader produced no batches.")

    pred_all = np.concatenate(pred_chunks, axis=0)
    true_all = np.concatenate(true_chunks, axis=0)
    return pred_all, true_all, participant_ids


def compute_mean_classification_loss(
    logits: np.ndarray,
    true_all: np.ndarray,
    *,
    is_binary: bool,
) -> float:
    """Same loss definitions as ``engine_for_finetuning.evaluate`` (BCE / CE on logits)."""
    logits_t = torch.as_tensor(logits)
    if is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
        true_t = torch.as_tensor(true_all, dtype=torch.float32)
        if true_t.dim() == 1:
            true_t = true_t.unsqueeze(1)
        if logits_t.dim() == 1:
            logits_t = logits_t.unsqueeze(1)
        return float(criterion(logits_t, true_t).item())

    criterion = torch.nn.CrossEntropyLoss()
    true_t = torch.as_tensor(true_all, dtype=torch.long).reshape(-1)
    if logits_t.ndim == 1:
        logits_t = logits_t.unsqueeze(1)
    return float(criterion(logits_t, true_t).item())
