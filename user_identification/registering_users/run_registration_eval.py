#!/usr/bin/env python3
"""
Evaluate registration using ONLY unknown-user data.

Protocol:
1. Index unknown split participants for one fold/task.
2. Randomly split unknown participants into registered vs unregistered.
3. For each registered participant, split segments into enrollment/probe.
4. Build one prototype per registered user from enrollment embeddings.
5. Evaluate open-world user classification on:
   - probe segments from registered users (should map to correct registered ID)
   - all segments from unregistered users (should map to UNKNOWN class)

No train/val/test-known user data are used in this evaluation logic.

Examples
--------
::

    python user_identification/registering_users/run_registration_eval.py \\
      --checkpoint final_user_identification/4s/fold_0/best_model.pth \\
      --hdf5_dir "${HOME}/scratch/fold_0" \\
      --segment_length 4s \\
      --task_type active \\
      --output_dir registering_users_results/fold_0_active
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.metrics.pairwise import euclidean_distances
from torch.utils.data import DataLoader

_project_root = Path(__file__).resolve().parent.parent.parent
_labram_dir = _project_root / "LaBraM"
for _p in (_project_root, _labram_dir):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from CNN.models.arcface_loss import ArcFaceLoss  # noqa: E402
from data_processing.target_transforms import (  # noqa: E402
    UserIdentificationTransform,
    create_user_identification_transform_from_hdf5,
)
from user_identification.analyze_confidence import (  # noqa: E402
    ood_distance_to_known_user_score,
    run_inference,
)
from user_identification.labram_model import LaBraMUserIdentificationWrapper  # noqa: E402
from user_identification.registering_users.registration_map_dataset import (  # noqa: E402
    UnknownRegistrationMapDataset,
)
from user_identification.registering_users.unknown_hdf5_index import (  # noqa: E402
    UnknownSampleRef,
    list_unknown_samples,
    split_unknown_participants_for_registration,
)


_THRESH_CRITERIA = (
    "f1_macro",
    "balanced_accuracy",
    "accuracy",
    "mcc",
    "weighted_recall_balance",
    "harmonic_recall_balance",
)
_THRESH_SWEEP_NAMES_TAIL = (
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "mcc",
    "unknown_recall",
    "mean_registered_recall",
    "weighted_recall_balance",
    "harmonic_recall_balance",
)


def _threshold_sweep_column_names(threshold_on: str) -> Tuple[str, ...]:
    first = (
        "known_user_score_threshold"
        if threshold_on == "known_user_score"
        else "distance_threshold"
    )
    return (first,) + _THRESH_SWEEP_NAMES_TAIL


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    needed = ["model_state_dict", "arcface_state_dict", "num_classes", "embedding_dim"]
    missing = [k for k in needed if k not in ckpt]
    if missing:
        raise KeyError(f"Checkpoint {path} missing keys: {missing}")
    return ckpt


def _build_model_and_criterion(
    ckpt: Dict[str, Any],
    device: torch.device,
    dropout_rate: float,
    model_name: str,
    arcface_margin: float,
    arcface_scale: float,
    arcface_easy_margin: bool,
) -> Tuple[torch.nn.Module, ArcFaceLoss]:
    num_classes = int(ckpt["num_classes"])
    embedding_dim = int(ckpt["embedding_dim"])
    model = LaBraMUserIdentificationWrapper(
        num_channels=60,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained_path=None,
        model_name=model_name,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    criterion = ArcFaceLoss(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        margin=arcface_margin,
        scale=arcface_scale,
        easy_margin=arcface_easy_margin,
    ).to(device)
    criterion.load_state_dict(ckpt["arcface_state_dict"], strict=True)
    criterion.eval()
    return model, criterion


def _stack_registration_prototypes(
    enroll_by_pid: Dict[str, List[UnknownSampleRef]],
    registered_sorted: List[str],
    model: torch.nn.Module,
    criterion: ArcFaceLoss,
    device: torch.device,
    ui_transform: UserIdentificationTransform,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    flat: List[UnknownSampleRef] = []
    for pid in registered_sorted:
        flat.extend(enroll_by_pid[pid])
    ds = UnknownRegistrationMapDataset(flat, ui_transform, transform=None)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    try:
        _, _, _, pids, embeddings = run_inference(
            model,
            loader,
            device,
            arcface_criterion=criterion,
            use_arcface=True,
            return_embeddings=True,
        )
    finally:
        ds.close()

    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.shape[0] != len(pids):
        raise RuntimeError("participant_id list length mismatch vs embeddings")

    dim = emb.shape[1]
    protos = np.zeros((len(registered_sorted), dim), dtype=np.float32)
    for i, pid in enumerate(registered_sorted):
        idx = [j for j, p in enumerate(pids) if p == pid]
        if not idx:
            raise RuntimeError(f"No enrollment embeddings for participant {pid!r}")
        mean = emb[idx].mean(axis=0)
        nrm = np.linalg.norm(mean)
        if nrm <= 0:
            raise RuntimeError(f"Zero-norm prototype for participant {pid!r}")
        protos[i] = (mean / nrm).astype(np.float32)
    return protos


def _embeddings_for_refs(
    refs: List[UnknownSampleRef],
    model: torch.nn.Module,
    criterion: ArcFaceLoss,
    device: torch.device,
    ui_transform: UserIdentificationTransform,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, List[str]]:
    ds = UnknownRegistrationMapDataset(refs, ui_transform, transform=None)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    try:
        _, _, _, pids, embeddings = run_inference(
            model,
            loader,
            device,
            arcface_criterion=criterion,
            use_arcface=True,
            return_embeddings=True,
        )
    finally:
        ds.close()
    return np.asarray(embeddings, dtype=np.float32), list(pids)


def _l2_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    d = euclidean_distances(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    return d.astype(np.float32)


def _predict_open_world(
    nearest_reg_idx: np.ndarray,
    min_dist: np.ndarray,
    threshold: float,
    unknown_class_idx: int,
) -> np.ndarray:
    pred = np.asarray(nearest_reg_idx, dtype=np.int64).copy()
    pred[np.asarray(min_dist, dtype=np.float64) > float(threshold)] = int(unknown_class_idx)
    return pred


def _predict_open_world_known_user_score(
    nearest_reg_idx: np.ndarray,
    min_dist: np.ndarray,
    score_threshold: float,
    unknown_class_idx: int,
) -> np.ndarray:
    """
    Same rejection semantics as analyze_confidence.open_set_predict on known_user_score:
    accept (assign nearest registered ID) when score >= threshold; else UNKNOWN.

    Here score = 1/(1+d) with d = min L2 distance to any registered prototype.
    """
    min_dist = np.asarray(min_dist, dtype=np.float64)
    scores = ood_distance_to_known_user_score(min_dist)
    pred = np.asarray(nearest_reg_idx, dtype=np.int64).copy()
    pred[scores < float(score_threshold)] = int(unknown_class_idx)
    return pred


def _multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_registered: int,
    unknown_class_idx: int,
) -> Dict[str, float]:
    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    y_p = np.asarray(y_pred, dtype=np.int64).ravel()
    labels = list(range(n_registered + 1))
    _, recalls, _, _ = precision_recall_fscore_support(
        y_t,
        y_p,
        labels=labels,
        average=None,
        zero_division=0,
    )
    rec = np.asarray(recalls, dtype=np.float64)
    if rec.shape[0] != n_registered + 1:
        raise RuntimeError("Unexpected recall vector size in multiclass metrics")
    return {
        "accuracy": float(accuracy_score(y_t, y_p)),
        "balanced_accuracy": float(balanced_accuracy_score(y_t, y_p)),
        "f1_macro": float(f1_score(y_t, y_p, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_t, y_p)),
        "unknown_recall": float(rec[unknown_class_idx]),
        "mean_registered_recall": float(np.mean(rec[:n_registered])),
    }


def _classification_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_registered: int,
    unknown_class_idx: int,
) -> Dict[str, int]:
    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    y_p = np.asarray(y_pred, dtype=np.int64).ravel()

    registered_mask = y_t != unknown_class_idx
    unregistered_mask = y_t == unknown_class_idx

    registered_correct = int(np.sum((y_p == y_t) & registered_mask))
    registered_rejected = int(np.sum((y_p == unknown_class_idx) & registered_mask))
    registered_wrong_registered = int(np.sum((y_p != y_t) & (y_p != unknown_class_idx) & registered_mask))
    unregistered_rejected = int(np.sum((y_p == unknown_class_idx) & unregistered_mask))
    unregistered_assigned_registered = int(np.sum((y_p != unknown_class_idx) & unregistered_mask))

    return {
        "registered_probe_correct_identity": registered_correct,
        "registered_probe_rejected_as_unknown": registered_rejected,
        "registered_probe_misclassified_as_other_registered": registered_wrong_registered,
        "unregistered_correctly_rejected_unknown": unregistered_rejected,
        "unregistered_falsely_assigned_registered": unregistered_assigned_registered,
        "n_registered_probe_samples": int(np.sum(registered_mask)),
        "n_unregistered_samples": int(np.sum(unregistered_mask)),
    }


def _threshold_sweep_open_world(
    nearest_reg_idx: np.ndarray,
    min_dist: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
    *,
    n_registered: int,
    unknown_class_idx: int,
    unknown_recall_weight: float,
    threshold_on: str,
) -> np.ndarray:
    if threshold_on not in ("distance", "known_user_score"):
        raise ValueError(f"threshold_on must be 'distance' or 'known_user_score', got {threshold_on!r}")
    if not (0.0 <= float(unknown_recall_weight) <= 1.0):
        raise ValueError(
            f"unknown_recall_weight must be in [0, 1], got {unknown_recall_weight!r}"
        )
    rows: List[Tuple[float, ...]] = []
    for t in thresholds:
        if threshold_on == "known_user_score":
            pred = _predict_open_world_known_user_score(
                nearest_reg_idx, min_dist, float(t), unknown_class_idx
            )
        else:
            pred = _predict_open_world(nearest_reg_idx, min_dist, float(t), unknown_class_idx)
        m = _multiclass_metrics(
            y_true,
            pred,
            n_registered=n_registered,
            unknown_class_idx=unknown_class_idx,
        )
        urec = float(m["unknown_recall"])
        rrec = float(m["mean_registered_recall"])
        weighted_balance = float(unknown_recall_weight * urec + (1.0 - unknown_recall_weight) * rrec)
        if urec + rrec <= 0.0:
            harmonic_balance = 0.0
        else:
            harmonic_balance = float((2.0 * urec * rrec) / (urec + rrec))
        rows.append(
            (
                float(t),
                m["accuracy"],
                m["balanced_accuracy"],
                m["f1_macro"],
                m["mcc"],
                urec,
                rrec,
                weighted_balance,
                harmonic_balance,
            )
        )
    return np.asarray(rows, dtype=np.float64)


def _argmax_threshold_row(sweep: np.ndarray, criterion: str) -> int:
    if criterion not in _THRESH_CRITERIA:
        raise ValueError(f"criterion must be one of {_THRESH_CRITERIA}, got {criterion!r}")
    col = {
        "accuracy": 1,
        "balanced_accuracy": 2,
        "f1_macro": 3,
        "mcc": 4,
        "weighted_recall_balance": 7,
        "harmonic_recall_balance": 8,
    }[criterion]
    vals = sweep[:, col]
    best = float(np.max(vals))
    tol = 1e-12
    candidates = np.flatnonzero(vals >= best - tol)
    if candidates.size == 1:
        return int(candidates[0])
    sub = sweep[candidates]
    # Tie-break on higher mean registered recall, then unknown recall, then closer to median threshold.
    med = float(np.median(sweep[:, 0]))
    order = np.lexsort(
        (
            -np.abs(sub[:, 0] - med),
            sub[:, 5],
            sub[:, 6],
        )
    )
    return int(candidates[order[-1]])


def _row_at_index(sweep: np.ndarray, idx: int, names: Tuple[str, ...]) -> Dict[str, float]:
    r = sweep[idx]
    return {name: float(r[j]) for j, name in enumerate(names)}


def _equiv_known_user_score_from_distance(d: float) -> float:
    """Boundary score when min distance equals d: same as analyze_confidence."""
    return float(1.0 / (1.0 + float(d)))


def _equiv_distance_cap_from_score_threshold(t_s: float) -> float:
    """Accept when min_dist <= this value iff accept when score >= t_s (score = 1/(1+d))."""
    ts = float(t_s)
    if ts <= 0.0:
        raise ValueError("known_user_score threshold must be > 0")
    return float(1.0 / ts - 1.0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--hdf5_dir", type=str, required=True, help="Fold directory containing unknown split HDF5 files.")
    p.add_argument("--segment_length", type=str, required=True, choices=["1s", "2s", "4s"])
    p.add_argument("--task_type", type=str, required=True, choices=["active", "passive", "both"])
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=192)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dropout_rate", type=float, default=0.1)
    p.add_argument("--model_name", type=str, default="labram_base_patch200_200")
    p.add_argument("--arcface_margin", type=float, default=0.5)
    p.add_argument("--arcface_scale", type=float, default=256.0)
    p.add_argument("--arcface_easy_margin", action="store_true")
    p.add_argument("--register_fraction_of_unknown", type=float, default=0.5)
    p.add_argument("--enrollment_fraction_per_user", type=float, default=0.5)
    p.add_argument(
        "--threshold_criterion",
        type=str,
        default="f1_macro",
        choices=list(_THRESH_CRITERIA),
        help="Metric for selecting open-world distance threshold.",
    )
    p.add_argument(
        "--unknown_recall_weight",
        type=float,
        default=0.5,
        help="Weight in [0,1] for unknown recall in weighted_recall_balance criterion.",
    )
    p.add_argument(
        "--threshold_grid_points",
        type=int,
        default=501,
        help="Number of candidate thresholds between min/max of the chosen signal (distance or score).",
    )
    p.add_argument(
        "--threshold_on",
        type=str,
        default="distance",
        choices=["distance", "known_user_score"],
        help=(
            "Open-world reject signal: raw min L2 distance to nearest prototype ('distance'), or "
            "known_user_score=1/(1+d) as in analyze_confidence ('known_user_score'). "
            "Decisions are equivalent up to the monotone map d↔1/(1+d); grids may pick slightly "
            "different neighbors with finite samples."
        ),
    )
    args = p.parse_args()
    if args.threshold_grid_points < 11:
        raise SystemExit("--threshold_grid_points must be >= 11")
    if not (0.0 <= float(args.unknown_recall_weight) <= 1.0):
        raise SystemExit("--unknown_recall_weight must be in [0, 1]")
    return args


def run_single_task(
    *,
    ckpt_path: Path,
    hdf5_dir: Path,
    segment_length: str,
    task_type: str,
    out_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    ui_unknown, _ = create_user_identification_transform_from_hdf5(
        str(hdf5_dir), unknown_label=UserIdentificationTransform.UNKNOWN_LABEL
    )

    ckpt = _load_checkpoint(ckpt_path, device)
    model, criterion = _build_model_and_criterion(
        ckpt,
        device,
        args.dropout_rate,
        args.model_name,
        args.arcface_margin,
        args.arcface_scale,
        args.arcface_easy_margin,
    )

    all_refs = list_unknown_samples(hdf5_dir, segment_length, task_type)
    registered, unregistered, enroll, probe, unreg_list = split_unknown_participants_for_registration(
        all_refs,
        register_fraction_of_unknown=args.register_fraction_of_unknown,
        enrollment_fraction_per_user=args.enrollment_fraction_per_user,
        seed=args.seed,
    )
    registered_sorted = sorted(registered)
    pid_to_reg_idx = {p: i for i, p in enumerate(registered_sorted)}
    n_registered = len(registered_sorted)
    unknown_class_idx = n_registered
    if n_registered < 1:
        raise RuntimeError("No registered users selected; cannot run classification.")

    reg_proto = _stack_registration_prototypes(
        enroll,
        registered_sorted,
        model,
        criterion,
        device,
        ui_unknown,
        args.batch_size,
        args.num_workers,
    )

    probe_refs: List[UnknownSampleRef] = []
    probe_true_idx: List[int] = []
    for pid in registered_sorted:
        for r in probe[pid]:
            probe_refs.append(r)
            probe_true_idx.append(pid_to_reg_idx[pid])
    probe_true_idx_arr = np.asarray(probe_true_idx, dtype=np.int64)
    unreg_true_idx_arr = np.full(len(unreg_list), unknown_class_idx, dtype=np.int64)

    emb_probe, _ = _embeddings_for_refs(
        probe_refs,
        model,
        criterion,
        device,
        ui_unknown,
        args.batch_size,
        args.num_workers,
    )
    emb_unreg, _ = _embeddings_for_refs(
        unreg_list,
        model,
        criterion,
        device,
        ui_unknown,
        args.batch_size,
        args.num_workers,
    )

    dist_probe = _l2_distance_matrix(emb_probe, reg_proto)
    dist_unreg = _l2_distance_matrix(emb_unreg, reg_proto)

    probe_nearest_idx = np.argmin(dist_probe, axis=1).astype(np.int64)
    unreg_nearest_idx = np.argmin(dist_unreg, axis=1).astype(np.int64)
    probe_min_dist = np.min(dist_probe, axis=1).astype(np.float64)
    unreg_min_dist = np.min(dist_unreg, axis=1).astype(np.float64)

    y_true_all = np.concatenate([probe_true_idx_arr, unreg_true_idx_arr], axis=0)
    nearest_all = np.concatenate([probe_nearest_idx, unreg_nearest_idx], axis=0)
    min_dist_all = np.concatenate([probe_min_dist, unreg_min_dist], axis=0)
    min_score_all = ood_distance_to_known_user_score(min_dist_all)

    threshold_on = str(getattr(args, "threshold_on", "distance"))
    sweep_names = _threshold_sweep_column_names(threshold_on)

    n_grid = int(getattr(args, "threshold_grid_points", 501))
    if threshold_on == "known_user_score":
        s_min = float(np.min(min_score_all))
        s_max = float(np.max(min_score_all))
        if s_max <= s_min:
            thresholds = np.array([s_min], dtype=np.float64)
        else:
            thresholds = np.linspace(s_min, s_max, n_grid, dtype=np.float64)
    else:
        d_min = float(np.min(min_dist_all))
        d_max = float(np.max(min_dist_all))
        if d_max <= d_min:
            thresholds = np.array([d_min], dtype=np.float64)
        else:
            thresholds = np.linspace(d_min, d_max, n_grid, dtype=np.float64)

    sweep = _threshold_sweep_open_world(
        nearest_all,
        min_dist_all,
        y_true_all,
        thresholds,
        n_registered=n_registered,
        unknown_class_idx=unknown_class_idx,
        unknown_recall_weight=float(getattr(args, "unknown_recall_weight", 0.5)),
        threshold_on=threshold_on,
    )

    crit = str(getattr(args, "threshold_criterion", "f1_macro"))
    sel_idx = _argmax_threshold_row(sweep, crit)
    sel_t = float(sweep[sel_idx, 0])
    if threshold_on == "known_user_score":
        pred_selected = _predict_open_world_known_user_score(
            nearest_all, min_dist_all, sel_t, unknown_class_idx
        )
    else:
        pred_selected = _predict_open_world(nearest_all, min_dist_all, sel_t, unknown_class_idx)
    metrics_selected = _multiclass_metrics(
        y_true_all,
        pred_selected,
        n_registered=n_registered,
        unknown_class_idx=unknown_class_idx,
    )
    breakdown_selected = _classification_breakdown(
        y_true_all,
        pred_selected,
        n_registered=n_registered,
        unknown_class_idx=unknown_class_idx,
    )

    acc_idx = _argmax_threshold_row(sweep, "accuracy")
    acc_t = float(sweep[acc_idx, 0])
    if threshold_on == "known_user_score":
        pred_acc = _predict_open_world_known_user_score(
            nearest_all, min_dist_all, acc_t, unknown_class_idx
        )
    else:
        pred_acc = _predict_open_world(nearest_all, min_dist_all, acc_t, unknown_class_idx)
    metrics_acc = _multiclass_metrics(
        y_true_all,
        pred_acc,
        n_registered=n_registered,
        unknown_class_idx=unknown_class_idx,
    )
    breakdown_acc = _classification_breakdown(
        y_true_all,
        pred_acc,
        n_registered=n_registered,
        unknown_class_idx=unknown_class_idx,
    )

    probe_only_metrics = _multiclass_metrics(
        probe_true_idx_arr,
        probe_nearest_idx,
        n_registered=n_registered,
        unknown_class_idx=unknown_class_idx,
    )

    cm_sel = confusion_matrix(
        y_true_all,
        pred_selected,
        labels=list(range(n_registered + 1)),
    )
    unknown_true_idx = unknown_class_idx
    unknown_pred_rate = float(cm_sel[unknown_true_idx, unknown_true_idx] / max(1, int(np.sum(cm_sel[unknown_true_idx]))))

    selected_threshold_report: Dict[str, Any] = {
        "sweep_row": _row_at_index(sweep, sel_idx, sweep_names),
        "metrics": metrics_selected,
        "assignment_summary": breakdown_selected,
        "unknown_rejection_rate": unknown_pred_rate,
    }
    ref_threshold_report: Dict[str, Any] = {
        "sweep_row": _row_at_index(sweep, acc_idx, sweep_names),
        "metrics": metrics_acc,
        "assignment_summary": breakdown_acc,
    }
    if threshold_on == "known_user_score":
        selected_threshold_report["known_user_score_threshold"] = sel_t
        selected_threshold_report["equivalent_distance_cap_for_accept"] = _equiv_distance_cap_from_score_threshold(
            sel_t
        )
        ref_threshold_report["known_user_score_threshold"] = acc_t
        ref_threshold_report["equivalent_distance_cap_for_accept"] = _equiv_distance_cap_from_score_threshold(acc_t)
    else:
        selected_threshold_report["distance_threshold"] = sel_t
        selected_threshold_report["equivalent_known_user_score_threshold"] = _equiv_known_user_score_from_distance(sel_t)
        ref_threshold_report["distance_threshold"] = acc_t
        ref_threshold_report["equivalent_known_user_score_threshold"] = _equiv_known_user_score_from_distance(acc_t)

    report: Dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "hdf5_dir": str(hdf5_dir),
        "segment_length": segment_length,
        "task_type": task_type,
        "registered_participant_ids": registered_sorted,
        "unregistered_participant_ids": sorted(unregistered),
        "n_unknown_participants_total": len(set(r.participant_id for r in all_refs)),
        "n_registered_participants": n_registered,
        "n_unregistered_participants": int(len(unregistered)),
        "register_fraction_of_unknown": args.register_fraction_of_unknown,
        "enrollment_fraction_per_user": args.enrollment_fraction_per_user,
        "seed": args.seed,
        "n_probe_registered_segments": int(emb_probe.shape[0]),
        "n_unregistered_segments": int(emb_unreg.shape[0]),
        "registered_user_classification_probe_only": {
            "definition": "Closed-set classification over registered IDs only (probe segments only).",
            "accuracy": float(probe_only_metrics["accuracy"]),
            "balanced_accuracy": float(probe_only_metrics["balanced_accuracy"]),
            "f1_macro": float(probe_only_metrics["f1_macro"]),
            "mcc": float(probe_only_metrics["mcc"]),
        },
        "open_world_user_classification": {
            "definition": "K+1 classes: K registered IDs + 1 UNKNOWN class for all unregistered users.",
            "threshold_on": threshold_on,
            "underlying_distance": "min_l2_distance_to_any_registered_prototype",
            "known_user_score": (
                "1/(1 + min_l2_distance_to_nearest_registered_prototype); "
                "same map as user_identification.analyze_confidence.ood_distance_to_known_user_score"
            ),
            "selection_criterion": crit,
            "unknown_recall_weight": float(getattr(args, "unknown_recall_weight", 0.5)),
            "selected_threshold": selected_threshold_report,
            "reference_max_plain_accuracy": ref_threshold_report,
            "threshold_sweep_column_names": list(sweep_names),
        },
    }

    with open(out_dir / "registration_open_set_report.json", "w") as f:
        json.dump(report, f, indent=2)

    np.savez_compressed(
        out_dir / "registration_eval_arrays.npz",
        registration_prototypes=reg_proto,
        dist_probe_to_registered=dist_probe.astype(np.float32),
        dist_unregistered_to_registered=dist_unreg.astype(np.float32),
        threshold_sweep=sweep.astype(np.float32),
        y_true_open_world=y_true_all.astype(np.int32),
        y_pred_selected=pred_selected.astype(np.int32),
        y_pred_max_plain_accuracy=pred_acc.astype(np.int32),
        nearest_registered_prediction=nearest_all.astype(np.int32),
        min_distance=min_dist_all.astype(np.float32),
        min_known_user_score=min_score_all.astype(np.float32),
        emb_probe=emb_probe.astype(np.float32),
        emb_unregistered=emb_unreg.astype(np.float32),
    )

    print(json.dumps(report, indent=2))
    print(f"Wrote: {out_dir / 'registration_open_set_report.json'}")
    return report


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; using CPU.", file=sys.stderr)

    hdf5_dir = Path(args.hdf5_dir).resolve()
    if not hdf5_dir.is_dir():
        raise SystemExit(f"hdf5_dir not found: {hdf5_dir}")
    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task_type == "both":
        combined: Dict[str, Any] = {}
        for tt in ("active", "passive"):
            sub = out_dir / tt
            sub.mkdir(parents=True, exist_ok=True)
            combined[tt] = run_single_task(
                ckpt_path=ckpt_path,
                hdf5_dir=hdf5_dir,
                segment_length=args.segment_length,
                task_type=tt,
                out_dir=sub,
                device=device,
                args=args,
            )
        both_path = out_dir / "registration_open_set_report_both.json"
        with open(both_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"Wrote combined report: {both_path}")
    else:
        run_single_task(
            ckpt_path=ckpt_path,
            hdf5_dir=hdf5_dir,
            segment_length=args.segment_length,
            task_type=args.task_type,
            out_dir=out_dir,
            device=device,
            args=args,
        )


if __name__ == "__main__":
    main()
