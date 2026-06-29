"""
Shared helpers for LaBraM user-identification embedding losses (ArcFace / FaceNet triplet).
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Tuple, Union

import torch
import torch.nn as nn

from CNN.models.arcface_loss import ArcFaceLoss
from CNN.models.triplet_loss import PrototypeClassifier, TripletTrainingCriterion

LossType = Literal["arcface", "triplet", "ce"]


def resolve_loss_type(
    *,
    loss_type: str | None = None,
    use_arcface: bool = False,
) -> LossType:
    if loss_type is not None:
        if loss_type not in ("arcface", "triplet", "ce"):
            raise ValueError(
                f"loss_type must be 'arcface', 'triplet', or 'ce', got {loss_type!r}"
            )
        return loss_type  # type: ignore[return-value]
    return "arcface" if use_arcface else "ce"


def uses_embedding_logits(loss_type: LossType) -> bool:
    return loss_type in ("arcface", "triplet")


def build_training_criterion(
    loss_type: LossType,
    num_classes: int,
    embedding_dim: int,
    *,
    arcface_margin: float = 0.5,
    arcface_scale: float = 256.0,
    arcface_easy_margin: bool = False,
    triplet_margin: float = 0.2,
    triplet_mining: str = "semi_hard",
    prototype_momentum: float = 0.9,
    triplet_logit_scale: float = 10.0,
    device: torch.device | None = None,
) -> nn.Module:
    if loss_type == "arcface":
        criterion = ArcFaceLoss(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            margin=arcface_margin,
            scale=arcface_scale,
            easy_margin=arcface_easy_margin,
        )
    elif loss_type == "triplet":
        if triplet_mining not in ("batch_hard", "semi_hard", "all"):
            raise ValueError(
                f"triplet_mining must be 'batch_hard', 'semi_hard', or 'all', got {triplet_mining!r}"
            )
        criterion = TripletTrainingCriterion(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            margin=triplet_margin,
            mining=triplet_mining,  # type: ignore[arg-type]
            prototype_momentum=prototype_momentum,
            logit_scale=triplet_logit_scale,
        )
    elif loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type!r}")

    if device is not None:
        criterion = criterion.to(device)
    return criterion


def load_criterion_from_checkpoint(
    checkpoint: Dict[str, Any],
    num_classes: int,
    embedding_dim: int,
    device: torch.device,
    *,
    arcface_margin: float = 0.5,
    arcface_scale: float = 256.0,
    arcface_easy_margin: bool = False,
    triplet_margin: float | None = None,
    triplet_logit_scale: float = 10.0,
) -> Tuple[nn.Module, LossType]:
    """
    Rebuild criterion from a training checkpoint for evaluation / analysis.

    Returns (criterion, loss_type).
    """
    loss_type: LossType = checkpoint.get("loss_type", "arcface")
    if loss_type not in ("arcface", "triplet"):
        raise ValueError(
            f"Checkpoint loss_type={loss_type!r} is not supported for embedding eval "
            f"(expected 'arcface' or 'triplet')."
        )

    ckpt_num = checkpoint.get("num_classes")
    ckpt_dim = checkpoint.get("embedding_dim")
    if ckpt_num is not None and int(ckpt_num) != num_classes:
        raise ValueError(
            f"Checkpoint num_classes={ckpt_num} != dataset num_classes={num_classes}"
        )
    if ckpt_dim is not None and int(ckpt_dim) != embedding_dim:
        raise ValueError(
            f"Checkpoint embedding_dim={ckpt_dim} != model embedding_dim={embedding_dim}"
        )

    if loss_type == "arcface":
        if "arcface_state_dict" not in checkpoint:
            raise KeyError(
                "ArcFace checkpoint missing 'arcface_state_dict'. "
                "Cannot run embedding-logit evaluation."
            )
        margin = float(checkpoint.get("arcface_margin", arcface_margin))
        scale = float(checkpoint.get("arcface_scale", arcface_scale))
        easy = bool(checkpoint.get("arcface_easy_margin", arcface_easy_margin))
        criterion = ArcFaceLoss(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            margin=margin,
            scale=scale,
            easy_margin=easy,
        )
        criterion.load_state_dict(checkpoint["arcface_state_dict"], strict=True)
    else:
        margin = float(
            triplet_margin
            if triplet_margin is not None
            else checkpoint.get("triplet_margin", 0.2)
        )
        logit_scale = float(checkpoint.get("triplet_logit_scale", triplet_logit_scale))
        momentum = float(checkpoint.get("prototype_momentum", 0.9))
        criterion = PrototypeClassifier(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            momentum=momentum,
            logit_scale=logit_scale,
        )
        if "prototype_state_dict" in checkpoint:
            criterion.load_state_dict(checkpoint["prototype_state_dict"], strict=True)
        elif "triplet_criterion_state_dict" in checkpoint:
            triplet_full = TripletTrainingCriterion(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                margin=margin,
                prototype_momentum=momentum,
                logit_scale=logit_scale,
            )
            triplet_full.load_state_dict(
                checkpoint["triplet_criterion_state_dict"], strict=True
            )
            criterion = triplet_full.prototype_classifier
        else:
            raise KeyError(
                "Triplet checkpoint missing 'prototype_state_dict' (or "
                "'triplet_criterion_state_dict'). Re-train or compute prototypes "
                "from the train split before closed-set eval."
            )

    criterion = criterion.to(device)
    criterion.eval()
    return criterion, loss_type


def checkpoint_extra_state(
    loss_type: LossType,
    criterion: nn.Module,
) -> Dict[str, Any]:
    """Serialize loss-specific tensors for resume / eval."""
    out: Dict[str, Any] = {"loss_type": loss_type}
    if loss_type == "arcface":
        if not isinstance(criterion, ArcFaceLoss):
            raise TypeError(f"Expected ArcFaceLoss, got {type(criterion).__name__}")
        out["arcface_state_dict"] = criterion.state_dict()
        out["arcface_margin"] = float(criterion.margin)
        out["arcface_scale"] = float(criterion.scale)
        out["arcface_easy_margin"] = bool(criterion.easy_margin)
    elif loss_type == "triplet":
        if isinstance(criterion, TripletTrainingCriterion):
            out["triplet_criterion_state_dict"] = criterion.state_dict()
            out["prototype_state_dict"] = criterion.prototype_classifier.state_dict()
            out["triplet_margin"] = float(criterion.triplet_loss.margin)
            out["triplet_mining"] = criterion.mining
            out["prototype_momentum"] = float(criterion.prototype_classifier.momentum)
            out["triplet_logit_scale"] = float(criterion.prototype_classifier.logit_scale)
        elif isinstance(criterion, PrototypeClassifier):
            out["prototype_state_dict"] = criterion.state_dict()
            out["prototype_momentum"] = float(criterion.momentum)
            out["triplet_logit_scale"] = float(criterion.logit_scale)
        else:
            raise TypeError(
                f"Expected TripletTrainingCriterion or PrototypeClassifier, "
                f"got {type(criterion).__name__}"
            )
    return out
