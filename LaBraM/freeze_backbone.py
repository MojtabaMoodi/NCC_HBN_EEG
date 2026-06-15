"""Freeze LaBraM backbone for linear-probe / partial fine-tuning."""
from __future__ import annotations

from typing import Any, List, Tuple

import torch.nn as nn


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (trainable_parameter_count, total_parameter_count)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _task_head_modules(model: Any) -> List[nn.Module]:
    if getattr(model, "multi_output", False):
        heads: List[nn.Module] = []
        if getattr(model, "gender_head", None) is not None:
            heads.append(model.gender_head)
        if getattr(model, "age_head", None) is not None:
            heads.append(model.age_head)
        return heads
    head = getattr(model, "head", None)
    if head is not None:
        return [head]
    return []


def apply_finetune_freeze(model: Any, unfreeze_last_n_blocks: int = 0) -> Tuple[int, int]:
    """
    Freeze the full backbone and train only task head(s) by default.

    With ``unfreeze_last_n_blocks > 0``, also unfreeze the last N transformer blocks
    and ``fc_norm`` (when present).
    """
    if unfreeze_last_n_blocks < 0:
        raise ValueError(f"unfreeze_last_n_blocks must be >= 0, got {unfreeze_last_n_blocks}")

    for param in model.parameters():
        param.requires_grad = False

    for head in _task_head_modules(model):
        for param in head.parameters():
            param.requires_grad = True

    blocks = getattr(model, "blocks", None)
    if unfreeze_last_n_blocks > 0:
        if blocks is None:
            raise AttributeError("Model has no 'blocks' attribute; cannot unfreeze transformer blocks.")
        n_blocks = len(blocks)
        if unfreeze_last_n_blocks > n_blocks:
            raise ValueError(
                f"unfreeze_last_n_blocks={unfreeze_last_n_blocks} exceeds transformer depth={n_blocks}"
            )
        for block in blocks[n_blocks - unfreeze_last_n_blocks :]:
            for param in block.parameters():
                param.requires_grad = True
        fc_norm = getattr(model, "fc_norm", None)
        if fc_norm is not None:
            for param in fc_norm.parameters():
                param.requires_grad = True

    # Head-only: skip backbone autograd (faster backward; forward cost unchanged).
    model.linear_probe_forward = unfreeze_last_n_blocks == 0

    trainable, total = count_parameters(model)
    if trainable == 0:
        raise RuntimeError(
            "apply_finetune_freeze left no trainable parameters; task head(s) must remain trainable."
        )
    return trainable, total
