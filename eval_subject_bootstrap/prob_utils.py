"""Classification logit → probability helpers for subject-level evaluation."""

from __future__ import annotations

import numpy as np
import torch


def logits_to_probabilities(logits: torch.Tensor, head_dim: int) -> np.ndarray:
    """Map classification head outputs to class probabilities.

    ``head_dim`` is the checkpoint head width (``model_config["num_classes"]``),
    not the label-space size:

    - ``head_dim == 1`` (LaBraM gender / BCEWithLogits): sigmoid on the single
      logit, then expand to ``[P(class=0), P(class=1)]``.
    - ``head_dim >= 2`` (softmax classifiers): softmax over the logit vector.

    Passing the label-space size (2) for a binary head incorrectly routes a
    ``[B, 1]`` logit through softmax, which collapses to a constant class-0
    prediction.
    """
    if head_dim < 1:
        raise ValueError(f"head_dim must be >= 1, got {head_dim}")

    if head_dim == 1:
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)
        if logits.shape[-1] != 1:
            raise ValueError(
                f"Binary classification head expects logits with last dim 1, "
                f"got shape {tuple(logits.shape)}"
            )
        prob_positive = torch.sigmoid(logits).detach().cpu().numpy()
        return np.concatenate([1.0 - prob_positive, prob_positive], axis=-1)

    if logits.ndim != 2 or logits.shape[-1] != head_dim:
        raise ValueError(
            f"Multiclass head (head_dim={head_dim}) expects logits shape [B, {head_dim}], "
            f"got {tuple(logits.shape)}"
        )
    return torch.softmax(logits, dim=-1).detach().cpu().numpy()
