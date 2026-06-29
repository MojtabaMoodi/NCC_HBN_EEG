"""
Embedding diagnostics for FaceNet triplet user-identification training.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
from einops import rearrange

from CNN.models.triplet_loss import compute_embedding_separation_stats


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


@torch.no_grad()
def collect_embeddings_from_loader(
    model: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    *,
    max_batches: Optional[int] = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract L2-normalized embeddings and labels from a LaBraM or collated dict loader."""
    model.eval()
    model_to_use = _unwrap_model(model)
    if not hasattr(model_to_use, "extract_features"):
        raise AttributeError(
            f"{model_to_use.__class__.__name__} must implement extract_features()"
        )

    all_emb = []
    all_labels = []
    for batch_idx, batch in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        if isinstance(batch, dict):
            eeg = batch["eeg_data"]
            if "user_identification" not in batch:
                raise KeyError(
                    "Batch dict missing 'user_identification'. "
                    f"Available keys: {list(batch.keys())}"
                )
            labels = batch["user_identification"]
        elif isinstance(batch, (list, tuple)):
            eeg, labels = batch[0], batch[1]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)!r}")

        eeg = eeg.float().to(device, non_blocking=True) / 100
        eeg = rearrange(eeg, "B N (A T) -> B N A T", T=200)
        labels = labels.to(device, non_blocking=True).long().view(-1)
        features = model_to_use.extract_features(eeg)
        all_emb.append(features.detach().cpu())
        all_labels.append(labels.detach().cpu())

    if not all_emb:
        raise RuntimeError("No batches collected for embedding diagnostics.")

    embeddings = torch.cat(all_emb, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return embeddings, labels


@torch.no_grad()
def diagnose_embedding_separation(
    model: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    *,
    max_batches: Optional[int] = 50,
    header: str = "Embedding separation",
) -> Dict[str, float]:
    """
    Compute global intra/inter-class distance stats on up to ``max_batches`` loader batches.

    Returns dict with mean_intra_distance, mean_inter_distance, inter_over_intra_ratio.
    """
    embeddings, labels = collect_embeddings_from_loader(
        model, data_loader, device, max_batches=max_batches
    )
    emb = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # Global pairwise stats (subsample if huge)
    max_pairs = 200_000
    n = emb.size(0)
    if n * (n - 1) // 2 > max_pairs and n > 512:
        idx = torch.randperm(n)[:512]
        emb = emb[idx]
        labels = labels[idx]
        n = emb.size(0)

    from CNN.models.triplet_loss import pairwise_squared_l2

    dist_sq = pairwise_squared_l2(emb)
    dist = dist_sq.clamp(min=0.0).sqrt()
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    not_self = ~torch.eye(n, dtype=torch.bool)
    intra_mask = same & not_self
    inter_mask = ~same
    if not intra_mask.any() or not inter_mask.any():
        raise RuntimeError(
            f"{header}: insufficient class diversity in {n} samples for separation stats."
        )

    mean_intra = float(dist[intra_mask].mean().item())
    mean_inter = float(dist[inter_mask].mean().item())
    ratio = mean_inter / mean_intra if mean_intra > 0 else float("nan")
    stats = {
        "mean_intra_distance": mean_intra,
        "mean_inter_distance": mean_inter,
        "inter_over_intra_ratio": float(ratio),
        "num_samples": float(n),
    }
    print(
        f"  {header}: intra={mean_intra:.4f}, inter={mean_inter:.4f}, "
        f"ratio={ratio:.4f} (target > 1.0; collapsed ~ 1.0)"
    )
    return stats


def compare_triplet_mining_on_batch(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
) -> Dict[str, Dict[str, float]]:
    """
    Compare mining strategies on one batch (for offline diagnosis / unit checks).

    Returns per-strategy loss and active_triplet_fraction.
    """
    from CNN.models.triplet_loss import TripletLoss

    results: Dict[str, Dict[str, float]] = {}
    for mining in ("semi_hard", "all", "batch_hard"):
        crit = TripletLoss(margin=margin, mining=mining)  # type: ignore[arg-type]
        loss = crit(embeddings, labels)
        results[mining] = {
            "loss": float(loss.item()),
            **crit.last_batch_stats,
        }
    return results
