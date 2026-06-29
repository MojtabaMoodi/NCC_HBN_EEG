"""
FaceNet-style triplet loss and prototype-based recognition (FaceNet-pure inference).

Reference: Schroff et al., "FaceNet: A Unified Embedding for Face Recognition
and Clustering" (2015), https://arxiv.org/pdf/1503.03832

Training: triplet loss with online mining within each mini-batch.
Default mining is **semi_hard** (FaceNet paper). Optional **batch_hard** for harder mining.

Inference / closed-set metrics: nearest L2-normalized class prototype (same geometry
used for open-set OOD in analyze_confidence.py).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

MiningStrategy = Literal["batch_hard", "semi_hard", "all"]


def batch_local_nearest_prototype_accuracy(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Fraction correct when classifying each sample to the nearest batch-computed prototype.

    FaceNet trains with triplet loss only; batch-local accuracy is the meaningful train
    monitor (argmax over ~2500 online EMA prototypes collapses to ~1/N chance).
    """
    if embeddings.dim() != 2:
        raise ValueError(f"embeddings must be 2D, got {embeddings.shape}")
    labels = labels.view(-1)
    if embeddings.size(0) != labels.size(0):
        raise ValueError("embeddings and labels batch size mismatch")
    if embeddings.size(0) < 2:
        return torch.tensor(0.0, device=embeddings.device)

    emb = F.normalize(embeddings, p=2, dim=1)
    batch_classes = labels.unique()
    if batch_classes.numel() < 2:
        return torch.tensor(0.0, device=embeddings.device)

    protos = []
    for c in batch_classes:
        c_int = int(c.item())
        mask = labels == c_int
        if int(mask.sum()) < 1:
            continue
        protos.append(F.normalize(emb[mask].mean(dim=0), dim=0))
    if len(protos) < 2:
        return torch.tensor(0.0, device=embeddings.device)

    protos_t = torch.stack(protos, dim=0)
    dist_sq = pairwise_squared_l2(torch.cat([emb, protos_t], dim=0))
    n = emb.size(0)
    cross = dist_sq[:n, n:]
    pred_cols = cross.argmin(dim=1)
    pred_classes = batch_classes[pred_cols]
    return (pred_classes == labels).float().mean()


def pairwise_squared_l2(emb: torch.Tensor) -> torch.Tensor:
    """
    Pairwise squared L2 distances for row-wise L2-normalized embeddings.

    For unit vectors: ||a - b||^2 = 2 - 2 * a·b
    """
    if emb.dim() != 2:
        raise ValueError(f"emb must be 2D (batch, dim), got shape {emb.shape}")
    gram = emb @ emb.t()
    sq_norm = (emb ** 2).sum(dim=1, keepdim=True)
    dist_sq = sq_norm + sq_norm.t() - 2.0 * gram
    return dist_sq.clamp(min=0.0)


def compute_embedding_separation_stats(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    """
    Diagnostic: mean intra-class vs inter-class L2 distance on a batch or loader slice.

    Healthy metric learning should show mean_inter > mean_intra (ratio > 1).
    Collapsed embeddings often have ratio ~ 1.
    """
    if embeddings.dim() != 2:
        raise ValueError(f"embeddings must be 2D, got {embeddings.shape}")
    if labels.dim() != 1:
        raise ValueError(f"labels must be 1D, got {labels.shape}")
    if embeddings.size(0) != labels.size(0):
        raise ValueError("embeddings and labels batch size mismatch")
    if embeddings.size(0) < 2:
        raise ValueError("need at least 2 samples for separation stats")

    emb = F.normalize(embeddings.detach(), p=2, dim=1)
    labels = labels.view(-1)
    dist_sq = pairwise_squared_l2(emb)
    dist = dist_sq.clamp(min=0.0).sqrt()

    batch_size = emb.size(0)
    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
    not_self = ~torch.eye(batch_size, dtype=torch.bool, device=emb.device)
    intra_mask = same_class & not_self
    inter_mask = ~same_class

    if not intra_mask.any():
        raise RuntimeError(
            "No intra-class pairs in batch for separation stats. "
            "Need at least one class with >= 2 samples."
        )
    if not inter_mask.any():
        raise RuntimeError(
            "No inter-class pairs in batch for separation stats. "
            "Need at least two distinct classes."
        )

    mean_intra = float(dist[intra_mask].mean().item())
    mean_inter = float(dist[inter_mask].mean().item())
    if mean_intra <= 0.0:
        raise RuntimeError(f"mean_intra distance is non-positive: {mean_intra}")
    ratio = mean_inter / mean_intra
    return {
        "mean_intra_distance": mean_intra,
        "mean_inter_distance": mean_inter,
        "inter_over_intra_ratio": float(ratio),
    }


class TripletLoss(nn.Module):
    """
    FaceNet triplet loss (Eq. 3) with batch-wise mining.

    Mining strategies:
    - **semi_hard** (FaceNet paper default): all anchor-positive pairs, semi-hard negative.
    - **batch_hard**: per anchor, hardest positive + hardest negative in batch.
    - **all**: all AP pairs, hardest negative in batch.
    """

    def __init__(
        self,
        margin: float = 0.2,
        mining: MiningStrategy = "semi_hard",
    ):
        super().__init__()
        if margin <= 0:
            raise ValueError(f"triplet margin must be > 0, got {margin}")
        if mining not in ("batch_hard", "semi_hard", "all"):
            raise ValueError(
                f"mining must be 'batch_hard', 'semi_hard', or 'all', got {mining!r}"
            )
        self.margin = float(margin)
        self.mining = mining
        self.last_batch_stats: Dict[str, float] = {}

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.dim() != 2:
            raise ValueError(
                f"embeddings must be 2D (batch, dim), got shape {embeddings.shape}"
            )
        if labels.dim() != 1:
            raise ValueError(f"labels must be 1D (batch,), got shape {labels.shape}")
        if embeddings.size(0) != labels.size(0):
            raise ValueError(
                f"batch size mismatch: embeddings {embeddings.size(0)} vs labels {labels.size(0)}"
            )
        batch_size = embeddings.size(0)
        if batch_size < 2:
            raise ValueError(
                f"triplet loss requires batch_size >= 2, got {batch_size}"
            )

        emb = F.normalize(embeddings, p=2, dim=1)
        labels = labels.view(-1)
        dist_sq = pairwise_squared_l2(emb)

        if self.mining == "batch_hard":
            loss, stats = self._batch_hard_loss(dist_sq, labels)
        else:
            loss, stats = self._pairwise_mined_loss(dist_sq, labels)

        self.last_batch_stats = stats
        return loss

    def _batch_hard_loss(
        self,
        dist_sq: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = dist_sq.size(0)
        device = dist_sq.device
        same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
        not_self = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        ap_mask = same_class & not_self
        diff_class = ~same_class

        if not ap_mask.any():
            raise RuntimeError(
                "No anchor-positive pairs in batch. Triplet training requires P×K sampling "
                "with K >= 2 so each mini-batch contains multiple segments per user."
            )
        if not diff_class.any():
            raise RuntimeError(
                "No negative pairs in batch. Need at least two distinct user classes."
            )

        pos_sq = dist_sq.masked_fill(~ap_mask, float("-inf")).max(dim=1).values
        neg_sq = dist_sq.masked_fill(~diff_class, float("inf")).min(dim=1).values
        valid = torch.isfinite(pos_sq) & torch.isfinite(neg_sq)
        if not bool(valid.any()):
            raise RuntimeError("No valid batch-hard triplets in mini-batch.")

        pos_sq = pos_sq[valid]
        neg_sq = neg_sq[valid]
        per_anchor = F.relu(pos_sq - neg_sq + self.margin)
        active = per_anchor > 0
        stats = {
            "active_triplet_fraction": float(active.float().mean().item()),
            "mean_d_ap": float(pos_sq.clamp(min=0.0).sqrt().mean().item()),
            "mean_d_an": float(neg_sq.sqrt().mean().item()),
            "mean_triplet_loss": float(per_anchor.mean().item()),
            "num_valid_anchors": float(valid.sum().item()),
        }
        return per_anchor.mean(), stats

    def _pairwise_mined_loss(
        self,
        dist_sq: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = dist_sq.size(0)
        device = dist_sq.device
        same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
        diff_class = ~same_class
        not_self = ~torch.eye(batch_size, dtype=torch.bool, device=device)

        ap_mask = same_class & not_self
        if not ap_mask.any():
            raise RuntimeError(
                "No anchor-positive pairs in batch. Triplet training requires P×K sampling "
                "with K >= 2 so each mini-batch contains multiple segments per user."
            )

        losses = []
        ap_indices = ap_mask.nonzero(as_tuple=False)
        for idx in range(ap_indices.size(0)):
            a = int(ap_indices[idx, 0].item())
            p = int(ap_indices[idx, 1].item())
            d_ap = dist_sq[a, p]

            neg_mask = diff_class[a]
            if not neg_mask.any():
                continue

            if self.mining == "semi_hard":
                semi_hard = neg_mask & (dist_sq[a] > d_ap)
                if not semi_hard.any():
                    continue
                d_an = dist_sq[a].masked_fill(~semi_hard, float("inf")).min()
            else:
                d_an = dist_sq[a].masked_fill(~neg_mask, float("inf")).min()

            if not math.isfinite(float(d_an.item())):
                continue

            losses.append(F.relu(d_ap - d_an + self.margin))

        if not losses:
            raise RuntimeError(
                f"No valid triplets after {self.mining} mining. "
                "Try batch_hard mining or larger P×K batches."
            )

        stacked = torch.stack(losses)
        active = stacked > 0
        stats = {
            "active_triplet_fraction": float(active.float().mean().item()),
            "mean_triplet_loss": float(stacked.mean().item()),
            "num_valid_triplets": float(stacked.numel()),
        }
        return stacked.mean(), stats


class PrototypeClassifier(nn.Module):
    """
    FaceNet-pure recognition head: L2-normalized class prototypes and nearest-prototype logits.

    Prototypes are updated via EMA during training. At open-set evaluation the same
    centroid geometry is used as in analyze_confidence.compute_centroids_from_embeddings.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        momentum: float = 0.9,
        logit_scale: float = 10.0,
    ):
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {num_classes}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {embedding_dim}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if logit_scale <= 0:
            raise ValueError(f"logit_scale must be > 0, got {logit_scale}")

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.momentum = float(momentum)
        self.logit_scale = float(logit_scale)

        self.register_buffer(
            "prototypes",
            torch.zeros(num_classes, embedding_dim),
        )
        self.register_buffer(
            "prototype_counts",
            torch.zeros(num_classes, dtype=torch.long),
        )

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """EMA update of per-class prototypes from a batch of embeddings."""
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2D, got {embeddings.shape}")
        if labels.dim() != 1:
            raise ValueError(f"labels must be 1D, got {labels.shape}")
        if embeddings.size(0) != labels.size(0):
            raise ValueError("embeddings and labels batch size mismatch")
        if embeddings.size(1) != self.embedding_dim:
            raise ValueError(
                f"embedding dim mismatch: got {embeddings.size(1)}, expected {self.embedding_dim}"
            )

        emb = F.normalize(embeddings.detach(), p=2, dim=1)
        labels = labels.view(-1)
        if labels.max() >= self.num_classes or labels.min() < 0:
            invalid = labels[(labels >= self.num_classes) | (labels < 0)]
            raise ValueError(
                f"Invalid labels for prototype update: {invalid.cpu().numpy()}. "
                f"Expected range [0, {self.num_classes - 1}]."
            )

        for c in labels.unique():
            c_int = int(c.item())
            mask = labels == c_int
            batch_mean = F.normalize(emb[mask].mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0)
            count = int(self.prototype_counts[c_int].item())
            if count == 0:
                self.prototypes[c_int] = batch_mean
            else:
                m = self.momentum
                updated = m * self.prototypes[c_int] + (1.0 - m) * batch_mean
                self.prototypes[c_int] = F.normalize(updated, p=2, dim=0)
            self.prototype_counts[c_int] = count + int(mask.sum().item())

    def set_prototypes_from_numpy(
        self,
        centroids: torch.Tensor,
        *,
        require_all_classes: bool = False,
    ) -> None:
        """Load L2-normalized centroids computed offline (e.g. full train pass)."""
        if centroids.shape != (self.num_classes, self.embedding_dim):
            raise ValueError(
                f"centroids shape must be ({self.num_classes}, {self.embedding_dim}), "
                f"got {tuple(centroids.shape)}"
            )
        normalized = F.normalize(centroids.to(dtype=self.prototypes.dtype), p=2, dim=1)
        if require_all_classes:
            norms = normalized.norm(dim=1)
            if (norms < 1e-6).any():
                bad = (norms < 1e-6).nonzero(as_tuple=True)[0].cpu().tolist()
                raise ValueError(
                    f"centroids missing or zero for class indices: {bad[:20]}"
                    + ("..." if len(bad) > 20 else "")
                )
        self.prototypes.copy_(normalized)
        self.prototype_counts.fill_(1)

    def compute_logits(
        self,
        embeddings: torch.Tensor,
        *,
        allow_partial_prototypes: bool = False,
    ) -> torch.Tensor:
        """
        Nearest-prototype logits: -scale * ||e - prototype_k||^2 (higher = closer class).

        When allow_partial_prototypes=True (training batches only), classes without an
        EMA prototype yet receive -inf logits so argmax skips them. Eval must use
        allow_partial_prototypes=False after a full train-split prototype refresh.
        """
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2D, got {embeddings.shape}")
        if embeddings.size(1) != self.embedding_dim:
            raise ValueError(
                f"embedding dim mismatch: got {embeddings.size(1)}, expected {self.embedding_dim}"
            )

        uninitialized = self.prototype_counts == 0
        if uninitialized.all():
            raise RuntimeError(
                "All class prototypes are uninitialized. Run training updates or "
                "load_prototypes before compute_logits."
            )
        if uninitialized.any() and not allow_partial_prototypes:
            missing = uninitialized.nonzero(as_tuple=True)[0].cpu().tolist()
            raise RuntimeError(
                f"Prototypes uninitialized for {int(uninitialized.sum())} classes "
                f"(e.g. indices {missing[:10]}). Cannot compute logits. "
                f"For triplet training batches use allow_partial_prototypes=True; "
                f"for val/test refresh prototypes from the full train split first."
            )

        emb = F.normalize(embeddings, p=2, dim=1)
        prototypes = F.normalize(self.prototypes, p=2, dim=1)
        dist_sq = pairwise_squared_l2(
            torch.cat([emb, prototypes], dim=0)
        )
        n = emb.size(0)
        cross = dist_sq[:n, n:]
        logits = -self.logit_scale * cross
        if allow_partial_prototypes and uninitialized.any():
            logits = logits.masked_fill(uninitialized.unsqueeze(0), float("-inf"))
        return logits

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Monitoring loss: cross-entropy on nearest-prototype logits (not used for training)."""
        logits = self.compute_logits(embeddings)
        return F.cross_entropy(logits, labels.view(-1).long())


class TripletTrainingCriterion(nn.Module):
    """
    Training wrapper: triplet loss + EMA prototype updates; eval via compute_logits().
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        margin: float = 0.2,
        mining: MiningStrategy = "semi_hard",
        prototype_momentum: float = 0.9,
        logit_scale: float = 10.0,
    ):
        super().__init__()
        self.triplet_loss = TripletLoss(margin=margin, mining=mining)
        self.prototype_classifier = PrototypeClassifier(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            momentum=prototype_momentum,
            logit_scale=logit_scale,
        )
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.mining = mining

    @property
    def last_batch_stats(self) -> Dict[str, float]:
        return dict(self.triplet_loss.last_batch_stats)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # FaceNet-pure training: triplet loss only. Class prototypes for eval are computed
        # offline from the full train split (refresh_prototypes_from_train_hdf5), not online EMA.
        return self.triplet_loss(embeddings, labels)

    def compute_logits(
        self,
        embeddings: torch.Tensor,
        *,
        allow_partial_prototypes: bool = False,
    ) -> torch.Tensor:
        return self.prototype_classifier.compute_logits(
            embeddings, allow_partial_prototypes=allow_partial_prototypes
        )

    def classification_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Eval monitoring: CE on nearest-prototype logits (does not update prototypes)."""
        logits = self.compute_logits(embeddings, allow_partial_prototypes=False)
        return F.cross_entropy(logits, labels.view(-1).long())
