"""
Map human-readable backbone names to CNN ``ModelFactory`` model_type strings.

Used by ``train_user_identification_backbone.py`` so backbone choice and factory
stay in one place (DRY).
"""

from __future__ import annotations

from typing import Dict, FrozenSet

# Keys are CLI values (lowercase); values are ModelFactory.create_model ``model_type``.
BACKBONE_TO_MODEL_TYPE: Dict[str, str] = {
    "cnn": "user_identification_cnn",
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
}

SUPPORTED_BACKBONES: FrozenSet[str] = frozenset(BACKBONE_TO_MODEL_TYPE.keys())


def resolve_model_type(backbone: str) -> str:
    """
    Return the ``model_type`` string for ``ModelFactory.create_model``.

    Raises:
        ValueError: If ``backbone`` is not a supported identifier.
    """
    key = backbone.strip().lower()
    if key not in BACKBONE_TO_MODEL_TYPE:
        raise ValueError(
            f"Unknown backbone {backbone!r}. Supported: {sorted(SUPPORTED_BACKBONES)}"
        )
    return BACKBONE_TO_MODEL_TYPE[key]


def use_batch_norm_backbone(backbone: str) -> bool:
    """ResNet variants in this repo use BatchNorm; the user-ID CNN uses layer norm."""
    return resolve_model_type(backbone) != "user_identification_cnn"
