"""CNN and ResNet (18/34/50) user identification via the shared CNN Experiment + ArcFace path."""

from .model_registry import (
    BACKBONE_TO_MODEL_TYPE,
    SUPPORTED_BACKBONES,
    resolve_model_type,
    use_batch_norm_backbone,
)

__all__ = [
    "BACKBONE_TO_MODEL_TYPE",
    "SUPPORTED_BACKBONES",
    "resolve_model_type",
    "use_batch_norm_backbone",
]

# CLI entrypoints (run as scripts; not re-exported):
# - train_user_identification_backbone.py
# - eval_active_passive_cnn_resnet.py
