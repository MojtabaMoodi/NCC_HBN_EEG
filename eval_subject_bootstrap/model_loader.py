"""
Model loading utilities for CNN, LaBraM, and ResNet models.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def _cnn_modules():
    cnn_dir = Path(__file__).parent.parent / "CNN"
    cnn_dir_str = str(cnn_dir.resolve())
    if cnn_dir_str not in sys.path:
        sys.path.insert(0, cnn_dir_str)
    model_factory_module = importlib.import_module("models.model_factory")
    trainer_module = importlib.import_module("trainer")
    return model_factory_module.ModelFactory, trainer_module.load_checkpoint


def load_cnn_model(
    model_type: str,
    checkpoint_path: str,
    num_classes: int,
    device: str,
    **model_kwargs,
) -> nn.Module:
    if num_classes is None:
        raise ValueError("num_classes is required for CNN models")
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"CNN checkpoint not found: {checkpoint_path}")

    ModelFactory, load_checkpoint = _cnn_modules()
    model = ModelFactory.create_model(
        model_type,
        num_classes=num_classes,
        num_channels=60,
        **model_kwargs,
    )

    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    load_checkpoint(checkpoint_path, model, device_obj, strict=False)
    model.eval()
    return model


def load_resnet_model(
    model_type: str,
    checkpoint_path: str,
    num_classes: int,
    device: str,
    **model_kwargs,
) -> nn.Module:
    if num_classes is None:
        raise ValueError("num_classes is required for ResNet models")
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"ResNet checkpoint not found: {checkpoint_path}")

    ModelFactory, load_checkpoint = _cnn_modules()
    model = ModelFactory.create_model(
        model_type,
        num_classes=num_classes,
        num_channels=60,
        **model_kwargs,
    )

    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    load_checkpoint(checkpoint_path, model, device_obj, strict=False)
    model.eval()
    return model


def get_labram_input_chans() -> list:
    standard_channel_names = [
        "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "F1", "F2", "F5", "F6", "F9", "F10",
        "AF3", "AF4", "AF7", "AF8", "AFZ", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6",
        "FT7", "FT8", "T7", "T8", "T9", "T10", "P7", "P3", "PZ", "P4", "P8", "P1", "P2",
        "P5", "P6", "PO3", "PO4", "PO7", "PO8", "POZ", "OZ", "O1", "O2", "C3", "C4",
        "C1", "C2", "C5", "C6", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPZ",
    ]

    labram_dir = Path(__file__).parent.parent / "LaBraM"
    labram_dir_str = str(labram_dir.resolve())
    if labram_dir_str not in sys.path:
        sys.path.insert(0, labram_dir_str)

    import utils as labram_utils

    return labram_utils.get_input_chans(standard_channel_names)


def load_labram_model(
    model_name: str,
    checkpoint_path: str,
    num_classes: int,
    device: str,
    ctor_kwargs: Dict[str, Any],
) -> nn.Module:
    if num_classes is None:
        raise ValueError("num_classes is required for LaBraM models")
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"LaBraM checkpoint not found: {checkpoint_path}")

    required_ctor_keys = {
        "drop_rate",
        "drop_path_rate",
        "attn_drop_rate",
        "use_mean_pooling",
        "init_scale",
        "use_rel_pos_bias",
        "use_abs_pos_emb",
        "init_values",
        "qkv_bias",
        "multi_output",
    }
    missing = required_ctor_keys - set(ctor_kwargs.keys())
    if missing:
        raise ValueError(
            f"LaBraM ctor_kwargs missing required keys: {sorted(missing)}"
        )

    labram_dir = Path(__file__).parent.parent / "LaBraM"
    labram_dir_str = str(labram_dir.resolve())
    if labram_dir_str not in sys.path:
        sys.path.insert(0, labram_dir_str)

    from timm.models import create_model

    importlib.import_module("modeling_finetune")

    model = create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=ctor_kwargs["drop_rate"],
        drop_path_rate=ctor_kwargs["drop_path_rate"],
        attn_drop_rate=ctor_kwargs["attn_drop_rate"],
        use_mean_pooling=ctor_kwargs["use_mean_pooling"],
        init_scale=ctor_kwargs["init_scale"],
        use_rel_pos_bias=ctor_kwargs["use_rel_pos_bias"],
        use_abs_pos_emb=ctor_kwargs["use_abs_pos_emb"],
        init_values=ctor_kwargs["init_values"],
        qkv_bias=ctor_kwargs["qkv_bias"],
        multi_output=ctor_kwargs["multi_output"],
        EEG_size=ctor_kwargs.get("EEG_size"),
    )

    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        new_state_dict[key] = value
    state_dict = {
        key: value
        for key, value in new_state_dict.items()
        if "relative_position_index" not in key
    }

    import utils as labram_utils

    labram_utils.load_state_dict(
        model,
        state_dict,
        prefix="",
        ignore_missing="relative_position_index",
    )
    model.to(device_obj)
    model.eval()
    return model


def load_model_from_config(model_config: Dict[str, Any], device: str) -> nn.Module:
    """
    Load model from configuration dictionary.

    Required keys: name, ckpt_path, model_type, num_classes, prediction_type
    LaBraM additionally requires ctor_kwargs with all architecture flags.
    """
    for key in ("name", "ckpt_path", "model_type", "num_classes", "prediction_type"):
        if key not in model_config:
            raise ValueError(f"model_config missing required key: {key!r}")

    name = model_config["name"].lower()
    ckpt_path = model_config["ckpt_path"]
    num_classes = model_config["num_classes"]
    model_type = model_config["model_type"]
    ctor_kwargs = model_config.get("ctor_kwargs", {})

    if name == "cnn":
        return load_cnn_model(model_type, ckpt_path, num_classes, device, **ctor_kwargs)

    if name == "resnet":
        return load_resnet_model(model_type, ckpt_path, num_classes, device, **ctor_kwargs)

    if name == "labram":
        if not ctor_kwargs:
            raise ValueError("LaBraM model_config requires non-empty ctor_kwargs")
        return load_labram_model(
            model_type,
            ckpt_path,
            num_classes,
            device,
            ctor_kwargs=ctor_kwargs,
        )

    raise ValueError(f"Unknown model name: {name!r}. Expected 'cnn', 'resnet', or 'labram'")


def get_model_output(
    model: nn.Module,
    x: torch.Tensor,
    output_key: Optional[str],
    model_name: Optional[str],
    input_chans: Optional[list],
    prediction_type: str,
    num_classes: int,
) -> torch.Tensor:
    """Forward pass returning logits (classification) or normalized regression output."""
    if prediction_type not in ("classification", "regression"):
        raise ValueError(f"prediction_type must be 'classification' or 'regression', got {prediction_type!r}")

    with torch.no_grad():
        is_labram = bool(model_name and model_name.lower() == "labram")
        if not is_labram:
            model_type_str = str(type(model)).lower()
            if "neuraltransformer" in model_type_str or "labram" in model_type_str:
                is_labram = True

        if is_labram:
            if x.dim() != 3:
                raise ValueError(f"LaBraM expects 3D input [B, C, T], got shape {tuple(x.shape)}")
            batch_size, num_channels, seq_length = x.shape
            patch_size = 200
            if seq_length % patch_size != 0:
                raise ValueError(
                    f"Sequence length {seq_length} is not divisible by LaBraM patch_size {patch_size}"
                )
            num_patches = seq_length // patch_size
            x = x.view(batch_size, num_channels, num_patches, patch_size)
            x = x / 100.0
            if input_chans is None:
                raise ValueError("input_chans is required for LaBraM inference")
            output = model(x, input_chans=input_chans)
        else:
            output = model(x)

    if isinstance(output, dict):
        if output_key is not None:
            if output_key not in output:
                raise KeyError(
                    f"output_key {output_key!r} not in model output keys: {list(output.keys())}"
                )
            output = output[output_key]
        else:
            for key in ("logits", "output", "pred", "prediction", "age"):
                if key in output:
                    output = output[key]
                    break
            else:
                raise ValueError(
                    f"Model returned dict without recognized output key. Keys: {list(output.keys())}"
                )

    if prediction_type == "regression":
        if output.ndim == 1:
            output = output.unsqueeze(-1)
        if output.shape[-1] != 1:
            raise ValueError(
                f"Regression model must output 1 value per sample, got shape {tuple(output.shape)}"
            )
        return output.squeeze(-1)

    # classification
    if num_classes == 1:
        if output.shape[-1] != 1:
            raise ValueError(
                f"Binary LaBraM (num_classes=1) expects output shape [B, 1], got {tuple(output.shape)}"
            )
        return output

    if output.shape[-1] != num_classes:
        raise ValueError(
            f"Classification model output dim {output.shape[-1]} != num_classes {num_classes}"
        )
    return output
