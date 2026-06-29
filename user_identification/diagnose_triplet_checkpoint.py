#!/usr/bin/env python3
"""
Diagnose FaceNet triplet checkpoints: embedding separation and mining strategy comparison.

Usage (from project root)::

  python user_identification/diagnose_triplet_checkpoint.py \\
    --checkpoint final_user_identification_triplet/4s/fold_0/best_model.pth \\
    --hdf5_dir ~/scratch/4s/fold_0 \\
    --segment_length 4s \\
    --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_project_root = Path(__file__).resolve().parent.parent
_labram_dir = _project_root / "LaBraM"
_data_processing_dir = _project_root / "data_processing"
for _path in (_project_root, _labram_dir, _data_processing_dir):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from data_processing.target_transforms import create_user_identification_transform_from_hdf5
from LaBraM.labram_dataset import prepare_labram_dataset
from user_identification.embedding_criterion_utils import load_criterion_from_checkpoint
from user_identification.labram_model import LaBraMUserIdentificationWrapper
from user_identification.triplet_diagnostics import (
    compare_triplet_mining_on_batch,
    diagnose_embedding_separation,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--hdf5_dir", type=str, required=True)
    p.add_argument("--segment_length", type=str, required=True, choices=["1s", "2s", "4s"])
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=30)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--compare_mining", action="store_true",
                   help="Compare semi_hard vs all vs batch_hard on one val batch")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    transform, num_classes = create_user_identification_transform_from_hdf5(args.hdf5_dir)
    train_ds, val_ds, test_ds = prepare_labram_dataset(
        dataset_type="user_identification",
        hdf5_dir=args.hdf5_dir,
        segment_length=args.segment_length,
        task_type="both",
        random_seed=42,
        user_identification_transform=transform,
        sampling_rate=200,
    )
    dataset = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        shuffle=(args.split == "train"),
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    embedding_dim = int(ckpt["embedding_dim"])
    model = LaBraMUserIdentificationWrapper(
        num_channels=60,
        num_classes=num_classes,
        dropout_rate=0.1,
        model_name="labram_base_patch200_200",
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    loss_type = ckpt.get("loss_type", "arcface")
    report: dict = {
        "checkpoint": str(ckpt_path.resolve()),
        "hdf5_dir": str(Path(args.hdf5_dir).resolve()),
        "split": args.split,
        "loss_type": loss_type,
        "checkpoint_val_accuracy_percent": ckpt.get("val_accuracy"),
        "checkpoint_test_accuracy_percent": ckpt.get("test_accuracy"),
    }

    print(f"\n{'='*60}")
    print(f"Triplet / embedding diagnostic")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  split: {args.split}")
    print(f"  loss_type: {loss_type}")
    print(f"{'='*60}\n")

    sep = diagnose_embedding_separation(
        model, loader, device,
        max_batches=args.max_batches,
        header=f"{args.split} separation",
    )
    report["embedding_separation"] = sep

    if loss_type == "triplet":
        criterion, _ = load_criterion_from_checkpoint(
            ckpt, num_classes=num_classes, embedding_dim=embedding_dim, device=device
        )
        report["triplet_margin"] = ckpt.get("triplet_margin")
        report["triplet_mining"] = ckpt.get("triplet_mining", "unknown")

    if args.compare_mining:
        from user_identification.triplet_diagnostics import collect_embeddings_from_loader

        emb, labels = collect_embeddings_from_loader(
            model, loader, device, max_batches=1
        )
        emb = emb.to(device)
        labels = labels.to(device)
        margin = float(ckpt.get("triplet_margin", 0.2))
        mining_cmp = compare_triplet_mining_on_batch(emb, labels, margin=margin)
        report["mining_comparison_one_batch"] = mining_cmp
        print("\nMining comparison (one batch):")
        for name, stats in mining_cmp.items():
            active = stats.get("active_triplet_fraction", float("nan"))
            loss = stats.get("loss", float("nan"))
            print(f"  {name:12s}  loss={loss:.4f}  active_fraction={active:.4f}")

    out_path = Path(args.output_json) if args.output_json else ckpt_path.parent / "triplet_diagnostic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDiagnostic JSON: {out_path}")


if __name__ == "__main__":
    main()
