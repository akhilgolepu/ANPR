#!/usr/bin/env python3
"""
Prepare ANPR datasets from raw sources with clean train/val/test splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset.prepare_dataset import (  # noqa: E402
    load_config,
    prepare_ocr_dataset,
    prepare_plate_dataset,
    prepare_vehicle_dataset,
)


def _validate_splits(splits: dict) -> None:
    train = float(splits.get("train", 0.8))
    val = float(splits.get("val", 0.1))
    test = float(splits.get("test", 0.1))
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Invalid split sum: train+val+test={total:.6f}, expected 1.0")


def main() -> None:
    parser = argparse.ArgumentParser(description="ANPR Dataset Preparation")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "dataset.yaml",
        help="Path to dataset config YAML",
    )
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = load_config(args.config)

    paths = cfg.get("paths", {})
    outputs = cfg.get("outputs", {})
    splits = cfg.get("splits", {})
    _validate_splits(splits)

    seed = int(splits.get("seed", 42))
    raw_root = ROOT / paths.get("raw_root", "datasets/raw")

    plate_sources = paths.get("plate_sources", [])
    plate_crop_sources = paths.get("plate_crop_sources", [])
    vehicle_sources = paths.get("vehicle_sources", [])

    vehicle_out = ROOT / outputs.get("vehicle_detection", "datasets/processed/vehicle_detection")
    plate_out = ROOT / outputs.get("plate_detection", "datasets/processed/plate_detection")
    ocr_out = ROOT / outputs.get("ocr_dataset", "datasets/processed/ocr_dataset")

    vehicle_classes = cfg.get(
        "vehicle_classes",
        {
            "car": 0,
            "bus": 1,
            "tempo": 2,
            "vehicle_truck": 3,
            "two_wheelers": 4,
            "auto": 2,
            "tractor": 3,
            "bicycle": 4,
        },
    )

    plate_crop = cfg.get("plate_crop", {})
    min_width = int(plate_crop.get("min_width", 20))
    min_height = int(plate_crop.get("min_height", 8))
    padding_ratio = float(plate_crop.get("padding_ratio", 0.1))

    print("Building dataset splits...")
    print(f"Raw root: {raw_root}")
    print(f"Splits: train={splits.get('train')} val={splits.get('val')} test={splits.get('test')} seed={seed}")

    vehicle_counts = prepare_vehicle_dataset(
        raw_root=raw_root,
        sources=vehicle_sources,
        out_root=vehicle_out,
        class_map=vehicle_classes,
        splits=splits,
        seed=seed,
    )

    plate_counts = prepare_plate_dataset(
        raw_root=raw_root,
        sources=plate_sources,
        out_root=plate_out,
        splits=splits,
        seed=seed,
        crop_sources=plate_crop_sources,
    )

    ocr_counts = prepare_ocr_dataset(
        raw_root=raw_root,
        sources=plate_sources,
        out_root=ocr_out,
        min_w=min_width,
        min_h=min_height,
        padding_ratio=padding_ratio,
        splits=splits,
        seed=seed,
    )

    summary = {
        "config": str(args.config.relative_to(ROOT)),
        "raw_root": str(raw_root.relative_to(ROOT)),
        "outputs": {
            "vehicle_detection": str(vehicle_out.relative_to(ROOT)),
            "plate_detection": str(plate_out.relative_to(ROOT)),
            "ocr_dataset": str(ocr_out.relative_to(ROOT)),
        },
        "splits": {
            "train": float(splits.get("train", 0.8)),
            "val": float(splits.get("val", 0.1)),
            "test": float(splits.get("test", 0.1)),
            "seed": seed,
        },
        "counts": {
            "vehicle_detection": vehicle_counts,
            "plate_detection": plate_counts,
            "ocr_dataset": ocr_counts,
        },
    }

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_json = reports_dir / "dataset_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print("\nDone.")
    print(f"Vehicle: {vehicle_counts}")
    print(f"Plate:   {plate_counts}")
    print(f"OCR:     {ocr_counts}")
    print(f"Summary: {out_json}")


if __name__ == "__main__":
    main()
