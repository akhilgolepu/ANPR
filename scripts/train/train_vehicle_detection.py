#!/usr/bin/env python3
"""
Train vehicle detection model (config-driven).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ultralytics import YOLO  # noqa: E402

from _train_common import build_train_kwargs, load_yaml  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train vehicle detector")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "training" / "vehicle_detection_train.yaml",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = load_yaml(args.config)
    model_name, train_kwargs = build_train_kwargs(
        cfg,
        ROOT,
        cli_device=args.device,
        cli_epochs=args.epochs,
        cli_batch=args.batch,
    )

    data_path = Path(train_kwargs["data"])
    if not data_path.exists():
        raise FileNotFoundError(
            f"Vehicle data file missing: {data_path}. Run: python scripts/prepare_dataset.py"
        )

    print("Training vehicle detector")
    print(f"Model: {model_name}")
    print(f"Data: {data_path}")
    print(f"Batch: {train_kwargs.get('batch')} (AutoBatch if -1)")
    print(f"Project/Name: {train_kwargs.get('project')} / {train_kwargs.get('name')}")

    model = YOLO(model_name)
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
