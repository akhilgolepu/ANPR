from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_train_kwargs(
    cfg: dict[str, Any],
    root: Path,
    cli_device: str | None = None,
    cli_epochs: int | None = None,
    cli_batch: int | None = None,
) -> tuple[str, dict[str, Any]]:
    model_name = str(cfg.get("model", "yolov8s.pt"))

    kwargs: dict[str, Any] = {
        "data": str((root / cfg.get("data")).resolve()) if cfg.get("data") else None,
        "epochs": cli_epochs if cli_epochs is not None else cfg.get("epochs", 100),
        "batch": cli_batch if cli_batch is not None else cfg.get("batch", -1),
        "imgsz": cfg.get("imgsz", 640),
        "patience": cfg.get("patience", 25),
        "device": cli_device if cli_device is not None else cfg.get("device", "0"),
        "workers": cfg.get("workers", 8),
        "cache": cfg.get("cache", True),
        "seed": cfg.get("seed", 42),
        "deterministic": cfg.get("deterministic", True),
        "amp": cfg.get("amp", True),
        "project": str((root / cfg.get("project", "runs")).resolve()),
        "name": cfg.get("name", "exp"),
        "hsv_h": cfg.get("hsv_h", 0.015),
        "hsv_s": cfg.get("hsv_s", 0.5),
        "hsv_v": cfg.get("hsv_v", 0.4),
        "degrees": cfg.get("degrees", 5),
        "translate": cfg.get("translate", 0.1),
        "scale": cfg.get("scale", 0.5),
        "shear": cfg.get("shear", 2),
        "perspective": cfg.get("perspective", 0.0005),
        "fliplr": cfg.get("fliplr", 0.5),
        "mosaic": cfg.get("mosaic", 1.0),
        "mixup": cfg.get("mixup", 0.1),
    }

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return model_name, kwargs
