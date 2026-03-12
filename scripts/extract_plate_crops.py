#!/usr/bin/env python3
"""
Step: Detect → Extract/Crop license plates

Runs your trained YOLOv8 plate detector on an image / folder / video and saves plate crops.

Usage:
  python scripts/extract_plate_crops.py --source path/to/image_or_dir_or_video
  python scripts/extract_plate_crops.py --source datasets/processed/plate_detection/images/val --out crops_out

Defaults:
    - weights: latest `best.pt` under `anpr_v2/runs/` or `runs/`
  - outputs: `<out>/crops/*.jpg` and `<out>/crops.csv`
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detection.plate_cropper import extract_plate_crops, find_latest_best_pt


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect plates and save cropped images")
    parser.add_argument("--source", type=Path, required=True, help="Image, directory, or video path")
    parser.add_argument("--out", type=Path, default=Path("outputs/plate_crops"), help="Output directory")
    parser.add_argument("--weights", type=Path, default=None, help="Path to YOLO weights (.pt). Defaults to latest best.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default="0", help="Device: 0, cpu, etc.")
    parser.add_argument("--max-det", type=int, default=50, help="Max detections per image/frame")
    args = parser.parse_args()

    source = args.source
    if not source.exists():
        raise FileNotFoundError(f"--source not found: {source}")

    weights = args.weights
    if weights is None:
        latest = find_latest_best_pt(ROOT / "runs")
        if latest is None:
            raise FileNotFoundError(
                "Could not find trained weights. Expected a `best.pt` under "
                "`runs/plate_detection/`. Provide --weights explicitly."
            )
        weights = latest
    if not weights.exists():
        raise FileNotFoundError(f"--weights not found: {weights}")

    out_dir = args.out
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    records = extract_plate_crops(
        weights=weights,
        source=source,
        out_dir=out_dir,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=str(args.device),
        max_det=args.max_det,
    )

    print(f"Saved {len(records)} plate crops to {out_dir}")
    print(f"CSV: {out_dir / 'crops.csv'}")


if __name__ == "__main__":
    main()

