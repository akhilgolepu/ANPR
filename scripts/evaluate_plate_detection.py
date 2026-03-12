#!/usr/bin/env python3
"""
Evaluate License Plate Detection Model

Runs validation on test/val set and reports accuracy metrics (mAP, Precision, Recall).
Does NOT save images - pure evaluation.

Usage:
    python scripts/evaluate_plate_detection.py
    python scripts/evaluate_plate_detection.py --weights path/to/best.pt --data datasets/processed/plate_detection/data.yaml
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detection.plate_cropper import find_latest_best_pt


def main():
    parser = argparse.ArgumentParser(description="Evaluate plate detection model accuracy")
    parser.add_argument("--weights", type=Path, default=None, help="Path to trained weights (.pt). Defaults to latest best.pt")
    parser.add_argument("--data", type=Path, default=None, help="Path to data.yaml. Defaults to datasets/processed/plate_detection/data.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default="0", help="Device: 0, cpu, etc.")
    parser.add_argument("--save-json", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    from ultralytics import YOLO

    # Find weights
    weights = args.weights
    if weights is None:
        latest = find_latest_best_pt(ROOT / "runs")
        if latest is None:
            raise FileNotFoundError(
                "Could not find trained weights. Expected a `best.pt` under "
                "`runs/plate_detection/`. Provide --weights explicitly."
            )
        weights = latest
        print(f"Using weights: {weights}")
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    # Find data.yaml
    data_path = args.data or (ROOT / "datasets/processed/plate_detection/data.yaml")
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_path}. Run dataset preparation first: python scripts/prepare_dataset.py")

    # Load model
    model = YOLO(str(weights))

    print(f"\nEvaluating on {args.split} split...")
    print(f"  Data: {data_path}")
    print(f"  Weights: {weights.name}")
    print(f"  Conf threshold: {args.conf}")
    print(f"  IoU threshold: {args.iou}")
    print()

    # Run validation
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save_json=args.save_json,
        save=False,  # Don't save images
        plots=False,  # Don't generate plots
        verbose=True,
    )

    # Print key metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    if hasattr(metrics, "box"):
        box = metrics.box
        import numpy as np
        
        # Handle numpy arrays - use item() for 0-d arrays, [0] for 1-d
        def to_float(val):
            if isinstance(val, np.ndarray):
                if val.ndim == 0:
                    return float(val.item())
                elif val.ndim == 1 and len(val) > 0:
                    return float(val[0])
                else:
                    return float(val)
            return float(val)
        
        p = to_float(box.p)
        r = to_float(box.r)
        map50 = to_float(box.map50)
        map_val = to_float(box.map)
        
        print(f"\nPrecision (P):     {p:.4f}")
        print(f"Recall (R):        {r:.4f}")
        print(f"mAP@0.5:           {map50:.4f}")
        print(f"mAP@0.5:0.95:       {map_val:.4f}")
        
        # Per-class if available
        if hasattr(box, "maps") and len(box.maps) > 0:
            maps_val = to_float(box.maps[0])
            print(f"\nPer-class mAP@0.5: {maps_val:.4f} (license_plate)")
        
        # Speed metrics (already printed by YOLO, but extract if available)
        if hasattr(metrics, "speed"):
            speed = metrics.speed
            if isinstance(speed, dict):
                preprocess = speed.get("preprocess", 0) * 1000
                inference = speed.get("inference", 0) * 1000
                postprocess = speed.get("postprocess", 0) * 1000
                total = preprocess + inference + postprocess
                print(f"\nSpeed: {total:.1f}ms per image ({preprocess:.1f}ms preprocess, {inference:.1f}ms inference, {postprocess:.1f}ms postprocess)")
    else:
        print("\nMetrics object structure:")
        print(f"  {type(metrics)}")
        print(f"  Attributes: {[a for a in dir(metrics) if not a.startswith('_')]}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
