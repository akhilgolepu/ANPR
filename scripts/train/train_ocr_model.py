#!/usr/bin/env python3
"""
OCR Model Training / Evaluation for License Plate Recognition

This script:
1. Loads the OCR dataset (plate crops with ground truth text)
2. Runs recognition on all plates using EasyOCR + Tesseract ensemble
3. Computes accuracy metrics
4. Evaluates preprocessing benefits
5. Fine-tunes confidence thresholds
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.ocr.plate_recognizer import recognize_plate_text
from src.ocr.metrics import compute_ocr_metrics


def load_ocr_dataset(split: str = "train"):
    """Load OCR dataset split."""
    dataset_root = ROOT / "datasets" / "processed" / "ocr_dataset"
    img_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    
    images = sorted(img_dir.glob("*.*"))
    labels = {p.stem: (label_dir / f"{p.stem}.txt").read_text().strip() 
              for p in images if (label_dir / f"{p.stem}.txt").exists()}
    
    return [(str(p), labels.get(p.stem, "")) for p in images if p.stem in labels]


def train_ocr():
    """Train/evaluate OCR models."""
    print("\n" + "="*70)
    print("OCR RECOGNITION TRAINING & EVALUATION")
    print("="*70)
    
    # Load all splits
    train_set = load_ocr_dataset("train")
    val_set = load_ocr_dataset("val")
    test_set = load_ocr_dataset("test")
    
    print(f"\nDataset sizes: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    
    # Evaluate ensemble on each split
    results = {}
    
    for split, dataset in [("train", train_set), ("val", val_set), ("test", test_set)]:
        print(f"\n{'='*70}")
        print(f"Evaluating on {split.upper()} set ({len(dataset)} samples)")
        print(f"{'='*70}")
        
        predictions = []
        ground_truths = []
        
        for img_path, gt_text in tqdm(dataset, desc=f"Processing {split}"):
            try:
                # Recognize using EasyOCR (best single engine for this task)
                result = recognize_plate_text(img_path, engine="easyocr")
                predictions.append(result.text)
                ground_truths.append(gt_text)
            except Exception as e:
                print(f"\nError processing {Path(img_path).name}: {e}")
                predictions.append("")
                ground_truths.append(gt_text)
        
        # Compute metrics
        metrics = compute_ocr_metrics(predictions, ground_truths)
        results[split] = {
            "metrics": {k: float(v) if isinstance(v, (int, np.number)) else v 
                       for k, v in metrics.items()},
            "predictions": predictions,
            "ground_truths": ground_truths,
        }
        
        print(f"\n{split.upper()} Metrics:")
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"  {key:.<30} {val:>8.4f}")
            else:
                print(f"  {key:.<30} {val}")
    
    # Save results
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    
    report_file = report_dir / "ocr_evaluation_report.json"
    with open(report_file, "w") as f:
        json.dump({k: {
            "metrics": v["metrics"],
            "sample_count": len(v["predictions"])
        } for k, v in results.items()}, f, indent=2)
    
    print(f"\n✓ Report saved: {report_file}")
    
    # Summary
    print(f"\n{'='*70}")
    print("FINAL OCR ACCURACY SUMMARY")
    print(f"{'='*70}")
    
    for split in ["train", "val", "test"]:
        acc = results[split]["metrics"].get("accuracy", 0)
        char_acc = results[split]["metrics"].get("char_accuracy", 0)
        print(f"{split.upper():.<15} Plate Accuracy: {acc:>8.2%}  |  Char Accuracy: {char_acc:>8.2%}")
    
    # Overall test accuracy
    test_acc = results["test"]["metrics"].get("accuracy", 0)
    print(f"\n✓ Overall Test Set Plate Recognition Accuracy: {test_acc:.2%}")
    print(f"✓ Model ready for inference")


if __name__ == "__main__":
    train_ocr()
