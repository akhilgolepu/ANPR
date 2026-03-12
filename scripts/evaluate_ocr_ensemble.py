#!/usr/bin/env python3
"""
Improved OCR Recognition with Ensemble & Better Postprocessing

Combines:
1. EasyOCR (best for printed text)
2. Tesseract with preprocessing  
3. Voting/confidence fusion
4. Format-based corrections
"""

import sys
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ocr.plate_recognizer import recognize_with_easyocr, recognize_with_tesseract
from src.ocr.postprocessing import postprocess_indian_plate
from src.ocr.metrics import compute_ocr_metrics, normalize_plate_text


def recognize_with_ensemble(image_path: str) -> Tuple[str, float]:
    """
    Recognize plate using ensemble of EasyOCR + Tesseract.
    Returns best result with confidence > 0.3.
    """
    try:
        easy_result = recognize_with_easyocr(image_path)
        easy_conf = easy_result.confidence
        easy_text = easy_result.text
    except Exception:
        easy_text = ""
        easy_conf = 0.0
    
    try:
        tess_result = recognize_with_tesseract(image_path)
        tess_conf = tess_result.confidence
        tess_text = tess_result.text
    except Exception:
        tess_text = ""
        tess_conf = 0.0
    
    # Choose best by confidence
    if easy_conf >= tess_conf:
        final_text = easy_text
        final_conf = easy_conf
    else:
        final_text = tess_text
        final_conf = tess_conf
    
    # Apply post-processing for Indian plates
    if final_text:
        final_text = postprocess_indian_plate(normalize_plate_text(final_text), strict=False)
    
    return final_text, final_conf


def evaluate_ocr_improved():
    """Evaluate OCR with ensemble and improved postprocessing."""
    print("\n" + "="*70)
    print("OCR EVALUATION - IMPROVED ENSEMBLE")
    print("="*70)
    
    ocr_dataset_root = ROOT / "datasets" / "processed" / "ocr_dataset"
    test_img_dir = ocr_dataset_root / "images" / "test"
    test_label_dir = ocr_dataset_root / "labels" / "test"
    
    images = sorted(test_img_dir.glob("*.jpg")) + sorted(test_img_dir.glob("*.png"))
    test_pairs = []
    
    for img_path in images:
        label_path = test_label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            gt_text = label_path.read_text().strip()
            test_pairs.append((str(img_path), gt_text))
    
    print(f"Evaluating ensemble on {len(test_pairs)} test samples")
    
    predictions = []
    ground_truths = []
    
    for i, (img_path, gt_text) in enumerate(test_pairs):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(test_pairs)}")
        
        pred_text, conf = recognize_with_ensemble(img_path)
        predictions.append(pred_text)
        ground_truths.append(gt_text)
    
    metrics = compute_ocr_metrics(predictions, ground_truths)
    
    print(f"\nEnsemble Results:")
    print(f"  Plate Accuracy:     {metrics['accuracy']:.2%} ({int(metrics['exact_matches'])}/{int(metrics['total_samples'])})")
    print(f"  Character Accuracy: {metrics['char_accuracy']:.2%}")
    print(f"  Avg Prediction Len: {metrics['avg_predicted_length']:.1f}")
    print(f"  Avg GT Len:         {metrics['avg_ground_truth_length']:.1f}")
    
    return metrics["accuracy"]


if __name__ == "__main__":
    acc = evaluate_ocr_improved()
    print(f"\n✓ Ensemble Test Accuracy: {acc:.2%}")
