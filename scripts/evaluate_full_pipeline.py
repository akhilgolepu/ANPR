#!/usr/bin/env python3
"""
Full ANPR Pipeline Evaluation

Evaluates:
1. Plate Detection (YOLOv8)
2. Plate Cropping accuracy
3. OCR Recognition accuracy
4. End-to-end pipeline metrics
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detection.plate_cropper import extract_plate_crops
from src.ocr.plate_recognizer import recognize_plate_text
from src.ocr.metrics import compute_ocr_metrics, normalize_plate_text


def evaluate_plate_detection_on_val():
    """
    Evaluate YOLOv8 plate detector on validation set.
    Uses YOLO's built-in validation metrics.
    """
    print("\n" + "="*70)
    print("PLATE DETECTION EVALUATION (YOLOv8)")
    print("="*70)
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("⚠️  YOLOv8 not available")
        return {}
    
    # Load best model
    model_path = ROOT / "runs" / "plate_detection" / "yolov8s_6402" / "weights" / "best.pt"
    if not model_path.exists():
        print(f"⚠️  Model not found: {model_path}")
        return {}
    
    model = YOLO(str(model_path))
    data_yaml = ROOT / "datasets" / "processed" / "plate_detection" / "data.yaml"
    
    # Run validation
    print(f"Validating on: {data_yaml}")
    results = model.val(data=str(data_yaml), device=0)
    
    metrics_dict = {
        "map@0.5": float(results.results_dict.get("metrics/mAP50(B)", 0)),
        "map@0.5:0.95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
        "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
    }
    
    print(f"\nValidation Metrics:")
    for key, val in metrics_dict.items():
        print(f"  {key:.<30} {val:>8.4f}")
    
    return metrics_dict


def evaluate_ocr_pipeline():
    """
    Evaluate OCR on cropped plates from test set.
    Realistic evaluation: detects plates, crops them, recognizes text.
    """
    print("\n" + "="*70)
    print("OCR RECOGNITION EVALUATION")
    print("="*70)
    
    ocr_dataset_root = ROOT / "datasets" / "processed" / "ocr_dataset"
    test_img_dir = ocr_dataset_root / "images" / "test"
    test_label_dir = ocr_dataset_root / "labels" / "test"
    
    # Load test set
    images = sorted(test_img_dir.glob("*.jpg")) + sorted(test_img_dir.glob("*.png"))
    test_pairs = []
    
    for img_path in images:
        label_path = test_label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            gt_text = label_path.read_text().strip()
            test_pairs.append((str(img_path), gt_text))
    
    print(f"Evaluating on {len(test_pairs)} test plate crops")
    
    predictions = []
    ground_truths = []
    confidences = []
    
    for img_path, gt_text in tqdm(test_pairs, desc="Recognizing"):
        try:
            result = recognize_plate_text(img_path, engine="easyocr")
            predictions.append(result.text)
            ground_truths.append(gt_text)
            confidences.append(result.confidence)
        except Exception as e:
            print(f"Error on {Path(img_path).name}: {e}")
            predictions.append("")
            ground_truths.append(gt_text)
            confidences.append(0.0)
    
    # Compute metrics
    metrics = compute_ocr_metrics(predictions, ground_truths)
    
    print(f"\nOCR Metrics:")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:.<30} {val:>8.4f}" if val < 1 else f"  {key:.<30} {int(val):>8}")
        else:
            print(f"  {key:.<30} {val:>8}")
    
    # Confidence stats
    print(f"\nConfidence Statistics:")
    print(f"  Mean Confidence          {np.mean(confidences):>8.4f}")
    print(f"  Median Confidence        {np.median(confidences):>8.4f}")
    print(f"  Min Confidence           {np.min(confidences):>8.4f}")
    print(f"  Max Confidence           {np.max(confidences):>8.4f}")
    
    return {**metrics, "mean_confidence": float(np.mean(confidences))}


def evaluate_end_to_end():
    """
    Full pipeline: image → detect plates → recognize text.
    Evaluate on actual vehicle images (if available).
    """
    print("\n" + "="*70)
    print("END-TO-END PIPELINE EVALUATION")
    print("="*70)
    
    # For demo, use test images from plate detection dataset
    dataset_root = ROOT / "datasets" / "processed" / "plate_detection" / "images" / "test"
    
    if not dataset_root.exists():
        print(f"⚠️  Test images not found at {dataset_root}")
        return {}
    
    test_images = list(dataset_root.glob("*.jpg")) + list(dataset_root.glob("*.png"))
    if not test_images:
        print(f"⚠️  No test images found")
        return {}
    
    print(f"Testing on {len(test_images[:50])} vehicle images (sample)")  # Limit to 50 for speed
    
    try:
        from ultralytics import YOLO
        weights_path = ROOT / "runs" / "plate_detection" / "yolov8s_6402" / "weights" / "best.pt"
        if not weights_path.exists():
            print(f"⚠️  Model weights not found: {weights_path}")
            return {}
        
        model = YOLO(str(weights_path))
        
        total_detections = 0
        successful_recognitions = 0
        total_time = 0
        
        for img_path in test_images[:50]:
            try:
                t0 = time.time()
                
                # Detect plates
                results = model.predict(source=str(img_path), conf=0.25, verbose=False)
                detections_count = len(results[0].boxes)
                total_detections += detections_count
                
                # For each detection, try to recognize
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        if x2 > x1 and y2 > y1:
                            crop = img[y1:y2, x1:x2]
                            if crop.size > 0:
                                try:
                                    result = recognize_plate_text(crop, engine="easyocr")
                                    if result.text and result.confidence > 0.3:
                                        successful_recognitions += 1
                                except:
                                    pass
                
                total_time += time.time() - t0
                
            except Exception as e:
                print(f"Error on {img_path.name}: {e}")
        
        avg_time = total_time / len(test_images[:50]) if test_images else 0
        
        metrics = {
            "test_images": len(test_images[:50]),
            "total_plates_detected": float(total_detections),
            "successful_recognitions": float(successful_recognitions),
            "avg_time_per_image": float(avg_time),
        }
        
        print(f"\nEnd-to-End Results:")
        print(f"  Test Images:              {metrics['test_images']:>8.0f}")
        print(f"  Plates Detected:          {metrics['total_plates_detected']:>8.0f}")
        print(f"  Successful Recognitions:  {metrics['successful_recognitions']:>8.0f}")
        print(f"  Avg Time per Image:       {metrics['avg_time_per_image']:>8.3f}s")
        
        return metrics
        
    except Exception as e:
        print(f"⚠️  Error: {e}")
        return {}


def main():
    """Run all evaluations."""
    print("\n" + "#"*70)
    print("# AUTOMATIC NUMBER PLATE RECOGNITION (ANPR) SYSTEM")
    print("# Full Pipeline Evaluation")
    print("#"*70)
    
    report = {}
    
    # Plate detection
    detection_metrics = evaluate_plate_detection_on_val()
    report["plate_detection"] = detection_metrics
    
    # OCR
    ocr_metrics = evaluate_ocr_pipeline()
    report["ocr_recognition"] = ocr_metrics
    
    # End-to-end
    e2e_metrics = evaluate_end_to_end()
    report["end_to_end"] = e2e_metrics
    
    # Save report
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    
    report_file = report_dir / "evaluation_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    # Summary
    print(f"\n{'='*70}")
    print("FINAL SYSTEM ACCURACY")
    print(f"{'='*70}")
    
    if detection_metrics:
        print(f"\nPlate Detection mAP@0.5:     {detection_metrics.get('map@0.5', 0):.4f}")
    
    if ocr_metrics:
        print(f"OCR Plate Recognition:      {ocr_metrics.get('accuracy', 0):.2%}")
        print(f"OCR Character Accuracy:     {ocr_metrics.get('char_accuracy', 0):.2%}")
    
    if e2e_metrics and e2e_metrics.get("total_plates_detected", 0) > 0:
        success_rate = e2e_metrics.get("successful_recognitions", 0) / e2e_metrics.get("total_plates_detected", 1)
        print(f"E2E Success Rate:           {success_rate:.2%}")
    
    print(f"\n✓ Report saved: {report_file}\n")


if __name__ == "__main__":
    main()
