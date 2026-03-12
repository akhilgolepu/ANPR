#!/usr/bin/env python3
"""
Evaluate OCR accuracy on OCR dataset with ground truth labels.

Compares recognized text against ground truth labels and computes accuracy metrics.

Usage:
    python scripts/evaluate_ocr.py --split val
    python scripts/evaluate_ocr.py --split test --engine easyocr --limit 100
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ocr.plate_recognizer import recognize_plate_text
from src.ocr.metrics import compute_comprehensive_metrics


def normalize_text(text: str) -> str:
    """Normalize text for comparison (uppercase, remove spaces/special chars)."""
    return "".join(c.upper() for c in text if c.isalnum())


def compute_accuracy(predicted: str, ground_truth: str) -> tuple[float, bool]:
    """
    Compute character-level and exact match accuracy.
    
    Returns:
        (char_accuracy, exact_match)
    """
    metrics = compute_comprehensive_metrics(predicted, ground_truth)
    return metrics["character_accuracy"], metrics["exact_match"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR accuracy on dataset")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--engine", default="easyocr", choices=["easyocr", "paddleocr", "tesseract"], help="OCR engine")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to test")
    parser.add_argument("--min-conf", type=float, default=0.0, help="Minimum confidence threshold")
    args = parser.parse_args()

    ocr_dataset = ROOT / "datasets" / "processed" / "ocr_dataset"
    images_dir = ocr_dataset / "images" / args.split
    labels_dir = ocr_dataset / "labels" / args.split

    if not images_dir.exists() or not labels_dir.exists():
        print(f"Error: OCR dataset not found at {ocr_dataset}")
        print("Run dataset preparation first: python scripts/prepare_dataset.py")
        sys.exit(1)

    # Get all images
    image_files = sorted(images_dir.glob("*.jpg"))
    if args.limit:
        image_files = image_files[:args.limit]

    if not image_files:
        print(f"No images found in {images_dir}")
        sys.exit(1)

    print(f"Evaluating OCR on {len(image_files)} images ({args.split} split)")
    print(f"Engine: {args.engine}")
    print()

    results = []
    char_accuracies = []
    exact_matches = 0
    levenshtein_distances = []
    normalized_edit_distances = []
    character_error_rates = []

    for i, img_path in enumerate(image_files):
        if (i + 1) % 10 == 0:
            print(f"Processing {i+1}/{len(image_files)}...")

        # Load ground truth
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        gt_text = label_path.read_text().strip()

        # Run OCR
        try:
            ocr_result = recognize_plate_text(img_path, engine=args.engine)
            
            if ocr_result.confidence < args.min_conf:
                continue

            pred_text = ocr_result.text
            metrics = compute_comprehensive_metrics(pred_text, gt_text)

            results.append({
                "image": img_path.name,
                "ground_truth": gt_text,
                "predicted": pred_text,
                "char_accuracy": metrics["character_accuracy"],
                "exact_match": metrics["exact_match"],
                "levenshtein_distance": metrics["levenshtein_distance"],
                "normalized_edit_distance": metrics["normalized_edit_distance"],
                "character_error_rate": metrics["character_error_rate"],
                "confidence": ocr_result.confidence,
            })

            char_accuracies.append(metrics["character_accuracy"])
            levenshtein_distances.append(metrics["levenshtein_distance"])
            normalized_edit_distances.append(metrics["normalized_edit_distance"])
            character_error_rates.append(metrics["character_error_rate"])
            if metrics["exact_match"]:
                exact_matches += 1

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue

    if not results:
        print("No valid results!")
        return

    # Print summary
    print("\n" + "=" * 80)
    print("OCR EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nTotal tested: {len(results)}")
    print(f"Exact matches: {exact_matches} ({exact_matches/len(results)*100:.2f}%)")
    print(f"\nCharacter-Level Metrics:")
    print(f"  Average character accuracy: {sum(char_accuracies)/len(char_accuracies):.4f}")
    print(f"  Min character accuracy: {min(char_accuracies):.4f}")
    print(f"  Max character accuracy: {max(char_accuracies):.4f}")
    print(f"\nEdit Distance Metrics:")
    print(f"  Average Levenshtein distance: {sum(levenshtein_distances)/len(levenshtein_distances):.2f}")
    print(f"  Average normalized edit distance: {sum(normalized_edit_distances)/len(normalized_edit_distances):.4f}")
    print(f"  Average Character Error Rate (CER): {sum(character_error_rates)/len(character_error_rates):.4f}")
    print(f"\nConfidence:")
    if results:
        avg_conf = sum(r["confidence"] for r in results) / len(results)
        print(f"  Average OCR confidence: {avg_conf:.4f}")

    # Show some examples
    print("\n" + "=" * 80)
    print("Sample Results (first 10):")
    print("=" * 80)
    print(f"{'Image':<30} {'Ground Truth':<20} {'Predicted':<20} {'Char Acc':<10} {'Exact':<6}")
    print("-" * 80)
    for r in results[:10]:
        exact_str = "✓" if r["exact_match"] else "✗"
        print(f"{r['image'][:28]:<30} {r['ground_truth'][:18]:<20} {r['predicted'][:18]:<20} {r['char_accuracy']:.4f}     {exact_str:<6}")

    # Show worst examples
    worst = sorted(results, key=lambda x: x["char_accuracy"])[:5]
    if worst:
        print("\n" + "=" * 80)
        print("Worst Results:")
        print("=" * 80)
        print(f"{'Image':<30} {'Ground Truth':<20} {'Predicted':<20} {'Char Acc':<10}")
        print("-" * 80)
        for r in worst:
            print(f"{r['image'][:28]:<30} {r['ground_truth'][:18]:<20} {r['predicted'][:18]:<20} {r['char_accuracy']:.4f}")


if __name__ == "__main__":
    main()
