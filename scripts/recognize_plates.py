#!/usr/bin/env python3
"""
Recognize License Plate Text from Cropped Images

Runs OCR on cropped plate images and outputs recognized text with confidence scores.

Usage:
    # Single image
    python scripts/recognize_plates.py --source outputs/plate_crops/crops/plate.jpg
    
    # Directory of crops
    python scripts/recognize_plates.py --source outputs/plate_crops/crops/
    
    # Use different OCR engine
    python scripts/recognize_plates.py --source crops/ --engine paddleocr
    
    # Save results to CSV
    python scripts/recognize_plates.py --source crops/ --output results.csv
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ocr.plate_recognizer import recognize_batch, recognize_plate_text


def main():
    parser = argparse.ArgumentParser(description="Recognize text from license plate images")
    parser.add_argument("--source", type=Path, required=True, help="Image file or directory of images")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV file (default: print to stdout)")
    parser.add_argument("--engine", default="easyocr", choices=["easyocr", "paddleocr", "tesseract"], help="OCR engine to use")
    parser.add_argument("--min-conf", type=float, default=0.0, help="Minimum confidence threshold (0-1)")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    args = parser.parse_args()

    source = args.source
    if not source.exists():
        print(f"Error: Source not found: {source}")
        sys.exit(1)

    # Collect image paths
    image_paths = []
    if source.is_file():
        if source.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
            image_paths = [source]
        else:
            print(f"Error: Not an image file: {source}")
            sys.exit(1)
    elif source.is_dir():
        image_paths = sorted([
            p for p in source.iterdir()
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
        ])
        if not image_paths:
            print(f"Error: No images found in {source}")
            sys.exit(1)
    else:
        print(f"Error: Invalid source: {source}")
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s)")
    print(f"Using OCR engine: {args.engine}")
    print()

    # Run OCR
    results = recognize_batch(image_paths, engine=args.engine, verbose=args.verbose)

    # Filter by confidence
    filtered_results = [
        (img_path, res) for img_path, res in zip(image_paths, results)
        if res.confidence >= args.min_conf
    ]

    # Output results
    if args.output:
        # Save to CSV
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "recognized_text", "confidence", "raw_text", "engine"])
            for img_path, res in filtered_results:
                writer.writerow([
                    str(img_path),
                    res.text,
                    f"{res.confidence:.4f}",
                    res.raw_text or "",
                    res.engine,
                ])
        print(f"\nResults saved to: {args.output}")
        print(f"Total: {len(filtered_results)}/{len(results)} (after confidence filter)")
    else:
        # Print to stdout
        print("=" * 80)
        print(f"{'Image':<40} {'Text':<20} {'Confidence':<12} {'Raw':<20}")
        print("=" * 80)
        for img_path, res in filtered_results:
            img_name = img_path.name[:38]
            text = res.text[:18] if res.text else "(empty)"
            conf_str = f"{res.confidence:.4f}"
            raw = (res.raw_text or "")[:18]
            print(f"{img_name:<40} {text:<20} {conf_str:<12} {raw:<20}")
        print("=" * 80)
        print(f"\nTotal: {len(filtered_results)}/{len(results)} recognized (confidence >= {args.min_conf})")

    # Summary stats
    if filtered_results:
        confidences = [res.confidence for _, res in filtered_results]
        avg_conf = sum(confidences) / len(confidences)
        print(f"Average confidence: {avg_conf:.4f}")
        print(f"Min confidence: {min(confidences):.4f}")
        print(f"Max confidence: {max(confidences):.4f}")


if __name__ == "__main__":
    main()
