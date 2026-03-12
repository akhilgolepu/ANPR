"""
Test PaddleOCR vs EasyOCR on Indian License Plates
PaddleOCR is optimized for Asian text and may perform better
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from paddleocr import PaddleOCR
import sys

sys.path.insert(0, '/home/akhil/3-2')


def test_paddleocr(test_image_dir: str, test_label_dir: str):
    """Test PaddleOCR accuracy"""
    
    print("\nInitializing PaddleOCR...")
    paddler = PaddleOCR()
    print("✓ PaddleOCR loaded\n")
    
    test_path = Path(test_image_dir)
    test_images = sorted(list(test_path.glob('*.jpg')) + list(test_path.glob('*.png')))
    
    print(f"{'='*70}")
    print(f"Testing PaddleOCR on {len(test_images)} license plates")
    print(f"{'='*70}\n")
    
    predictions = []
    ground_truths = []
    correct = 0
    char_correct = 0
    char_total = 0
    
    for img_path in tqdm(test_images, desc="Testing"):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Resize to standard size
        image = cv2.resize(image, (128, 64))
        
        # Recognize with PaddleOCR
        try:
            results = paddler.ocr(image, cls=True)
            if results and results[0]:
                pred_text = ''.join([line[1][0].upper() for line in results[0]]).replace(' ', '')
            else:
                pred_text = ""
        except:
            pred_text = ""
        
        # Read ground truth
        stem = img_path.stem
        label_path = Path(test_label_dir) / f"{stem}.txt"
        if label_path.exists():
            with open(label_path) as f:
                gt_text = f.read().strip()
        else:
            gt_text = ""
        
        predictions.append(pred_text)
        ground_truths.append(gt_text)
        
        # Metrics
        if pred_text == gt_text:
            correct += 1
        
        # Character-level
        max_len = max(len(pred_text), len(gt_text))
        for i in range(max_len):
            p_char = pred_text[i] if i < len(pred_text) else ''
            g_char = gt_text[i] if i < len(gt_text) else ''
            if p_char == g_char:
                char_correct += 1
            char_total += 1
    
    # Metrics
    accuracy = correct / len(test_images)
    char_accuracy = char_correct / char_total if char_total > 0 else 0
    
    # Save report
    report = {
        'engine': 'PaddleOCR',
        'accuracy': accuracy,
        'char_accuracy': char_accuracy,
        'correct': correct,
        'total': len(test_images),
        'predictions_sample': [(p, g) for p, g in zip(predictions[:15], ground_truths[:15])]
    }
    
    report_path = Path('/home/akhil/3-2/reports/paddleocr_eval.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"PaddleOCR Results")
    print(f"{'='*70}")
    print(f"Plate Accuracy:      {accuracy:.1%} ({correct}/{len(test_images)})")
    print(f"Character Accuracy:  {char_accuracy:.1%}")
    print(f"Comparison to EasyOCR baseline (18.9%):")
    if accuracy > 0.189:
        print(f"  ✓ BETTER: +{(accuracy-0.189)*100:.1f}% improvement")
    else:
        print(f"  ✗ WORSE:  {(accuracy-0.189)*100:.1f}% (regressed)")
    print(f"  Multiplier: {accuracy/0.189:.2f}x")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    test_image_dir = '/home/akhil/3-2/datasets/processed/ocr_dataset/images/test'
    test_label_dir = '/home/akhil/3-2/datasets/processed/ocr_dataset/labels/test'
    test_paddleocr(test_image_dir, test_label_dir)
