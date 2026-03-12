"""
Compare multiple OCR engines on license plate recognition
Tests EasyOCR, Tesseract, and PaddleOCR
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict
import sys

sys.path.insert(0, '/home/akhil/3-2')

# Import EasyOCR
import easyocr
# Import PaddleOCR
from paddleocr import PaddleOCR
# Import Tesseract
import pytesseract

from src.ocr.metrics import normalize_plate_text, compute_char_accuracy


class OCRComparison:
    """Compare multiple OCR engines"""
    
    def __init__(self):
        print("Initializing OCR engines...")
        
        # EasyOCR (English)
        self.easyocr = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        print("✓ EasyOCR loaded")
        
        # PaddleOCR
        self.paddleocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=torch.cuda.is_available())
        print("✓ PaddleOCR loaded")
        
        print("✓ All engines ready\n")
    
    def recognize_easyocr(self, image_path):
        """Recognize text using EasyOCR"""
        try:
            result = self.easyocr.readtext(str(image_path), detail=1)
            if not result:
                return "", 0.0
            
            # Sort by x-coordinate (left to right)
            result = sorted(result, key=lambda x: x[0][0][0])
            
            text = ''.join([item[1] for item in result]).strip()
            confidence = np.mean([item[2] for item in result])
            
            return text, confidence
        except Exception as e:
            return "", 0.0
    
    def recognize_paddleocr(self, image_path):
        """Recognize text using PaddleOCR"""
        try:
            result = self.paddleocr.ocr(str(image_path), cls=True)
            if not result or not result[0]:
                return "", 0.0
            
            # PaddleOCR returns nested list, flatten and sort
            texts = []
            confidences = []
            for line in result:
                for item in line:
                    texts.append(item[1])
                    confidences.append(item[2])
            
            text = ''.join(texts).strip()
            confidence = np.mean(confidences) if confidences else 0.0
            
            return text, confidence
        except Exception as e:
            return "", 0.0
    
    def recognize_tesseract(self, image_path):
        """Recognize text using Tesseract"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return "", 0.0
            
            # Preprocess for Tesseract
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            text = pytesseract.image_to_string(enhanced).strip()
            # Tesseract doesn't return confidence easily
            confidence = 0.5 if text else 0.0
            
            return text, confidence
        except Exception as e:
            return "", 0.0
    
    def postprocess(self, text):
        """Postprocess OCR output"""
        return normalize_plate_text(text)
    
    def evaluate(self, test_image_dir, test_label_dir):
        """Evaluate all engines"""
        test_path = Path(test_image_dir)
        label_path = Path(test_label_dir)
        
        test_images = sorted(list(test_path.glob('*.jpg')) + list(test_path.glob('*.png')))
        
        results = {
            'easyocr': {'correct': 0, 'char_correct': 0, 'char_total': 0, 'predictions': []},
            'paddleocr': {'correct': 0, 'char_correct': 0, 'char_total': 0, 'predictions': []},
            'tesseract': {'correct': 0, 'char_correct': 0, 'char_total': 0, 'predictions': []},
        }
        
        print(f"Evaluating {len(test_images)} test images...\n")
        
        for img_path in tqdm(test_images, desc="Processing"):
            stem = img_path.stem
            label_file = label_path / f"{stem}.txt"
            
            if not label_file.exists():
                continue
            
            with open(label_file) as f:
                gt_text = f.read().strip()
            
            # EasyOCR
            easy_text, easy_conf = self.recognize_easyocr(img_path)
            easy_text = self.postprocess(easy_text)
            
            if easy_text == gt_text:
                results['easyocr']['correct'] += 1
            char_correct_easy, char_total_easy = compute_char_accuracy(easy_text, gt_text)
            results['easyocr']['char_correct'] += char_correct_easy
            results['easyocr']['char_total'] += char_total_easy
            results['easyocr']['predictions'].append({
                'image': stem,
                'gt': gt_text,
                'pred': easy_text,
                'confidence': float(easy_conf)
            })
            
            # PaddleOCR
            paddle_text, paddle_conf = self.recognize_paddleocr(img_path)
            paddle_text = self.postprocess(paddle_text)
            
            if paddle_text == gt_text:
                results['paddleocr']['correct'] += 1
            char_correct_paddle, char_total_paddle = compute_char_accuracy(paddle_text, gt_text)
            results['paddleocr']['char_correct'] += char_correct_paddle
            results['paddleocr']['char_total'] += char_total_paddle
            results['paddleocr']['predictions'].append({
                'image': stem,
                'gt': gt_text,
                'pred': paddle_text,
                'confidence': float(paddle_conf)
            })
            
            # Tesseract
            tess_text, tess_conf = self.recognize_tesseract(img_path)
            tess_text = self.postprocess(tess_text)
            
            if tess_text == gt_text:
                results['tesseract']['correct'] += 1
            char_correct_tess, char_total_tess = compute_char_accuracy(tess_text, gt_text)
            results['tesseract']['char_correct'] += char_correct_tess
            results['tesseract']['char_total'] += char_total_tess
            results['tesseract']['predictions'].append({
                'image': stem,
                'gt': gt_text,
                'pred': tess_text,
                'confidence': float(tess_conf)
            })
        
        # Compute metrics
        total_images = len(results['easyocr']['predictions'])
        
        report = {
            'total_images': total_images,
            'easyocr': {
                'accuracy': results['easyocr']['correct'] / total_images if total_images > 0 else 0,
                'char_accuracy': results['easyocr']['char_correct'] / results['easyocr']['char_total'] if results['easyocr']['char_total'] > 0 else 0,
                'exact_matches': results['easyocr']['correct'],
            },
            'paddleocr': {
                'accuracy': results['paddleocr']['correct'] / total_images if total_images > 0 else 0,
                'char_accuracy': results['paddleocr']['char_correct'] / results['paddleocr']['char_total'] if results['paddleocr']['char_total'] > 0 else 0,
                'exact_matches': results['paddleocr']['correct'],
            },
            'tesseract': {
                'accuracy': results['tesseract']['correct'] / total_images if total_images > 0 else 0,
                'char_accuracy': results['tesseract']['char_correct'] / results['tesseract']['char_total'] if results['tesseract']['char_total'] > 0 else 0,
                'exact_matches': results['tesseract']['correct'],
            },
        }
        
        return report, results
    
    def print_report(self, report):
        """Print evaluation report"""
        print("\n" + "="*70)
        print("OCR ENGINE COMPARISON REPORT")
        print("="*70)
        print(f"\nTotal Test Images: {report['total_images']}\n")
        
        print(f"{'Engine':<15} {'Accuracy':<15} {'Char Accuracy':<15} {'Exact Matches':<15}")
        print("-"*70)
        
        for engine in ['easyocr', 'paddleocr', 'tesseract']:
            acc = report[engine]['accuracy']
            char_acc = report[engine]['char_accuracy']
            matches = report[engine]['exact_matches']
            
            print(f"{engine:<15} {acc:>6.1%}        {char_acc:>6.1%}           {matches:>6} / {report['total_images']:<6}")
        
        print("\n" + "="*70 + "\n")
        
        # Show improvement
        easy_acc = report['easyocr']['accuracy']
        paddle_acc = report['paddleocr']['accuracy']
        tess_acc = report['tesseract']['accuracy']
        
        improvement = (paddle_acc - easy_acc) / easy_acc if easy_acc > 0 else 0
        
        if paddle_acc > easy_acc:
            print(f"✓ PaddleOCR improves over EasyOCR by {improvement:+.1%}")
        elif easy_acc > paddle_acc:
            print(f"✗ EasyOCR still better than PaddleOCR by {abs(improvement):+.1%}")
        
        print()


def main():
    test_image_dir = '/home/akhil/3-2/datasets/processed/ocr_dataset/images/test'
    test_label_dir = '/home/akhil/3-2/datasets/processed/ocr_dataset/labels/test'
    
    comparator = OCRComparison()
    report, results = comparator.evaluate(test_image_dir, test_label_dir)
    comparator.print_report(report)
    
    # Save detailed results
    report_path = Path('/home/akhil/3-2/reports/ocr_comparison_report.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save report (without predictions for brevity)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Report saved to {report_path}")
    
    # Show examples of predictions
    print("\nExample Predictions (PaddleOCR):")
    print("-"*70)
    
    paddle_preds = results['paddleocr']['predictions']
    
    # Show correct predictions
    correct_preds = [p for p in paddle_preds if p['pred'] == p['gt']]
    print(f"\n✓ Correct predictions ({len(correct_preds)}):")
    for pred in correct_preds[:3]:
        print(f"  {pred['image']}: {pred['gt']} (conf: {pred['confidence']:.3f})")
    
    # Show incorrect predictions
    incorrect_preds = [p for p in paddle_preds if p['pred'] != p['gt']]
    print(f"\n✗ Incorrect predictions ({len(incorrect_preds)}):")
    for pred in incorrect_preds[:3]:
        print(f"  {pred['image']}: GT={pred['gt']}, Pred={pred['pred']} (conf: {pred['confidence']:.3f})")


if __name__ == '__main__':
    main()
