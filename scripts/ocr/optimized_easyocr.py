"""
Optimized OCR with Enhanced Postprocessing
Uses EasyOCR + Format Validation + Dictionary Corrections
Expected accuracy improvement: 18.9% → 40-50%
"""

import cv2
import numpy as np
import easyocr
from pathlib import Path
from tqdm import tqdm
import json
import re
from typing import Tuple, List
import sys

sys.path.insert(0, '/home/akhil/3-2')


class OptimizedOCR:
    """EasyOCR with enhanced postprocessing"""
    
    def __init__(self):
        print("Loading EasyOCR model...")
        self.reader = easyocr.Reader(['en'], gpu=True)
        print("✓ EasyOCR loaded")
        
        # Common character corrections for plates
        self.char_corrections = {
            'O': '0',  'l': '1', 'I': '1', 'Z': '2', 'S': '5',
            'B': '8', 'G': '9',  'D': '0',  'q': '9',
        }
        
        # Common partial corrections
        self.partial_corrections = {
            'MH OIAD': 'MH 01AD',  # Common false recognition
            'MH 9IAD': 'MH 91AD',
            'MH 0TAD': 'MH 01AD',
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Minimal preprocessing - just enhance contrast"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    
    def recognize_with_confidence(self, image: np.ndarray) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Recognize text with per-character confidence
        Returns: (concatenated_text, list_of_character_confidence_tuples)
        """
        # Preprocess
        preprocessed = self.preprocess_image(image)
        
        # Recognize
        results = self.reader.readtext(preprocessed, detail=1)
        
        if not results:
            return "", []
        
        # Extract characters with confidences
        char_confidences = []
        text = ""
        
        for detection in results:
            bbox, char, conf = detection
            char_upper = char.upper().strip()
            if char_upper:
                text += char_upper
                char_confidences.append((char_upper, conf))
        
        return text.replace(' ', ''), char_confidences
    
    def correct_low_confidence_chars(self, text: str, char_confs: List[Tuple[str, float]],
                                    confidence_threshold: float = 0.4) -> str:
        """
        Correct low-confidence character predictions
        """
        if not char_confs:
            return text
        
        corrected = list(text)
        
        for i, (char, conf) in enumerate(char_confs):
            if i >= len(corrected):
                break
            
            # If confidence is low, try substitution based on context
            if conf < confidence_threshold:
                # Position-based corrections (Indian plate format)
                if i < 2:  # First 2 should be letters
                    if char.isdigit():
                        corrected[i] = 'A'  # Default to A
                elif i < 4:  # Next 2 should be digits
                    if char.isalpha():
                        corrected[i] = '0'  # Default to 0
                elif i < 6:  # Next 2 can be letters or digits
                    if char in self.char_corrections:
                        corrected[i] = self.char_corrections[char]
                else:  # Last 4 should be digits
                    if char.isalpha():
                        corrected[i] = '0'
        
        return ''.join(corrected)
    
    def validate_and_correct_format(self, text: str) -> Tuple[str, bool]:
        """
        Validate and correct Indian license plate format
        Format: AA-DD-AA-DDDD (without dashes)
        """
        text = text.replace(' ', '').replace('-', '')
        
        if len(text) < 8 or len(text) > 10:
            return text, False
        
        # Pad or trim to 9-10 chars
        if len(text) < 9:
            text = text + 'X' * (9 - len(text))
        
        corrected = ""
        valid = True
        
        # Position 0-1: Letters
        for i in range(2):
            if i < len(text):
                if text[i].isalpha():
                    corrected += text[i]
                else:
                    corrected += 'A'
                    valid = False
            else:
                corrected += 'A'
                valid = False
        
        # Position 2-3: Digits
        for i in range(2, 4):
            if i < len(text):
                if text[i].isdigit():
                    corrected += text[i]
                else:
                    corrected += '0'
                    valid = False
            else:
                corrected += '0'
                valid = False
        
        # Position 4-5: Mostly letters, but can have digits (depends on state/category)
        for i in range(4, min(6, len(text))):
            if text[i].isalpha():
                corrected += text[i]
            elif text[i].isdigit():
                # Could be valid (some plates have mixed)
                corrected += text[i]
            else:
                corrected += 'A'
                valid = False
        
        # Position 6+: Digits
        for i in range(6, len(text)):
            if text[i].isdigit():
                corrected += text[i]
            else:
                corrected += '0'
                valid = False
        
        return corrected[:10], valid
    
    def recognize_optimized(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Optimized recognition with postprocessing
        """
        # Get raw recognition with confidences
        raw_text, char_confs = self.recognize_with_confidence(image)
        
        if not raw_text:
            return "", 0.0
        
        # Correct low-confidence characters
        corrected = self.correct_low_confidence_chars(raw_text, char_confs)
        
        # Validate and enforce format
        formatted, is_valid = self.validate_and_correct_format(corrected)
        
        # Calculate final confidence
        if char_confs:
            base_conf = np.mean([c[1] for c in char_confs])
        else:
            base_conf = 0.5
        
        # Confidence boost if format is valid
        if is_valid:
            final_conf = min(base_conf * 1.2, 1.0)
        else:
            final_conf = base_conf * 0.8
        
        return formatted, final_conf


def evaluate_optimized_ocr(test_image_dir: str, test_label_dir: str):
    """Evaluate optimized OCR"""
    ocr = OptimizedOCR()
    
    test_path = Path(test_image_dir)
    test_images = sorted(list(test_path.glob('*.jpg')) + list(test_path.glob('*.png')))
    
    print(f"\n{'='*70}")
    print(f"Evaluating Optimized EasyOCR with Postprocessing")
    print(f"Test set: {len(test_images)} images")
    print(f"{'='*70}\n")
    
    predictions = []
    ground_truths = []
    correct = 0
    char_correct = 0
    char_total = 0
    format_valid_count = 0
    
    for img_path in tqdm(test_images, desc="Processing", total=len(test_images)):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Resize to standard size
        image = cv2.resize(image, (128, 64))
        
        # Predict with optimization
        pred_text, confidence = ocr.recognize_optimized(image)
        
        # Check if valid format
        _, is_valid = ocr.validate_and_correct_format(pred_text)
        if is_valid:
            format_valid_count += 1
        
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
    
    # Compute metrics  
    accuracy = correct / len(test_images)
    char_accuracy = char_correct / char_total if char_total > 0 else 0
    format_validity = format_valid_count / len(test_images)
    
    # Save results
    report = {
        'accuracy': accuracy,
        'char_accuracy': char_accuracy,
        'format_validity': format_validity,
        'correct': correct,
        'total': len(test_images),
        'predictions_sample': [(p, g) for p, g in zip(predictions[:20], ground_truths[:20])]
    }
    
    report_path = Path('/home/akhil/3-2/reports/optimized_ocr_report.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"RESULTS: Optimized EasyOCR with Postprocessing")
    print(f"{'='*70}")
    print(f"Plate Accuracy:     {accuracy:.1%} ({correct}/{len(test_images)})")
    print(f"Character Accuracy: {char_accuracy:.1%}")
    print(f"Format Validity:    {format_validity:.1%}")
    print(f"Improvement vs baseline: +{(accuracy - 0.189) * 100:.1f}%")
    print(f"Expected total boost (vs raw EasyOCR): 1.0x → {accuracy/0.189:.1f}x")
    print(f"Report: {report_path}")
    print(f"{'='*70}\n")
    
    return accuracy, char_accuracy


if __name__ == '__main__':
    test_image_dir = '/home/akhil/3-2/datasets/processed/ocr_dataset/images/test'
    test_label_dir = '/home/akhil/3-2/datasets/processed/ocr_dataset/labels/test'
    
    acc, char_acc = evaluate_optimized_ocr(test_image_dir, test_label_dir)
