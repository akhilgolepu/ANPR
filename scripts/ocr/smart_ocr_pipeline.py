"""
Simplified and Practical OCR Pipeline
Uses EasyOCR with intelligent postprocessing and format validation
Expected improvement: 18.9% → 35-45%
"""

import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import sys
import re
sys.path.insert(0, '/home/akhil/3-2')

import easyocr


class SmartOCRPipeline:
    """Practical OCR pipeline with intelligent postprocessing"""
    
    def __init__(self):
        """Initialize OCR engine"""
        print("Loading EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=True)
        print("✓ EasyOCR loaded")
        
        # Character confusion mapping (common OCR errors)
        self.char_fixes = {
            'O': '0',  # Letter O to digit 0
            'o': '0',
            'I': '1',  # Letter I to digit 1
            'i': '1',
            'L': '1',  # Letter L to digit 1
            'l': '1',
            'Z': '2',
            'z': '2',
            'G': '6',
            'g': '6',
            'S': '5',
            's': '5',
            'B': '8',
            'b': '8',
        }
    
    def preprocess_image(self, image):
        """Intelligent preprocessing for better OCR"""
        if image is None or image.size == 0:
            return image
        
        # Convert to RGB if needed (EasyOCR expects RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # CLAHE for contrast enhancement
        if len(rgb_image.shape) == 3:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb_image
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Convert back to RGB for EasyOCR
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return enhanced_rgb
    
    def smart_fix_characters(self, text, confidence_scores):
        """Intelligently fix OCR character errors based on position and context"""
        text = text.upper().replace(' ', '').replace('-', '')
        
        if len(text) < 8:  # Plates should be at least 8-10 chars
            return text
        
        fixed = list(text)
        
        # Position-based character fixing
        # Indian plate format: AA-DD-AA-DDDD (positions 0-9)
        char_positions = {
            (0, 1, 4, 5): 'letter',    # Positions 0,1,4,5 should be letters
            (2, 3, 6, 7, 8, 9): 'digit'  # Positions 2,3,6,7,8,9 should be digits
        }
        
        for i, char in enumerate(fixed):
            if i < len(fixed):
                if i in [0, 1, 4, 5]:  # Letter positions
                    # Convert common letter->digit confusions to letters
                    if char in 'OIL':  # Common letter misreadings
                        if char == 'O':
                            fixed[i] = 'O'  # Keep as O (it's a letter)
                        elif char == 'I':
                            fixed[i] = 'I'  # Keep as I (it's a letter)
                        elif char == 'L':
                            fixed[i] = 'L'  # Keep as L (it's a letter)
                else:  # Digit positions
                    # Convert letter->digit confusions
                    if char in self.char_fixes and self.char_fixes[char] in '0123456789':
                        fixed[i] = self.char_fixes[char]
        
        return ''.join(fixed)
    
    def validate_format(self, text):
        """Validate if text matches Indian license plate format"""
        text = text.upper().replace(' ', '').replace('-', '')
        
        # Allow 8-10 character plates
        if len(text) < 8:
            return False, 0.0
        
        # Truncate to 10 characters for Indian plates
        text = text[:10]
        
        if len(text) == 10:
            # Check format: AA-DD-AA-DDDD
            checks = [
                text[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                text[1] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                text[2] in '0123456789',
                text[3] in '0123456789',
                text[4] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                text[5] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                text[6] in '0123456789',
                text[7] in '0123456789',
                text[8] in '0123456789',
                text[9] in '0123456789',
            ]
            
            valid = all(checks)
            confidence = sum(checks) / len(checks)
            
            return valid, confidence
        
        return False, 0.0
    
    def extract_plate_number(self, text, confidence_scores):
        """Extract most likely plate number from text"""
        text = text.upper().replace(' ', '').replace('-', '')
        
        # Try to find 10-character sequences
        plates = []
        
        # Try sliding window if text is longer than 10
        for i in range(len(text) - 9):
            substr = text[i:i+10]
            valid, conf = self.validate_format(substr)
            if valid:
                plates.append((substr, conf))
        
        # If no valid 10-char found, try 9-char
        if not plates:
            for i in range(len(text) - 8):
                substr = text[i:i+9]
                valid, conf = self.validate_format(substr)
                if valid:
                    plates.append((substr, conf))
        
        # Return best plate or empty
        if plates:
            best = max(plates, key=lambda x: x[1])
            return best[0], best[1]
        
        # Fallback: just validate the whole text
        valid, conf = self.validate_format(text)
        if valid:
            return text[:10], conf
        
        return text if len(text) >= 8 else "", 0.0
    
    def recognize(self, image):
        """
        Recognize license plate with intelligent postprocessing
        
        Args:
            image: Plate crop image
        
        Returns:
            text: Recognized plate number (formatted)
            confidence: Confidence score
        """
        if image is None or image.size == 0:
            return "", 0.0
        
        # Preprocess
        processed = self.preprocess_image(image)
        
        # Recognize with EasyOCR
        try:
            results = self.reader.readtext(processed)
            
            if not results:
                return "", 0.0
            
            # Extract text and confidences
            texts = []
            confidences = []
            for detection in results:
                text, conf = detection[1], detection[2]
                texts.append(text)
                confidences.append(conf)
            
            # Combine all detected text
            combined_text = ''.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # Apply intelligent fixes
            fixed_text = self.smart_fix_characters(combined_text, confidences)
            
            # Extract and validate plate
            plate_text, format_conf = self.extract_plate_number(fixed_text, confidences)
            
            # Combined confidence: average OCR confidence + format validity
            final_confidence = avg_confidence * 0.6 + format_conf * 0.4
            
            return plate_text, final_confidence
            
        except Exception as e:
            print(f"Recognition error: {e}")
            return "", 0.0


def evaluate_smart_ocr(test_dir='/home/akhil/3-2/datasets/processed/ocr_dataset/images/test'):
    """Evaluate smart OCR pipeline"""
    
    model = SmartOCRPipeline()
    
    test_path = Path(test_dir)
    test_images = sorted(list(test_path.glob('*.jpg')) + list(test_path.glob('*.png')))
    
    if not test_images:
        print(f"No test images found in {test_dir}")
        return None
    
    print(f"\nEvaluating Smart OCR on {len(test_images)} test samples...")
    print("="*60)
    
    correct = 0
    char_correct = 0
    char_total = 0
    valid_format = 0
    high_confidence_correct = 0
    results_list = []
    
    for img_path in tqdm(test_images, desc="Testing"):
        # Read image and label
        img = cv2.imread(str(img_path))
        stem = img_path.stem
        label_path = img_path.parent.parent / 'labels' / (stem + '.txt')
        
        if label_path.exists():
            with open(label_path) as f:
                gt_text = f.read().strip().upper()
        else:
            gt_text = ""
        
        # Predict
        pred_text, confidence = model.recognize(img)
        
        # Metrics
        if pred_text == gt_text:
            correct += 1
            if confidence > 0.5:
                high_confidence_correct += 1
        
        if model.validate_format(pred_text)[0]:
            valid_format += 1
        
        # Character accuracy
        min_len = min(len(pred_text), len(gt_text))
        for i in range(min_len):
            if pred_text[i] == gt_text[i]:
                char_correct += 1
            char_total += 1
        
        results_list.append({
            'image': stem,
            'ground_truth': gt_text,
            'prediction': pred_text,
            'confidence': float(confidence),
            'correct': pred_text == gt_text,
            'valid_format': model.validate_format(pred_text)[0]
        })
    
    accuracy = correct / len(test_images) if len(test_images) > 0 else 0
    char_accuracy = char_correct / char_total if char_total > 0 else 0
    format_accuracy = valid_format / len(test_images) if len(test_images) > 0 else 0
    high_conf_accuracy = high_confidence_correct / correct if correct > 0 else 0
    
    # Save results
    results_path = Path('/home/akhil/3-2/reports/smart_ocr_results.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'char_accuracy': char_accuracy,
            'format_accuracy': format_accuracy,
            'high_confidence_correct': high_confidence_correct,
            'total_correct': correct,
            'total_samples': len(test_images),
            'samples': results_list
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Smart OCR Pipeline - Evaluation Results")
    print(f"{'='*60}")
    print(f"Plate Accuracy:        {accuracy:.1%} ({correct}/{len(test_images)})")
    print(f"Character Accuracy:    {char_accuracy:.1%}")
    print(f"Valid Format Rate:     {format_accuracy:.1%}")
    print(f"{'='*60}")
    print(f"\nComparison to Baseline (EasyOCR only - 18.9%):")
    print(f"  → Absolute improvement: +{(accuracy - 0.189) * 100:.1f} pp")
    if accuracy > 0.189:
        print(f"  → Relative improvement: {accuracy / 0.189:.2f}x")
    print(f"\nHigh confidence predictions: {high_confidence_correct}/{correct if correct > 0 else 1}")
    print(f"Results saved: {results_path}")
    print(f"{'='*60}\n")
    
    return {
        'accuracy': accuracy,
        'char_accuracy': char_accuracy,
        'format_accuracy': format_accuracy,
        'correct': correct,
        'total': len(test_images)
    }


if __name__ == '__main__':
    results = evaluate_smart_ocr()
    
    if results:
        print("\n✓ Evaluation complete!")
        print(f"\nSummary:")
        print(f"  • Baseline (EasyOCR raw): 18.9%")
        print(f"  • Smart OCR: {results['accuracy']:.1%}")
        print(f"  • Improvement: +{(results['accuracy'] - 0.189) * 100:.1f} percentage points")
