"""
Improved OCR Pipeline for License Plate Recognition
Uses PaddleOCR (optimized for Asian text) + intelligent postprocessing
Expected improvement: 18.9% → 45-55%
"""

import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/akhil/3-2')

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("⚠️  PaddleOCR not installed. Install with: pip install paddleocr")

import easyocr


class ImprovedOCRPipeline:
    """Advanced OCR pipeline with multiple engines and intelligent postprocessing"""
    
    def __init__(self, use_paddle=True, use_easyocr=True):
        """Initialize OCR engines"""
        self.use_paddle = use_paddle and PADDLE_AVAILABLE
        self.use_easyocr = use_easyocr
        
        if self.use_paddle:
            print("Loading PaddleOCR...")
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
            print("✓ PaddleOCR loaded")
        
        if self.use_easyocr:
            print("Loading EasyOCR...")
            self.easy_reader = easyocr.Reader(['en'], gpu=True)
            print("✓ EasyOCR loaded")
        
        # Indian license plate format: AA-DD-AA-DDDD
        # Example: TS-09-AB-1234 or TS09AB1234
        self.plate_format_regex = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$'
        
        # Character confusion mapping (common OCR errors)
        self.char_fixes = {
            'O': '0',  # Letter O to digit 0
            'o': '0',
            'I': '1',  # Letter I to digit 1
            'i': '1',
            'L': '1',  # Letter L to digit 1
            'l': '1',
            'Z': '2',
            'G': '6',
            'g': '6',
            'S': '5',
            's': '5',
            'B': '8',
            'b': '8',
        }
    
    def preprocess_image(self, image):
        """Advanced preprocessing for better OCR"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Adaptive thresholding for better text visibility
        # This helps with varying lighting conditions
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Optional: Denoise
        denoised = cv2.fastNlMeansDenoising(binary, h=10)
        
        return denoised
    
    def fix_common_errors(self, text):
        """Fix common OCR character confusions"""
        text = text.upper().replace(' ', '')
        
        # Fix character confusions
        fixed = ""
        for i, char in enumerate(text):
            if char in self.char_fixes:
                # Apply fix intelligently based on position
                # Positions 0-1: Should be letters
                # Positions 2-3: Should be digits
                # Positions 4-5: Should be letters
                # Positions 6-9: Should be digits
                
                if i in [0, 1, 4, 5]:  # Letter positions
                    if char in '01':  # If it's a digit, might be misread letter
                        fixed += self.char_fixes[char] if char in 'Oil' else char
                    else:
                        fixed += char
                else:  # Digit positions
                    if char in 'OILZGSBilozgsb':  # If it's a letter, might be misread digit
                        fixed += self.char_fixes[char]
                    else:
                        fixed += char
            else:
                fixed += char
        
        return fixed
    
    def validate_format(self, text):
        """Validate if text matches Indian license plate format"""
        text = text.upper().replace(' ', '').replace('-', '')
        
        if len(text) != 10:
            return False, 0
        
        # Check format: AA-DD-AA-DDDD
        # 0-1: letters
        # 2-3: digits
        # 4-5: letters
        # 6-9: digits
        
        checks = [
            (text[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 0.15),
            (text[1] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 0.15),
            (text[2] in '0123456789', 0.1),
            (text[3] in '0123456789', 0.1),
            (text[4] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 0.15),
            (text[5] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 0.15),
            (text[6] in '0123456789', 0.1),
            (text[7] in '0123456789', 0.1),
            (text[8] in '0123456789', 0.1),
            (text[9] in '0123456789', 0.1),
        ]
        
        valid = all(check[0] for check in checks)
        confidence = sum(check[1] for check in checks if check[0])
        
        return valid, confidence
    
    def recognize_paddle(self, image):
        """Recognize using PaddleOCR"""
        if not self.use_paddle:
            return None, 0
        
        try:
            results = self.paddle_ocr.ocr(image, cls=True)
            
            if not results or not results[0]:
                return None, 0
            
            # Extract text with highest confidence
            texts = []
            for detection in results[0]:
                text, conf = detection[1], detection[2]
                texts.append((text, conf))
            
            if not texts:
                return None, 0
            
            # Combine all text
            combined_text = ''.join([t[0] for t in texts])
            avg_conf = np.mean([t[1] for t in texts])
            
            return combined_text, avg_conf
        except Exception as e:
            print(f"PaddleOCR error: {e}")
            return None, 0
    
    def recognize_easy(self, image):
        """Recognize using EasyOCR"""
        if not self.use_easyocr:
            return None, 0
        
        try:
            results = self.easy_reader.readtext(image)
            
            if not results:
                return None, 0
            
            # Extract text with highest confidence
            texts = []
            for detection in results:
                text, conf = detection[1], detection[2]
                texts.append((text, conf))
            
            if not texts:
                return None, 0
            
            # Combine all text
            combined_text = ''.join([t[0] for t in texts])
            avg_conf = np.mean([t[1] for t in texts])
            
            return combined_text, avg_conf
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return None, 0
    
    def recognize(self, image):
        """
        Recognize license plate with intelligent pipeline
        
        Args:
            image: Plate crop image
        
        Returns:
            text: Recognized plate number
            confidence: Confidence score
        """
        # Preprocess
        processed = self.preprocess_image(image)
        
        # Try multiple engines with voting
        results = []
        
        # PaddleOCR (optimized for Asian text)
        if self.use_paddle:
            text_p, conf_p = self.recognize_paddle(processed)
            if text_p:
                text_p = self.fix_common_errors(text_p)
                valid_p, format_conf = self.validate_format(text_p)
                results.append({
                    'text': text_p,
                    'confidence': conf_p * 0.7 + format_conf * 0.3,  # Weighted by format validity
                    'valid': valid_p
                })
        
        # EasyOCR
        if self.use_easyocr:
            text_e, conf_e = self.recognize_easy(processed)
            if text_e:
                text_e = self.fix_common_errors(text_e)
                valid_e, format_conf = self.validate_format(text_e)
                results.append({
                    'text': text_e,
                    'confidence': conf_e * 0.7 + format_conf * 0.3,
                    'valid': valid_e
                })
        
        if not results:
            return None, 0
        
        # Select best result
        valid_results = [r for r in results if r['valid']]
        if valid_results:
            best = max(valid_results, key=lambda x: x['confidence'])
        else:
            # Fallback to highest confidence (even if not valid format)
            best = max(results, key=lambda x: x['confidence'])
        
        return best['text'], best['confidence']


def evaluate_improved_ocr(model=None, test_dir='/home/akhil/3-2/datasets/processed/ocr_dataset/images/test'):
    """Evaluate improved OCR pipeline"""
    
    if model is None:
        model = ImprovedOCRPipeline(use_paddle=PADDLE_AVAILABLE, use_easyocr=True)
    
    test_path = Path(test_dir)
    test_images = sorted(list(test_path.glob('*.jpg')) + list(test_path.glob('*.png')))
    
    if not test_images:
        print(f"No test images found in {test_dir}")
        return None
    
    print(f"\nEvaluating on {len(test_images)} test samples...")
    
    correct = 0
    char_correct = 0
    char_total = 0
    valid_format = 0
    results_list = []
    
    for img_path in tqdm(test_images, desc="Testing"):
        # Read image and label
        img = cv2.imread(str(img_path))
        stem = img_path.stem
        label_path = img_path.parent.parent / 'labels' / (stem + '.txt')
        
        if label_path.exists():
            with open(label_path) as f:
                gt_text = f.read().strip()
        else:
            gt_text = ""
        
        # Predict
        pred_text, confidence = model.recognize(img)
        
        if pred_text is None:
            pred_text = ""
            confidence = 0
        
        # Metrics
        if pred_text == gt_text:
            correct += 1
        
        if model.validate_format(pred_text)[0]:
            valid_format += 1
        
        # Character accuracy
        for p, g in zip(pred_text, gt_text):
            if p == g:
                char_correct += 1
            char_total += 1
        
        results_list.append({
            'image': stem,
            'ground_truth': gt_text,
            'prediction': pred_text,
            'confidence': confidence,
            'correct': pred_text == gt_text
        })
    
    accuracy = correct / len(test_images)
    char_accuracy = char_correct / char_total if char_total > 0 else 0
    format_accuracy = valid_format / len(test_images)
    
    # Save results
    results_path = Path('/home/akhil/3-2/reports/improved_ocr_results.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'char_accuracy': char_accuracy,
            'format_accuracy': format_accuracy,
            'samples': results_list
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Improved OCR Pipeline - Evaluation Results")
    print(f"{'='*60}")
    print(f"Plate Accuracy:        {accuracy:.1%} ({correct}/{len(test_images)})")
    print(f"Character Accuracy:    {char_accuracy:.1%}")
    print(f"Valid Format Rate:     {format_accuracy:.1%}")
    print(f"{'='*60}")
    print(f"\nImprovement over baseline (18.9%):")
    print(f"  → +{(accuracy - 0.189) * 100:.1f}% absolute")
    print(f"  → {accuracy / 0.189:.1f}x relative improvement")
    print(f"\nResults saved: {results_path}")
    
    return {
        'accuracy': accuracy,
        'char_accuracy': char_accuracy,
        'format_accuracy': format_accuracy,
        'correct': correct,
        'total': len(test_images)
    }


if __name__ == '__main__':
    if PADDLE_AVAILABLE:
        print("✓ PaddleOCR available - using optimized pipeline")
    else:
        print("⚠️  PaddleOCR not available - will use EasyOCR only")
        print("  Install: pip install paddleocr")
    
    evaluate_improved_ocr()
