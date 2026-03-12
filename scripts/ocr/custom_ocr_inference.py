"""
Inference script for custom OCR model
Uses trained CNN+LSTM model to recognize license plates
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '/home/akhil/3-2')

from src.ocr.custom_ocr_model import CustomOCRModel, CharacterMapping


class CustomOCRInference:
    """Inference engine for custom OCR model"""
    
    def __init__(self, model_path, device='cuda'):
        """
        Args:
            model_path: Path to saved model checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        self.char_mapping = CharacterMapping()
        
        # Load model
        self.model = CustomOCRModel(num_classes=self.char_mapping.num_classes)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Loaded model from {model_path}")
    
    def recognize(self, image):
        """
        Recognize text in plate image
        
        Args:
            image: (H, W, 3) BGR image or path to image
        
        Returns:
            text: Recognized text
            confidence: Average confidence score
        """
        # Load image if string path
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                return "", 0.0
        
        # Preprocess
        img = cv2.resize(image, (128, 64))
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # Add batch dim
        
        # Inference
        with torch.no_grad():
            logits = self.model(img_tensor)  # (1, seq_len, num_classes)
            probs = F.softmax(logits, dim=2)  # (1, seq_len, num_classes)
            
            # Greedy decode
            predictions = torch.argmax(logits, dim=2)  # (1, seq_len)
            confidences = torch.max(probs, dim=2)[0]  # (1, seq_len)
        
        # Decode to text
        pred_indices = predictions[0].cpu().numpy()
        pred_confidences = confidences[0].cpu().numpy()
        
        text = self.char_mapping.decode(pred_indices)
        
        # Average confidence (excluding blanks)
        valid_mask = pred_indices != self.char_mapping.blank_idx
        if valid_mask.any():
            avg_confidence = pred_confidences[valid_mask].mean()
        else:
            avg_confidence = 0.0
        
        return text, float(avg_confidence)
    
    def recognize_batch(self, images_list):
        """
        Recognize multiple images
        
        Args:
            images_list: List of images or image paths
        
        Returns:
            results: List of (text, confidence) tuples
        """
        results = []
        for img in images_list:
            text, conf = self.recognize(img)
            results.append((text, conf))
        return results


class BeamSearchDecoder:
    """Beam search decoder for better accuracy"""
    
    def __init__(self, char_mapping, beam_width=5):
        self.char_mapping = char_mapping
        self.beam_width = beam_width
    
    def decode(self, logits):
        """
        Beam search decode
        
        Args:
            logits: (seq_len, num_classes)
        
        Returns:
            text: Best decoded text
            confidence: Score of best path
        """
        seq_len, num_classes = logits.shape
        
        # Initialize: (text, score)
        beams = [("", 0.0)]
        
        for t in range(seq_len):
            # Get probabilities at time step t
            probs = F.softmax(logits[t], dim=0).cpu().numpy()
            
            new_beams = []
            for text, score in beams:
                # Try appending each character
                for char_idx in range(num_classes):
                    char_prob = probs[char_idx]
                    
                    # Skip if this is blank and last was same character
                    if char_idx == self.char_mapping.blank_idx:
                        new_text = text
                    else:
                        new_text = text + self.char_mapping.idx_to_char.get(char_idx, '')
                    
                    new_score = score + np.log(char_prob + 1e-10)
                    new_beams.append((new_text, new_score))
            
            # Keep top-k beams
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:self.beam_width]
        
        best_text = beams[0][0].strip()
        best_score = beams[0][1]
        
        return best_text, np.exp(best_score)


def evaluate_custom_model(model_path, test_dir):
    """
    Evaluate custom OCR model on test set
    
    Args:
        model_path: Path to trained model
        test_dir: Directory with test images
    """
    from tqdm import tqdm
    
    ocr = CustomOCRInference(model_path)
    test_path = Path(test_dir)
    
    # Find all test images
    test_images = sorted(list(test_path.glob('*.jpg')) + list(test_path.glob('*.png')))
    
    if not test_images:
        print(f"No images found in {test_dir}")
        return
    
    print(f"\nEvaluating on {len(test_images)} test images...")
    
    correct = 0
    char_correct = 0
    char_total = 0
    
    for img_path in tqdm(test_images):
        # Read ground truth
        stem = img_path.stem
        label_path = img_path.parent.parent / 'labels' / (stem + '.txt')
        
        if label_path.exists():
            with open(label_path) as f:
                gt_text = f.read().strip()
        else:
            gt_text = ""
        
        # Predict
        pred_text, conf = ocr.recognize(str(img_path))
        
        # Metrics
        if pred_text == gt_text:
            correct += 1
        
        for p, g in zip(pred_text, gt_text):
            if p == g:
                char_correct += 1
            char_total += 1
    
    accuracy = correct / len(test_images)
    char_accuracy = char_correct / char_total if char_total > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"Custom OCR Model Evaluation Results")
    print(f"{'='*50}")
    print(f"Plate Accuracy:     {accuracy:.1%} ({correct}/{len(test_images)})")
    print(f"Character Accuracy: {char_accuracy:.1%}")
    print(f"{'='*50}\n")
    
    return {
        'accuracy': accuracy,
        'char_accuracy': char_accuracy,
        'correct': correct,
        'total': len(test_images)
    }


if __name__ == '__main__':
    # Example usage
    model_path = '/home/akhil/3-2/models/ocr/custom_ocr_best.pt'
    test_dir = '/home/akhil/3-2/datasets/processed/ocr_dataset/images/test'
    
    if Path(model_path).exists():
        evaluate_custom_model(model_path, test_dir)
    else:
        print(f"Model not found: {model_path}")
        print("Train model first with: python scripts/train/train_custom_ocr.py")
