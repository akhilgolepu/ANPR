"""
ANPR Processing Service
Handles detection and OCR processing
"""

import csv
import json
import time
from pathlib import Path
from typing import List, Optional
import tempfile

import cv2
import numpy as np

from schemas import ANPRResult, PlateDetection


class SmartOCR:
    """Smart OCR pipeline with postprocessing"""
    
    def __init__(self):
        import easyocr
        self.reader = easyocr.Reader(['en'], gpu=True)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image for better OCR"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def fix_common_errors(self, text: str) -> str:
        """Fix common OCR errors specific to Indian license plates"""
        text = text.upper().strip()
        
        # Common replacements
        replacements = {
            'O': '0',  # Often confused
            'I': '1',  # Often confused
            'S': '5',  # Sometimes confused
            'B': '8',  # Sometimes confused
            'L': '1',  # Sometimes confused
        }
        
        # Apply context-aware fixes
        fixed = text
        
        # Indian plate format: AA-DD-AA-DDDD
        # State code (2 letters), then should be all digits, then 2 letters, then 4 digits
        parts = fixed.split('-') if '-' in fixed else [fixed]
        
        result_parts = []
        for i, part in enumerate(parts):
            if i == 0:  # First part: 2 letters
                part = ''.join(c for c in part if c.isalpha())[:2]
            elif i == 1:  # Second part: 2 digits
                part = ''.join(c for c in part if c.isdigit())[:2]
            elif i == 2:  # Third part: 2 letters
                part = ''.join(c for c in part if c.isalpha())[:2]
            elif i == 3:  # Fourth part: 4 digits
                part = ''.join(c for c in part if c.isdigit())[:4]
            result_parts.append(part)
        
        return '-'.join(result_parts)
    
    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        """Recognize plate text from image"""
        # Preprocess
        processed = self.preprocess_image(image)
        
        # OCR
        results = self.reader.readtext(processed, detail=1)
        
        if not results:
            return "", 0.0
        
        # Extract text and confidence
        texts = [result[1] for result in results]
        confidences = [result[2] for result in results]
        
        plate_text = ''.join(texts)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Fix common errors
        fixed_text = self.fix_common_errors(plate_text)
        
        return fixed_text, avg_confidence


class ProcessorService:
    """Main service for processing images and videos"""
    
    def __init__(self, model_root: Path, crops_dir: Path, results_dir: Path):
        self.model_root = model_root
        self.crops_dir = crops_dir
        self.results_dir = results_dir
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load YOLO and OCR models"""
        from ultralytics import YOLO
        # Find and load YOLO model - try multiple paths
        possible_paths = [
            self.model_root / "runs" / "plate_detection" / "yolov8s_640" / "weights" / "best.pt",
            self.model_root / "runs" / "plate_detection" / "yolov8s_6402" / "weights" / "best.pt",
            self.model_root / "models" / "weights" / "yolov8s_license_plate_best.pt",
        ]
        
        weights_path = None
        for path in possible_paths:
            if path.exists():
                weights_path = path
                break
        
        # If still not found, search recursively
        if weights_path is None:
            import glob
            matches = list(self.model_root.glob("**/yolov8s**/weights/best.pt"))
            if matches:
                # Use the most recently modified one
                weights_path = max(matches, key=lambda p: p.stat().st_mtime)
        
        if weights_path is None or not weights_path.exists():
            raise FileNotFoundError(
                f"Could not find YOLO weights. Searched: {[str(p) for p in possible_paths]}"
            )
        
        print(f"Loading YOLO model from: {weights_path}")
        self.detector = YOLO(str(weights_path))
        
        # Initialize OCR
        print("Loading EasyOCR model...")
        self.ocr = SmartOCR()
        print("Models loaded successfully!")
    
    def process_images(self, job_id: str, image_paths: List[Path]) -> ANPRResult:
        """Process multiple images"""
        start_time = time.time()
        detections = []
        
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            # Run detection
            results = self.detector.predict(image, conf=0.25, iou=0.7)
            
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Extract plate crop
                    plate_crop = image[y1:y2, x1:x2]
                    
                    # Recognize text
                    plate_text, confidence = self.ocr.recognize(plate_crop)
                    
                    # Save crops
                    vehicle_crop_path = self.crops_dir / f"{job_id}_vehicle_{i}.jpg"
                    plate_crop_path = self.crops_dir / f"{job_id}_plate_{i}.jpg"
                    
                    cv2.imwrite(str(vehicle_crop_path), image[max(0, y1-50):min(image.shape[0], y2+50), max(0, x1-50):min(image.shape[1], x2+50)])
                    cv2.imwrite(str(plate_crop_path), plate_crop)
                    
                    detections.append(
                        PlateDetection(
                            plate_text=plate_text,
                            confidence=float(confidence),
                            bbox=[x1, y1, x2, y2],
                            vehicle_crop_url=f"/static/{vehicle_crop_path.name}",
                            plate_crop_url=f"/static/{plate_crop_path.name}",
                        )
                    )
        
        processing_time = time.time() - start_time
        
        result = ANPRResult(
            job_id=job_id,
            status="completed",
            input_type="image",
            total_detections=len(detections),
            processing_time=processing_time,
            detections=detections,
        )
        
        # Save result
        self._save_result(job_id, result)
        
        return result
    
    def process_video(self, job_id: str, video_path: Path) -> ANPRResult:
        """Process video file"""
        start_time = time.time()
        detections = []
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        output_video_path = self.results_dir / f"{job_id}_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        frame_idx = 0
        detection_idx = 0
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection on every frame (or skip frames for speed)
            if frame_idx % max(1, int(fps)) == 0:  # Process 1 frame per second
                results = self.detector.predict(frame, conf=0.25, iou=0.7)
                
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Extract and recognize plate
                        plate_crop = frame[y1:y2, x1:x2]
                        plate_text, confidence = self.ocr.recognize(plate_crop)
                        
                        # Draw text
                        cv2.putText(frame, plate_text, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        # Save crops
                        vehicle_crop_path = self.crops_dir / f"{job_id}_vehicle_{detection_idx}.jpg"
                        plate_crop_path = self.crops_dir / f"{job_id}_plate_{detection_idx}.jpg"
                        
                        cv2.imwrite(str(vehicle_crop_path), frame[max(0, y1-50):min(height, y2+50), max(0, x1-50):min(width, x2+50)])
                        cv2.imwrite(str(plate_crop_path), plate_crop)
                        
                        detections.append(
                            PlateDetection(
                                plate_text=plate_text,
                                confidence=float(confidence),
                                bbox=[x1, y1, x2, y2],
                                vehicle_crop_url=f"/static/{vehicle_crop_path.name}",
                                plate_crop_url=f"/static/{plate_crop_path.name}",
                            )
                        )
                        detection_idx += 1
            
            # Write frame
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        
        processing_time = time.time() - start_time
        
        result = ANPRResult(
            job_id=job_id,
            status="completed",
            input_type="video",
            total_detections=len(detections),
            processing_time=processing_time,
            detections=detections,
            output_file_url=f"/videos/{output_video_path.name}",
        )
        
        # Save result
        self._save_result(job_id, result)
        
        return result
    
    def _save_result(self, job_id: str, result: ANPRResult):
        """Save result to JSON file"""
        result_file = self.results_dir / f"{job_id}.json"
        
        with open(result_file, "w") as f:
            json.dump(result.dict(), f, indent=2)
