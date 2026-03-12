"""
ANPR Processing Service
Handles detection and OCR processing
"""

import csv
import json
import re
import time
from pathlib import Path
from typing import List, Optional
import tempfile

import cv2
import numpy as np

from schemas import ANPRResult, PlateDetection


# ---------------------------------------------------------------------------
#  Indian RTO state/UT codes — used to anchor and validate plate prefixes
# ---------------------------------------------------------------------------
_INDIAN_STATE_CODES: frozenset[str] = frozenset({
    # States
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH",
    "JK", "KA", "KL", "LA", "MP", "MH", "MN", "ML", "MZ", "NL",
    "OD", "PB", "RJ", "SK", "TN", "TG", "TS", "TR", "UK", "UP", "WB",
    # Union Territories
    "CH", "DD", "DL", "LD", "PY",
    # Bharat series
    "BH",
})

# OCR confusion maps — applied only at specific zones of the plate
# Zone 1 (state code):  digits that look like letters
_OCR_LETTER_FIX: dict[str, str] = {
    "0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B",
}
# Zone 2 (district) and Zone 4 (number): letters that look like digits
_OCR_DIGIT_FIX: dict[str, str] = {
    "O": "0", "I": "1", "Z": "2", "S": "5", "G": "6", "B": "8",
}


def _fix_char_to_letter(c: str) -> str:
    """Best-effort convert a single OCR'd char to a letter (state-code zone)."""
    return _OCR_LETTER_FIX.get(c, c)


def _fix_char_to_digit(c: str) -> str:
    """Best-effort convert a single OCR'd char to a digit (district/number zone)."""
    return _OCR_DIGIT_FIX.get(c, c)


def _try_state_code(two: str) -> str | None:
    """
    Given two raw OCR characters, return the matching state code or None.
    Attempts: as-is → fix both chars → fix char[0] only → fix char[1] only.
    """
    candidates = [
        two,
        _fix_char_to_letter(two[0]) + _fix_char_to_letter(two[1]),
        _fix_char_to_letter(two[0]) + two[1],
        two[0] + _fix_char_to_letter(two[1]),
    ]
    for c in candidates:
        upper = c.upper()
        if upper in _INDIAN_STATE_CODES:
            return upper
    return None


def postprocess_indian_plate(raw_ocr: str) -> str:
    """
    Clean and validate an Indian number plate string from raw TrOCR output.

    Indian plate structure:
        [STATE 2L] [DISTRICT 2D] [SERIES 1–4L] [NUMBER 4D]
        e.g.  TS32T2514   AP28AL4708   DL09CAB5521   MH04CE8821

    Zone-based OCR-confusion corrections
    ─────────────────────────────────────
    • State   (chars 0–1) : digit→letter fixes   (e.g. 5→S, 0→O)
    • District (chars 2–3): letter→digit fixes    (e.g. O→0, I→1)
    • Series  (chars 4–N) : NO fixes — consecutive alpha chars taken as-is
    • Number  (last 4)    : letter→digit fixes    (e.g. B→8, I→1)

    If the state code cannot be resolved the stripped string is returned
    unchanged so we never produce output worse than the raw fallback.
    """
    stripped = re.sub(r"[^A-Za-z0-9]", "", raw_ocr).upper()

    if len(stripped) < 6:
        return stripped     # too short to parse — return as-is

    # ── Zone 1: state code (2 letters) ──────────────────────────────────────
    state = _try_state_code(stripped[:2])
    if state is None:
        return stripped     # unrecognised prefix — return stripped only

    remaining = stripped[2:]    # everything after the 2-char state code

    # ── Zone 2: district digits (exactly 2) ─────────────────────────────────
    district = ""
    i = 0
    while i < len(remaining) and len(district) < 2:
        ch = _fix_char_to_digit(remaining[i])
        if ch.isdigit():
            district += ch
            i += 1
        else:
            break   # hit something un-fixable — abort district parse

    if len(district) < 2:
        return state + remaining    # district parse failed

    remaining = remaining[i:]

    # ── Zone 3: series (1–4 consecutive ALPHA chars, no digit→letter fix) ───
    # We only take characters that are already alpha — digits delimit the end
    # of the series and the start of the 4-digit number.
    series = ""
    i = 0
    while i < len(remaining) and remaining[i].isalpha() and len(series) < 4:
        series += remaining[i]
        i += 1

    if not series:
        return state + district + remaining     # no series found

    remaining = remaining[i:]

    # ── Zone 4: number (exactly 4 digits, letter→digit fixes allowed) ───────
    number = ""
    i = 0
    while i < len(remaining) and len(number) < 4:
        ch = _fix_char_to_digit(remaining[i])
        if ch.isdigit():
            number += ch
        # skip non-digit garbage in the number field
        i += 1

    plate = state + district + series
    if number:
        plate += number
    return plate


class TrOCREngine:
    """
    OCR pipeline using microsoft/trocr-base-printed (Transformer-based OCR).

    - Phase 2 preprocessing: CLAHE contrast enhancement + bilateral denoising
    - Saves model locally after first download (no re-download on restart)
    - Confidence derived from beam-search sequence probability
    """

    MODEL_ID = "microsoft/trocr-base-printed"

    def __init__(self, model_save_dir: Path):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_save_dir.mkdir(parents=True, exist_ok=True)

        config_file = model_save_dir / "config.json"
        if config_file.exists():
            print(f"  Loading TrOCR from local cache: {model_save_dir}")
            self.processor = TrOCRProcessor.from_pretrained(str(model_save_dir))
            self.model = VisionEncoderDecoderModel.from_pretrained(str(model_save_dir))
        else:
            print(f"  First run – downloading TrOCR ({self.MODEL_ID}) …")
            self.processor = TrOCRProcessor.from_pretrained(self.MODEL_ID)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.MODEL_ID)
            self.processor.save_pretrained(str(model_save_dir))
            self.model.save_pretrained(str(model_save_dir))
            print(f"  TrOCR saved to {model_save_dir}")

        self.model.to(self.device).eval()
        print(f"  TrOCR ready on {'GPU' if self.device.type == 'cuda' else 'CPU'}")

    # ------------------------------------------------------------------
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Phase 2 preprocessing: CLAHE contrast + bilateral denoising."""
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image.copy()
        )
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        cleaned = cv2.bilateralFilter(contrast, 11, 17, 17)
        return cleaned

    def recognize(self, image: np.ndarray) -> tuple[str, float, str]:
        """
        Recognize plate text with TrOCR.

        Returns:
            (clean_text, confidence, raw_text)
            - clean_text  : state-code–anchored, structure-validated plate string
            - confidence  : 0–1 beam-search sequence probability
            - raw_text    : raw model output before post-processing
        """
        import math
        import torch
        from PIL import Image as PILImage

        cleaned = self._preprocess(image)
        pil_img = PILImage.fromarray(cleaned).convert("RGB")

        pixel_values = (
            self.processor(images=pil_img, return_tensors="pt")
            .pixel_values
            .to(self.device)
        )

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                num_beams=4,
                max_new_tokens=32,
                return_dict_in_generate=True,
                output_scores=True,
            )

        raw_text = self.processor.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )[0]
        clean_text = postprocess_indian_plate(raw_text)

        # sequence_scores = mean log-prob over tokens → exp gives probability proxy
        if getattr(outputs, "sequences_scores", None) is not None:
            score = outputs.sequences_scores[0].item()
            confidence = float(min(1.0, math.exp(max(-10.0, score))))
        else:
            confidence = 0.75  # safe fallback

        return clean_text, confidence, raw_text

    # ------------------------------------------------------------------
    @staticmethod
    def print_accuracy_summary(job_id: str, confidences: list) -> None:
        """Print a per-job TrOCR confidence / accuracy summary to stdout."""
        if not confidences:
            return
        avg    = sum(confidences) / len(confidences)
        high   = sum(1 for c in confidences if c >= 0.80)
        medium = sum(1 for c in confidences if 0.60 <= c < 0.80)
        low    = sum(1 for c in confidences if c < 0.60)
        total  = len(confidences)
        print(f"\n{'='*52}")
        print(f"TrOCR Accuracy Summary  [job: {job_id[:8]}…]")
        print(f"  Detections       : {total}")
        print(f"  Avg Confidence   : {avg * 100:.1f}%")
        print(f"  High  (≥80%)     : {high}/{total}")
        print(f"  Medium (60-80%)  : {medium}/{total}")
        print(f"  Low   (<60%)     : {low}/{total}")
        print(f"{'='*52}\n")


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
            self.model_root / "reports" / "plate_detection" / "yolov8s_6402" / "weights" / "best.pt",
            self.model_root / "reports" / "plate_detection" / "yolov8s_640" / "weights" / "best.pt",
            self.model_root / "runs" / "plate_detection" / "yolov8s_6402" / "weights" / "best.pt",
            self.model_root / "runs" / "plate_detection" / "yolov8s_640" / "weights" / "best.pt",
            self.model_root / "models" / "weights" / "yolov8s_license_plate_best.pt",
        ]
        
        weights_path = None
        for path in possible_paths:
            if path.exists():
                weights_path = path
                break
        
        # If still not found, search recursively (Python 3.12: ** must be a whole component)
        if weights_path is None:
            matches = [
                p for p in self.model_root.glob("**/weights/best.pt")
                if "yolov8s" in p.parts[-3]
            ]
            if matches:
                # Use the most recently modified one
                weights_path = max(matches, key=lambda p: p.stat().st_mtime)
        
        if weights_path is None or not weights_path.exists():
            raise FileNotFoundError(
                f"Could not find YOLO weights. Searched: {[str(p) for p in possible_paths]}"
            )
        
        print(f"Loading YOLO model from: {weights_path}")
        self.detector = YOLO(str(weights_path))
        
        # Initialize TrOCR
        trocr_save_dir = self.model_root / "models" / "trocr"
        print("Loading TrOCR model (microsoft/trocr-base-printed)...")
        self.ocr = TrOCREngine(trocr_save_dir)
        print("Models loaded successfully!")
    
    def process_images(self, job_id: str, image_paths: List[Path]) -> ANPRResult:
        """Process multiple images"""
        start_time = time.time()
        detections = []
        job_confidences = []
        
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
                    
                    # Recognize text with TrOCR
                    plate_text, confidence, raw_text = self.ocr.recognize(plate_crop)
                    job_confidences.append(confidence)

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
                            raw_ocr_text=raw_text,
                            ocr_engine="trocr",
                        )
                    )
        
        processing_time = time.time() - start_time

        TrOCREngine.print_accuracy_summary(job_id, job_confidences)

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
        job_confidences = []
        
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
                        
                        # Extract and recognize plate with TrOCR
                        plate_crop = frame[y1:y2, x1:x2]
                        plate_text, confidence, raw_text = self.ocr.recognize(plate_crop)
                        job_confidences.append(confidence)

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
                                raw_ocr_text=raw_text,
                                ocr_engine="trocr",
                            )
                        )
                        detection_idx += 1
            
            # Write frame
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        
        processing_time = time.time() - start_time

        TrOCREngine.print_accuracy_summary(job_id, job_confidences)

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
