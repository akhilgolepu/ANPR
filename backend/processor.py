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
    # Core digit-looks-like-letter confusions on Indian plate fonts
    "0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B",
}
# Zone 2 (district) and Zone 4 (number): letters that look like digits
_OCR_DIGIT_FIX: dict[str, str] = {
    # Core letter-looks-like-digit confusions on Indian plate fonts
    "O": "0", "I": "1", "J": "1", "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8",
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
        [STATE 2L] [DISTRICT 2D] [SERIES 1-4L] [NUMBER 4D]
        e.g.  TS32T2514   AP28AL4708   DL09CAB5521   MH04CE8821

    Key improvement — sliding-window state-code search:
        TrOCR sometimes prepends a stray character before the state code
        (e.g. reads "ATS32T2514" instead of "TS32T2514").  We scan offsets
        0, 1 and 2 to find the first valid state-code match, then discard
        everything before it.  This anchors the plate to the only unambiguous
        two-letter prefix in the entire Indian RTO code table.

    Zone-based OCR-confusion corrections
    ─────────────────────────────────────
    - State   (chars 0-1) : digit->letter fixes   (e.g. 5->S, 0->O)
    - District (chars 2-3): letter->digit fixes    (e.g. O->0, I->1)
    - Series  (chars 4-N) : no fixes — consecutive alpha chars taken as-is
    - Number  (last 4)    : letter->digit fixes    (e.g. B->8, I->1)
    """
    stripped = re.sub(r"[^A-Za-z0-9]", "", raw_ocr).upper()

    if len(stripped) < 4:
        return stripped     # too short to parse — return as-is

    # ── Sliding-window state-code search (offsets 0, 1, 2) ──────────────────
    # Accept the first offset where a valid (possibly OCR-corrected) state code
    # is found.  This handles stray leading characters without guessing.
    state = None
    remaining = ""
    for offset in range(min(3, len(stripped) - 1)):
        candidate = stripped[offset: offset + 2]
        found = _try_state_code(candidate)
        if found:
            state = found
            remaining = stripped[offset + 2:]
            break

    if state is None:
        return stripped     # unrecognised prefix — return stripped only

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

    # ── Zone 3: series (1-4 consecutive ALPHA chars) ─────────────────────────
    series = ""
    i = 0
    while i < len(remaining) and remaining[i].isalpha() and len(series) < 4:
        series += remaining[i]
        i += 1

    if not series:
        return state + district + remaining     # no series found

    remaining = remaining[i:]

    # ── Zone 4: number (exactly 4 digits, letter->digit fixes allowed) ───────
    number = ""
    i = 0
    while i < len(remaining) and len(number) < 4:
        ch = _fix_char_to_digit(remaining[i])
        if ch.isdigit():
            number += ch
        i += 1

    plate = state + district + series
    if number:
        plate += number
    return plate


def _vote_characters(texts: list[str]) -> str:
    """
    Character-level majority vote across multiple OCR hypothesis strings.

    Given several candidate strings (e.g. the 5 decoder beams from TrOCR),
    narrows to the strings that share the most-common length (modal length),
    then picks the plurality character at each position.  This cancels out
    single-step decoder mistakes that would dominate a greedy or top-beam
    read, without requiring a second inference pass.

    Falls back to texts[0] unchanged if voting cannot be determined.
    """
    from collections import Counter

    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]

    # Restrict to strings with the modal (most common) length.
    target_len = Counter(len(t) for t in texts).most_common(1)[0][0]
    candidates = [t for t in texts if len(t) == target_len]
    if not candidates:
        return texts[0]

    # Plurality vote at each character position.
    return "".join(
        Counter(t[pos] for t in candidates).most_common(1)[0][0]
        for pos in range(target_len)
    )


def _top_candidate_list(texts: list[str], limit: int = 3) -> list[str]:
    """Return unique cleaned OCR candidates in beam order."""
    unique: list[str] = []
    seen: set[str] = set()
    for text in texts:
        cleaned = postprocess_indian_plate(text)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
        if len(unique) >= limit:
            break
    return unique


def score_indian_plate_format(text: str) -> float:
    """Score how well a plate string fits the expected Indian plate structure."""
    stripped = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    if not stripped:
        return 0.0

    score = 0.0
    if 8 <= len(stripped) <= 10:
        score += 0.10

    state = stripped[:2]
    if len(state) == 2 and _try_state_code(state) is not None:
        score += 0.35

    district = stripped[2:4]
    if len(district) == 2 and all(_fix_char_to_digit(ch).isdigit() for ch in district):
        score += 0.20

    suffix = stripped[4:]
    number = suffix[-4:] if len(suffix) >= 4 else ""
    series = suffix[:-4] if len(suffix) >= 4 else suffix

    if 1 <= len(series) <= 4 and series.isalpha():
        score += 0.20

    if len(number) == 4 and all(_fix_char_to_digit(ch).isdigit() for ch in number):
        score += 0.15

    return round(min(score, 1.0), 3)


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
            self.processor = TrOCRProcessor.from_pretrained(str(model_save_dir), use_fast=True)
            self.model = VisionEncoderDecoderModel.from_pretrained(str(model_save_dir))
        else:
            print(f"  First run – downloading TrOCR ({self.MODEL_ID}) …")
            self.processor = TrOCRProcessor.from_pretrained(self.MODEL_ID, use_fast=True)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.MODEL_ID)
            self.processor.save_pretrained(str(model_save_dir))
            self.model.save_pretrained(str(model_save_dir))
            print(f"  TrOCR saved to {model_save_dir}")

        self.model.to(self.device).eval()
        print(f"  TrOCR ready on {'GPU' if self.device.type == 'cuda' else 'CPU'}")

    # ------------------------------------------------------------------
    # Minimum crop height TrOCR handles well — anything smaller is upscaled.
    _MIN_H = 48
    # Target height we resize all crops to before padding.
    _TARGET_H = 128
    # White border padding (pixels) added around the resized crop.
    _PAD = 12
    # Confidence threshold below which we try the Otsu binarisation variant.
    _FALLBACK_THRESHOLD = 0.75

    def _prepare_crop(self, gray: np.ndarray) -> np.ndarray:
        """
        Resize → sharpen → pad a grayscale plate crop for TrOCR.

        Steps
        ─────
        1. Upscale to _TARGET_H height (maintains aspect ratio, INTER_CUBIC).
        2. Unsharp-mask sharpening to clarify character edges.
        3. Add a white (_PAD px) border — TrOCR was trained on word images
           with surrounding whitespace; this recovers that context.
        """
        h, w = gray.shape
        # 1. Upscale if needed
        scale = self._TARGET_H / max(h, 1)
        if scale > 1.0:
            new_w = max(1, int(w * scale))
            gray = cv2.resize(gray, (new_w, self._TARGET_H), interpolation=cv2.INTER_CUBIC)
        # 2. Unsharp mask (sharpen)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
        gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
        # 3. White border padding
        gray = cv2.copyMakeBorder(
            gray, self._PAD, self._PAD, self._PAD, self._PAD,
            cv2.BORDER_CONSTANT, value=255,
        )
        return gray

    def _clahe_variant(self, gray: np.ndarray) -> np.ndarray:
        """CLAHE contrast enhancement + bilateral denoising (primary variant)."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        denoised = cv2.bilateralFilter(contrast, 11, 17, 17)
        return self._prepare_crop(denoised)

    def _otsu_variant(self, gray: np.ndarray) -> np.ndarray:
        """Otsu binarisation — better for high-contrast / clean plates."""
        _, binarised = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self._prepare_crop(binarised)

    def _adaptive_variant(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive threshold — better for plates with uneven illumination."""
        adapted = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
        )
        return self._prepare_crop(adapted)

    # ------------------------------------------------------------------
    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect and correct plate skew using Hough line analysis.

        Only runs on crops tall enough for reliable edge detection
        (>= _MIN_H px).  Skips correction when the estimated angle is
        < 0.5° (negligible) or > 20° (likely noise / perspective — a
        perspective warp needs homography, not a simple rotation).
        """
        h, w = gray.shape
        if h < self._MIN_H:
            return gray  # too small — Hough would fire on noise

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # Hough threshold scales with width so we don't over-fire on tiny crops.
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=max(20, w // 4))
        if lines is None:
            return gray

        angles = []
        for line in lines[:20]:  # top-20 strongest lines is enough
            rho, theta = line[0]
            angle = float(np.degrees(theta)) - 90.0
            if abs(angle) < 20.0:  # discard near-vertical / steep lines
                angles.append(angle)

        if not angles:
            return gray

        skew = float(np.median(angles))
        if abs(skew) < 0.5:
            return gray  # negligible tilt — skip warpAffine overhead

        M = cv2.getRotationMatrix2D((w // 2, h // 2), skew, 1.0)
        return cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image.copy()
        )
        return self._deskew(gray)

    def _run_trocr(self, processed_gray: np.ndarray):
        """Run a single TrOCR inference pass. Returns raw text, confidence, and beam candidates."""
        import math
        import torch
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(processed_gray).convert("RGB")
        pixel_values = (
            self.processor(images=pil_img, return_tensors="pt")
            .pixel_values
            .to(self.device)
        )
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                num_beams=5,
                num_return_sequences=5,   # return ALL beams, not just the top one
                max_new_tokens=32,
                return_dict_in_generate=True,
                output_scores=True,
            )
        # Decode all 5 beams and elect each character by plurality vote.
        # This reduces errors caused by a single bad decoder step locking in
        # the wrong character for the rest of the sequence.
        all_texts = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)
        raw = _vote_characters(all_texts)
        # Confidence comes from the top beam (sequences_scores[0]).
        if getattr(outputs, "sequences_scores", None) is not None:
            score = outputs.sequences_scores[0].item()
            conf = float(min(1.0, math.exp(max(-10.0, score))))
        else:
            conf = 0.75
        return raw, conf, all_texts

    def recognize(self, image: np.ndarray) -> tuple[str, float, str, list[str], float]:
        """
        Recognize plate text with TrOCR using a multi-variant strategy.

        Strategy
        ────────
        1. Primary pass: CLAHE + bilateral + upscale + pad + sharpen.
        2. If confidence < _FALLBACK_THRESHOLD, run Otsu binarisation and
           adaptive threshold variants; keep whichever gives highest confidence.

        Returns:
            (clean_text, confidence, raw_text, top_candidates, format_score)
        """
        gray = self._to_gray(image)

        # ── Primary pass ──────────────────────────────────────────────────────
        raw, conf, texts = self._run_trocr(self._clahe_variant(gray))
        best_raw, best_conf, best_texts = raw, conf, texts

        # ── Fallback variants (only when primary is uncertain) ────────────────
        if conf < self._FALLBACK_THRESHOLD:
            for variant_fn in (self._otsu_variant, self._adaptive_variant):
                r, c, texts = self._run_trocr(variant_fn(gray))
                if c > best_conf:
                    best_raw, best_conf, best_texts = r, c, texts
                    # Early-exit if we've already reached a good confidence
                    if best_conf >= self._FALLBACK_THRESHOLD:
                        break

        clean_text = postprocess_indian_plate(best_raw)
        return clean_text, best_conf, best_raw, _top_candidate_list(best_texts), score_indian_plate_format(clean_text)

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

    def _expand_box(self, x1: int, y1: int, x2: int, y2: int, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        """Expand a detected plate box so OCR doesn't lose edge characters."""
        height, width = shape[:2]
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        pad_x = max(2, int(box_w * 0.12))
        pad_y = max(2, int(box_h * 0.28))
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width, x2 + pad_x),
            min(height, y2 + pad_y),
        )

    def _crop_quality(self, crop: np.ndarray) -> float:
        gray = crop if len(crop.shape) == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        area = float(gray.shape[0] * gray.shape[1])
        return sharpness + (area / 100.0)
    
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
                    
                    # Expand crop slightly so edge characters are not clipped.
                    px1, py1, px2, py2 = self._expand_box(x1, y1, x2, y2, image.shape)
                    plate_crop = image[py1:py2, px1:px2]
                    
                    # Recognize text with TrOCR
                    plate_text, confidence, raw_text, top_candidates, format_score = self.ocr.recognize(plate_crop)
                    job_confidences.append(confidence)

                    # Use global detection index so crops from different images don't overwrite each other
                    detection_idx = len(detections)

                    # Save crops
                    vehicle_crop_path = self.crops_dir / f"{job_id}_vehicle_{detection_idx}.jpg"
                    plate_crop_path = self.crops_dir / f"{job_id}_plate_{detection_idx}.jpg"

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
                            top_ocr_candidates=top_candidates,
                            format_score=float(format_score),
                            source_file_name=image_path.name,
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

    def process_video(self, job_id: str, video_path: Path, progress_callback=None) -> ANPRResult:
        """Process video file"""
        start_time = time.time()
        detections = []
        track_index_by_plate: dict[str, int] = {}
        track_quality: dict[str, float] = {}
        
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
        sample_stride = max(1, int(fps / 2)) if fps > 0 else 1
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if progress_callback and total_frames > 0 and frame_idx % max(1, sample_stride) == 0:
                progress_callback((frame_idx / total_frames) * 100.0, "Scanning video frames")

            # Run detection on sampled frames and keep the sharpest / largest observation.
            if frame_idx % sample_stride == 0:
                results = self.detector.predict(frame, conf=0.25, iou=0.7)
                
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        px1, py1, px2, py2 = self._expand_box(x1, y1, x2, y2, frame.shape)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Extract and recognize plate with TrOCR
                        plate_crop = frame[py1:py2, px1:px2]
                        plate_text, confidence, raw_text, top_candidates, format_score = self.ocr.recognize(plate_crop)
                        job_confidences.append(confidence)
                        frame_second = (frame_idx / fps) if fps > 0 else 0.0
                        observation_quality = (float(confidence) * 100.0) + (float(format_score) * 25.0) + self._crop_quality(plate_crop)

                        # Draw text
                        cv2.putText(frame, plate_text, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        dedup_key = plate_text if len(plate_text) >= 4 else f"frame-{frame_idx}-det-{detection_idx}"
                        if dedup_key in track_index_by_plate:
                            det = detections[track_index_by_plate[dedup_key]]
                            det.seen_count += 1
                            det.last_seen_sec = float(frame_second)
                            if observation_quality > track_quality[dedup_key]:
                                vehicle_crop_path = self.crops_dir / f"{job_id}_vehicle_{track_index_by_plate[dedup_key]}.jpg"
                                plate_crop_path = self.crops_dir / f"{job_id}_plate_{track_index_by_plate[dedup_key]}.jpg"
                                cv2.imwrite(str(vehicle_crop_path), frame[max(0, y1-50):min(height, y2+50), max(0, x1-50):min(width, x2+50)])
                                cv2.imwrite(str(plate_crop_path), plate_crop)
                                det.plate_text = plate_text
                                det.confidence = float(confidence)
                                det.bbox = [x1, y1, x2, y2]
                                det.raw_ocr_text = raw_text
                                det.top_ocr_candidates = top_candidates
                                det.format_score = float(format_score)
                                det.source_frame = frame_idx
                                track_quality[dedup_key] = observation_quality
                        else:
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
                                    top_ocr_candidates=top_candidates,
                                    format_score=float(format_score),
                                    seen_count=1,
                                    first_seen_sec=float(frame_second),
                                    last_seen_sec=float(frame_second),
                                    source_frame=frame_idx,
                                    source_file_name=video_path.name,
                                )
                            )
                            track_index_by_plate[dedup_key] = detection_idx
                            track_quality[dedup_key] = observation_quality
                            detection_idx += 1
            
            # Write frame
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        
        processing_time = time.time() - start_time

        if progress_callback:
            progress_callback(100.0, "Finalizing video result")

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
