"""
License Plate OCR Recognition Module

Supports multiple OCR engines:
- EasyOCR (default, best accuracy)
- PaddleOCR (alternative)
- Tesseract (fallback)

Usage:
    from src.ocr.plate_recognizer import recognize_plate_text
    
    text, conf = recognize_plate_text("path/to/plate.jpg", engine="easyocr")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None

from .postprocessing import postprocess_indian_plate


_EASYOCR_READER = None
_PADDLE_OCR = None


def _gpu_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def _get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr

        _EASYOCR_READER = easyocr.Reader(['en'], gpu=_gpu_available(), verbose=False)
    return _EASYOCR_READER


def _get_paddle_ocr():
    global _PADDLE_OCR
    if _PADDLE_OCR is None:
        from paddleocr import PaddleOCR

        _PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=_gpu_available(), show_log=False)
    return _PADDLE_OCR


@dataclass
class OCRResult:
    """OCR recognition result."""
    text: str
    confidence: float
    engine: str
    raw_text: str | None = None  # Original OCR output before cleaning


def clean_plate_text(text: str) -> str:
    """
    Clean and normalize license plate text (basic cleaning only).
    
    For Indian plate format post-processing, use postprocess_indian_plate() instead.
    """
    if not text:
        return ""
    
    # Remove common separators and special chars
    cleaned = re.sub(r'[^A-Za-z0-9]', '', text.upper())
    return cleaned


# Import preprocessing from dedicated module
from .preprocessing import preprocess_plate_image


def recognize_with_easyocr(image_path: Path | str | np.ndarray) -> OCRResult:
    """
    Recognize plate text using EasyOCR.
    
    Args:
        image_path: Path to image or numpy array (BGR)
    
    Returns:
        OCRResult with text and confidence
    """
    try:
        import easyocr
    except ImportError:
        raise ImportError(
            "EasyOCR not installed. Install with: pip install easyocr\n"
            "Note: EasyOCR downloads models on first use (~100MB)"
        )
    
    reader = _get_easyocr_reader()
    
    # Load image if path provided
    if isinstance(image_path, (str, Path)):
        img = cv2.imread(str(image_path))
        if img is None:
            return OCRResult(text="", confidence=0.0, engine="easyocr", raw_text="")
    else:
        img = image_path.copy()
    
    # Try with preprocessing first, fallback to original if needed
    results = []
    try:
        # Try with preprocessing
        img_processed = preprocess_plate_image(img, use_preprocessing=True)
        results = reader.readtext(img_processed)
        # If preprocessing gives low confidence or no results, try without preprocessing
        if not results or (results and results[0][2] < 0.3):
            results_orig = reader.readtext(img)
            # Use original if it has better confidence
            if results_orig and (not results or results_orig[0][2] > results[0][2]):
                results = results_orig
    except Exception:
        # Fallback to original image if preprocessing fails
        results = reader.readtext(img)
    
    if not results:
        return OCRResult(text="", confidence=0.0, engine="easyocr", raw_text="")
    
    # Combine all detected text (plates may have multiple lines)
    texts = []
    confidences = []
    for (bbox, text, conf) in results:
        texts.append(text.strip())
        confidences.append(conf)
    
    combined_text = " ".join(texts)
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    
    # Basic cleaning
    cleaned = clean_plate_text(combined_text)
    
    # Apply Indian plate format post-processing
    postprocessed = postprocess_indian_plate(cleaned, strict=False)
    
    return OCRResult(
        text=postprocessed,
        confidence=avg_conf,
        engine="easyocr",
        raw_text=combined_text,
    )


def recognize_with_paddleocr(image_path: Path | str | np.ndarray) -> OCRResult:
    """
    Recognize plate text using PaddleOCR.
    
    Args:
        image_path: Path to image or numpy array (BGR)
    
    Returns:
        OCRResult with text and confidence
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise ImportError(
            "PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr"
        )
    
    ocr = _get_paddle_ocr()
    
    # Load image if path provided
    if isinstance(image_path, (str, Path)):
        img_path = str(image_path)
    else:
        # Save temp image for PaddleOCR
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, image_path)
            img_path = tmp.name
    
    # Run OCR
    results = ocr.ocr(img_path, cls=True)
    
    if not results or not results[0]:
        return OCRResult(text="", confidence=0.0, engine="paddleocr", raw_text="")
    
    # Extract text and confidence
    texts = []
    confidences = []
    for line in results[0]:
        if line and len(line) >= 2:
            text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
            conf = float(line[1][1]) if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 0.5
            texts.append(text.strip())
            confidences.append(conf)
    
    combined_text = " ".join(texts)
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    
    cleaned = clean_plate_text(combined_text)
    
    return OCRResult(
        text=cleaned,
        confidence=avg_conf,
        engine="paddleocr",
        raw_text=combined_text,
    )


def recognize_with_tesseract(image_path: Path | str | np.ndarray) -> OCRResult:
    """
    Recognize plate text using Tesseract OCR (pytesseract).
    
    Args:
        image_path: Path to image or numpy array (BGR)
    
    Returns:
        OCRResult with text and confidence
    """
    try:
        import pytesseract
    except ImportError:
        raise ImportError(
            "pytesseract not installed. Install with: pip install pytesseract\n"
            "Also install Tesseract OCR: sudo apt-get install tesseract-ocr (Linux)"
        )
    
    # Load image if path provided
    if isinstance(image_path, (str, Path)):
        img = cv2.imread(str(image_path))
        if img is None:
            return OCRResult(text="", confidence=0.0, engine="tesseract", raw_text="")
    else:
        img = image_path.copy()
    
    # Convert BGR to RGB for Tesseract
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Preprocess: grayscale, threshold
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Tesseract config: alphanumeric, single line
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    # Run OCR
    data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)
    
    texts = []
    confidences = []
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        text = data['text'][i].strip()
        conf = float(data['conf'][i]) if data['conf'][i] != -1 else 0.0
        if text:
            texts.append(text)
            confidences.append(conf)
    
    combined_text = " ".join(texts)
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    
    cleaned = clean_plate_text(combined_text)
    
    return OCRResult(
        text=cleaned,
        confidence=avg_conf / 100.0,  # Tesseract returns 0-100, normalize to 0-1
        engine="tesseract",
        raw_text=combined_text,
    )


def recognize_plate_text(
    image_path: Path | str | np.ndarray,
    engine: Literal["easyocr", "paddleocr", "tesseract"] = "easyocr",
) -> OCRResult:
    """
    Recognize license plate text from an image.
    
    Args:
        image_path: Path to plate image or numpy array (BGR format)
        engine: OCR engine to use ("easyocr", "paddleocr", or "tesseract")
    
    Returns:
        OCRResult with recognized text, confidence, and engine used
    
    Example:
        >>> result = recognize_plate_text("plate.jpg", engine="easyocr")
        >>> print(f"Plate: {result.text}, Confidence: {result.confidence:.2f}")
    """
    if engine == "easyocr":
        return recognize_with_easyocr(image_path)
    elif engine == "paddleocr":
        return recognize_with_paddleocr(image_path)
    elif engine == "tesseract":
        return recognize_with_tesseract(image_path)
    else:
        raise ValueError(f"Unknown engine: {engine}. Choose from: easyocr, paddleocr, tesseract")


def recognize_batch(
    image_paths: list[Path | str],
    engine: Literal["easyocr", "paddleocr", "tesseract"] = "easyocr",
    verbose: bool = True,
) -> list[OCRResult]:
    """
    Recognize text from multiple plate images.
    
    Args:
        image_paths: List of image paths
        engine: OCR engine to use
        verbose: Print progress
    
    Returns:
        List of OCRResult objects
    """
    results = []
    total = len(image_paths)
    
    for i, img_path in enumerate(image_paths):
        if verbose and (i + 1) % 10 == 0:
            print(f"Processing {i+1}/{total}...")
        
        try:
            result = recognize_plate_text(img_path, engine=engine)
            results.append(result)
        except Exception as e:
            if verbose:
                print(f"Error processing {img_path}: {e}")
            results.append(OCRResult(text="", confidence=0.0, engine=engine, raw_text=None))
    
    return results
