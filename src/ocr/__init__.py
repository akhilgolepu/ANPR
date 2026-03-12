# OCR models (license plate recognition)

from .plate_recognizer import (
    OCRResult,
    recognize_plate_text,
    recognize_batch,
    clean_plate_text,
)
from .preprocessing import (
    PlatePreprocessor,
    preprocess_plate_image,
)
from .postprocessing import (
    postprocess_indian_plate,
    postprocess_batch,
    validate_and_correct_format,
)
from .metrics import (
    compute_ocr_metrics,
    compute_char_accuracy,
    normalize_plate_text,
)

__all__ = [
    "OCRResult",
    "recognize_plate_text",
    "recognize_batch",
    "clean_plate_text",
    "PlatePreprocessor",
    "preprocess_plate_image",
    "postprocess_indian_plate",
    "postprocess_batch",
    "validate_and_correct_format",
    "compute_ocr_metrics",
    "compute_char_accuracy",
    "normalize_plate_text",
]
