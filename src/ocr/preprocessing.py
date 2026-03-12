"""
License Plate Image Preprocessing Pipeline

This module provides a step-by-step preprocessing pipeline for license plate images
to improve OCR accuracy. Each step is clearly documented and can be enabled/disabled.

Preprocessing Steps:
1. Resize: Ensure minimum size for OCR engines
2. Grayscale Conversion: Convert color to grayscale
3. Contrast Enhancement: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
4. Denoising: Remove noise using Non-local Means Denoising
5. BGR Conversion: Convert back to BGR format for OCR engines
"""

from __future__ import annotations

import cv2
import numpy as np


class PlatePreprocessor:
    """
    Step-by-step preprocessing pipeline for license plate images.
    
    Each preprocessing step is clearly documented and can be toggled.
    """
    
    def __init__(
        self,
        min_size: int = 80,  # Increased from 64 for better OCR
        target_height: int = 64,  # Fixed height for OCR
        enable_resize: bool = True,
        enable_grayscale: bool = True,
        enable_contrast: bool = True,
        enable_denoise: bool = True,
        maintain_aspect: bool = True,  # Maintain aspect ratio
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: tuple[int, int] = (8, 8),
        denoise_h: float = 7.0,
    ):
        """
        Initialize preprocessing pipeline.
        
        Args:
            min_size: Minimum image dimension (width or height) after resize
            enable_resize: Enable resizing small images
            enable_grayscale: Enable grayscale conversion
            enable_contrast: Enable CLAHE contrast enhancement
            enable_denoise: Enable denoising
            clahe_clip_limit: CLAHE clip limit (higher = more contrast)
            clahe_tile_size: CLAHE tile grid size
            denoise_h: Denoising strength (higher = more aggressive)
        """
        self.min_size = min_size
        self.target_height = target_height
        self.enable_resize = enable_resize
        self.enable_grayscale = enable_grayscale
        self.enable_contrast = enable_contrast
        self.enable_denoise = enable_denoise
        self.maintain_aspect = maintain_aspect
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.denoise_h = denoise_h
        
        # Initialize CLAHE once (reusable)
        if self.enable_contrast:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_size
            )
        else:
            self.clahe = None
    
    def preprocess(self, img: np.ndarray, verbose: bool = False) -> tuple[np.ndarray, dict]:
        """
        Apply preprocessing pipeline to image.
        
        Args:
            img: Input image (BGR format)
            verbose: Print each preprocessing step
        
        Returns:
            (processed_image, step_info_dict)
        """
        steps_applied = []
        current = img.copy()
        
        # Step 1: Resize with aspect ratio preservation
        if self.enable_resize:
            h, w = current.shape[:2]
            original_h, original_w = h, w
            
            # Ensure minimum size
            if min(h, w) < self.min_size:
                scale = self.min_size / min(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                current = cv2.resize(current, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                h, w = new_h, new_w
                steps_applied.append(f"Resized (min): {original_h}x{original_w} -> {h}x{w}")
                if verbose:
                    print(f"  Step 1: Resized from {original_h}x{original_w} to {h}x{w}")
            
            # Resize to target height while maintaining aspect ratio
            if self.maintain_aspect and h != self.target_height:
                aspect_ratio = w / h
                new_h = self.target_height
                new_w = int(new_h * aspect_ratio)
                current = cv2.resize(current, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                steps_applied.append(f"Resized (target height): {h}x{w} -> {new_h}x{new_w}")
                if verbose:
                    print(f"  Step 1b: Resized to target height {new_h}px (aspect ratio preserved)")
        
        # Step 2: Convert to grayscale
        if self.enable_grayscale:
            if len(current.shape) == 3:
                current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
                steps_applied.append("Grayscale conversion")
                if verbose:
                    print(f"  Step 2: Converted to grayscale")
        
        # Step 3: Contrast enhancement (CLAHE)
        if self.enable_contrast and self.clahe is not None:
            current = self.clahe.apply(current)
            steps_applied.append(f"CLAHE contrast (clip={self.clahe_clip_limit})")
            if verbose:
                print(f"  Step 3: Applied CLAHE contrast enhancement")
        
        # Step 4: Denoising
        if self.enable_denoise:
            current = cv2.fastNlMeansDenoising(current, h=self.denoise_h)
            steps_applied.append(f"Denoising (h={self.denoise_h})")
            if verbose:
                print(f"  Step 4: Applied denoising")
        
        # Step 5: Convert back to BGR (OCR engines expect BGR)
        if len(current.shape) == 2:
            current = cv2.cvtColor(current, cv2.COLOR_GRAY2BGR)
            steps_applied.append("BGR conversion")
            if verbose:
                print(f"  Step 5: Converted to BGR format")
        
        info = {
            "steps": steps_applied,
            "final_shape": current.shape,
            "original_shape": img.shape,
        }
        
        return current, info


def preprocess_plate_image(
    img: np.ndarray,
    use_preprocessing: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """
    Preprocess license plate image for OCR (simplified interface).
    
    This is a convenience function that uses PlatePreprocessor with default settings.
    For more control, use PlatePreprocessor directly.
    
    Args:
        img: Input image (BGR format)
        use_preprocessing: If False, only resize (no enhancement)
        verbose: Print preprocessing steps
    
    Returns:
        Preprocessed image (BGR format)
    """
    if not use_preprocessing:
        # Only resize if needed
        h, w = img.shape[:2]
        min_size = 64
        if min(h, w) < min_size:
            scale = min_size / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return img
    
    preprocessor = PlatePreprocessor()
    processed, info = preprocessor.preprocess(img, verbose=verbose)
    return processed
