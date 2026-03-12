"""
Indian License Plate Format Post-Processing

Applies rule-based corrections based on Indian license plate format:
- Format: [A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}
- Examples: TS09AB1234, AP39CD5678, KA01MN4321

Post-processing steps:
1. Character position-based correction (O/0, I/1, etc.)
2. Format validation and correction
3. Common OCR error fixes
"""

import re
from typing import Optional


# Indian license plate format regex
INDIAN_PLATE_PATTERN = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$')


def correct_character_by_position(text: str, pos: int, expected_type: str) -> str:
    """
    Correct character based on position and expected type.
    
    Args:
        text: Input text
        pos: Character position (0-indexed)
        expected_type: 'letter' or 'digit'
    
    Returns:
        Corrected character
    """
    if pos >= len(text):
        return ''
    
    char = text[pos].upper()
    
    if expected_type == 'digit':
        # In digit positions, common confusions
        corrections = {
            'O': '0',  # O → 0
            'I': '1',  # I → 1
            'S': '5',  # S → 5 (less common)
            'Z': '2',  # Z → 2 (less common)
            'B': '8',  # B → 8 (less common)
        }
    else:  # letter
        # In letter positions, common confusions
        corrections = {
            '0': 'O',  # 0 → O
            '1': 'I',  # 1 → I
            '5': 'S',  # 5 → S
            '2': 'Z',  # 2 → Z
            '8': 'B',  # 8 → B
        }
    
    return corrections.get(char, char)


def apply_position_based_corrections(text: str) -> str:
    """
    Apply position-based character corrections for Indian plate format.
    
    Format: [A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}
    Positions:
    0-1: Letters (state code)
    2-3: Digits (RTO)
    4-5: Letters (1-2 letters)
    6-9: Digits (4 digits)
    
    Args:
        text: Raw OCR output
    
    Returns:
        Corrected text
    """
    if not text or len(text) < 4:
        return text
    
    text = text.upper().strip()
    # Remove non-alphanumeric first
    text = re.sub(r'[^A-Z0-9]', '', text)
    
    if len(text) < 4:
        return text
    
    corrected = list(text)
    
    # Ensure minimum length (pad if needed, but prefer truncation for long strings)
    if len(corrected) > 12:
        # Try to extract valid pattern from longer string
        pattern = re.compile(r'([A-Z]{2})([0-9]{2})([A-Z]{1,2})([0-9]{4})')
        match = pattern.search(text)
        if match:
            return ''.join(match.groups())
        corrected = corrected[:12]  # Truncate to reasonable length
    
    # Position 0-1: State code (letters)
    for i in range(min(2, len(corrected))):
        if corrected[i]:
            corrected[i] = correct_character_by_position(text, i, 'letter')
    
    # Position 2-3: RTO code (digits)
    for i in range(2, min(4, len(corrected))):
        if corrected[i]:
            corrected[i] = correct_character_by_position(text, i, 'digit')
    
    # Position 4-5: Series (letters, 1-2 chars)
    # Find where series ends (next should be digits)
    series_end = min(6, len(corrected))
    for i in range(4, series_end):
        if corrected[i]:
            corrected[i] = correct_character_by_position(text, i, 'letter')
    
    # Position 6-9: Number (digits)
    for i in range(6, min(10, len(corrected))):
        if corrected[i]:
            corrected[i] = correct_character_by_position(text, i, 'digit')
    
    result = ''.join(corrected[:10])  # Max 10 characters
    return result


def fix_common_ocr_errors(text: str) -> str:
    """
    Fix common OCR errors using heuristics.
    
    Args:
        text: Input text
    
    Returns:
        Corrected text
    """
    if not text:
        return text
    
    text = text.upper().strip()
    
    # Remove common OCR artifacts
    text = re.sub(r'[^A-Z0-9]', '', text)  # Remove non-alphanumeric
    
    # Common character confusions (context-aware)
    # These are applied globally but position-based is preferred
    replacements = {
        # Less aggressive - only apply if context suggests it
    }
    
    return text


def validate_and_correct_format(text: str) -> Optional[str]:
    """
    Validate Indian plate format and attempt correction.
    
    Args:
        text: Input text
    
    Returns:
        Corrected text if valid format can be achieved, None otherwise
    """
    if not text:
        return None
    
    text = text.upper().strip()
    text = re.sub(r'[^A-Z0-9]', '', text)  # Remove spaces/special chars
    
    # Check if already matches format
    if INDIAN_PLATE_PATTERN.match(text):
        return text
    
    # Try to extract valid format from longer strings
    # Pattern: 2 letters + 2 digits + 1-2 letters + 4 digits
    pattern = re.compile(r'([A-Z]{2})([0-9]{2})([A-Z]{1,2})([0-9]{4})')
    match = pattern.search(text)
    
    if match:
        state, rto, series, number = match.groups()
        # Ensure series is 1-2 characters
        if len(series) > 2:
            series = series[:2]
        return f"{state}{rto}{series}{number}"
    
    # If text is close to 10 characters, try position-based correction
    if 8 <= len(text) <= 12:
        corrected = apply_position_based_corrections(text)
        if INDIAN_PLATE_PATTERN.match(corrected):
            return corrected
    
    return None


def postprocess_indian_plate(ocr_text: str, strict: bool = False) -> str:
    """
    Post-process OCR output for Indian license plate format.
    
    Steps:
    1. Clean text (remove spaces, special chars)
    2. Apply position-based corrections
    3. Validate format
    4. Return best match
    
    Args:
        ocr_text: Raw OCR output
        strict: If True, only return valid format matches
    
    Returns:
        Post-processed plate text
    """
    if not ocr_text:
        return ""
    
    # Step 1: Clean
    cleaned = re.sub(r'[^A-Z0-9]', '', ocr_text.upper().strip())
    
    if not cleaned:
        return ""
    
    # Step 2: Apply position-based corrections
    corrected = apply_position_based_corrections(cleaned)
    
    # Step 3: Validate format
    validated = validate_and_correct_format(corrected)
    
    if validated:
        return validated
    
    # If strict mode, return empty if no valid format
    if strict:
        return ""
    
    # Otherwise, return corrected version (even if not perfect format)
    return corrected[:10]  # Max 10 characters


def postprocess_batch(ocr_results: list[str], strict: bool = False) -> list[str]:
    """
    Post-process multiple OCR results.
    
    Args:
        ocr_results: List of raw OCR outputs
        strict: If True, only return valid format matches
    
    Returns:
        List of post-processed plate texts
    """
    return [postprocess_indian_plate(text, strict=strict) for text in ocr_results]
