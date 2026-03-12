"""
Pydantic models for ANPR API
"""

from typing import List, Optional
from pydantic import BaseModel


class PlateDetection(BaseModel):
    plate_text: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    vehicle_crop_url: str
    plate_crop_url: str


class ANPRResult(BaseModel):
    job_id: str
    status: str  # 'completed' or 'error'
    input_type: str  # 'image' or 'video'
    total_detections: int
    processing_time: float
    detections: List[PlateDetection]
    output_file_url: Optional[str] = None
    error: Optional[str] = None
