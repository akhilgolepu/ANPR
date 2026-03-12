"""
Pydantic models for ANPR API
"""

from datetime import date, datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, EmailStr, field_validator


class PlateDetection(BaseModel):
    plate_text: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    vehicle_crop_url: str
    plate_crop_url: str
    raw_ocr_text: Optional[str] = None  # Raw TrOCR output before cleaning
    ocr_engine: str = "trocr"


class ANPRResult(BaseModel):
    job_id: str
    status: str  # 'completed' or 'error'
    input_type: str  # 'image' or 'video'
    total_detections: int
    processing_time: float
    detections: List[PlateDetection]
    output_file_url: Optional[str] = None
    error: Optional[str] = None


# ============================================================================
# Vehicle Registry Schemas
# ============================================================================

VehicleStatus = Literal["Clear", "Stolen/Missing", "Recovered"]
VehicleType   = Literal["Car", "Bike", "Truck", "Bus", "Auto", "Other"]


class VehicleRecord(BaseModel):
    """Full vehicle record returned from the registry."""
    plate_number:          str
    vehicle_make:          str
    vehicle_model:         str
    vehicle_year:          Optional[int]            = None
    vehicle_color:         str
    vehicle_type:          VehicleType              = "Car"
    owner_name:            str
    owner_phone:           Optional[str]            = None
    owner_email:           Optional[str]            = None
    owner_address:         Optional[str]            = None
    registered_rto_state:  str
    registered_rto_code:   str
    chassis_number:        Optional[str]            = None
    engine_number:         Optional[str]            = None
    registration_date:     Optional[date]           = None
    registration_expiry:   Optional[date]           = None
    insurance_expiry:      Optional[date]           = None
    status:                VehicleStatus            = "Clear"
    police_complaint_id:   Optional[str]            = None
    missing_date:          Optional[datetime]       = None
    recovery_date:         Optional[datetime]       = None
    created_at:            datetime
    updated_at:            datetime


class VehicleCreate(BaseModel):
    """Payload for adding a new vehicle."""
    plate_number:          str
    vehicle_make:          str
    vehicle_model:         str
    vehicle_year:          Optional[int]            = None
    vehicle_color:         str
    vehicle_type:          VehicleType              = "Car"
    owner_name:            str
    owner_phone:           Optional[str]            = None
    owner_email:           Optional[str]            = None
    owner_address:         Optional[str]            = None
    registered_rto_state:  str
    registered_rto_code:   str
    chassis_number:        Optional[str]            = None
    engine_number:         Optional[str]            = None
    registration_date:     Optional[date]           = None
    registration_expiry:   Optional[date]           = None
    insurance_expiry:      Optional[date]           = None

    @field_validator("plate_number")
    @classmethod
    def normalise_plate(cls, v: str) -> str:
        import re
        return re.sub(r"\s+", "", v).upper()


class VehicleUpdate(BaseModel):
    """Payload for editing an existing vehicle (all fields optional)."""
    vehicle_make:          Optional[str]            = None
    vehicle_model:         Optional[str]            = None
    vehicle_year:          Optional[int]            = None
    vehicle_color:         Optional[str]            = None
    vehicle_type:          Optional[VehicleType]    = None
    owner_name:            Optional[str]            = None
    owner_phone:           Optional[str]            = None
    owner_email:           Optional[str]            = None
    owner_address:         Optional[str]            = None
    registered_rto_state:  Optional[str]            = None
    registered_rto_code:   Optional[str]            = None
    chassis_number:        Optional[str]            = None
    engine_number:         Optional[str]            = None
    registration_date:     Optional[date]           = None
    registration_expiry:   Optional[date]           = None
    insurance_expiry:      Optional[date]           = None


class ComplaintRequest(BaseModel):
    complaint_id:      str
    reported_by:       Optional[str] = None
    reporting_station: Optional[str] = None
    theft_location:    Optional[str] = None


class RecoveryRequest(BaseModel):
    resolution_notes: Optional[str] = None
    officer:          Optional[str] = None


class ActionResponse(BaseModel):
    success: bool
    message: str
    vehicle: Optional[VehicleRecord] = None
