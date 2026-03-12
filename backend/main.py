"""
ANPR FastAPI Backend
Handles image/video uploads, plate detection, and OCR recognition
"""

import os
import sys
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from schemas import (
    ANPRResult, PlateDetection,
    VehicleRecord, VehicleCreate, VehicleUpdate,
    ComplaintRequest, RecoveryRequest, ActionResponse,
)
from processor import ProcessorService

# Get absolute paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Add parent to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

# Initialize FastAPI app
app = FastAPI(
    title="ANPR API",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOADS_DIR = PROJECT_ROOT / "outputs" / "api_uploads"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "api_results"
CROPS_DIR = RESULTS_DIR / "crops"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files for serving crops
app.mount("/static", StaticFiles(directory=str(CROPS_DIR)), name="static")

# Mount static files for serving video outputs
app.mount("/videos", StaticFiles(directory=str(RESULTS_DIR)), name="videos")

# Initialize processor service
try:
    processor = ProcessorService(
        model_root=PROJECT_ROOT,
        crops_dir=CROPS_DIR,
        results_dir=RESULTS_DIR,
    )
    print("✓ ProcessorService initialized successfully")
except Exception as e:
    print(f"✗ Error initializing ProcessorService: {e}")
    processor = None


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok" if processor else "initializing",
        "service": "ANPR API",
        "models": "loaded" if processor else "loading",
    }


@app.post("/api/process-images")
async def process_images(files: List[UploadFile] = File(...)) -> ANPRResult:
    """
    Process one or multiple images for ANPR
    
    Returns:
        ANPRResult with detected plates and recognized text
    """
    if not processor:
        return ANPRResult(
            job_id="",
            status="error",
            input_type="image",
            total_detections=0,
            processing_time=0,
            detections=[],
            error="Backend processor not initialized. Try again in a moment.",
        )
    
    job_id = str(uuid.uuid4())
    
    try:
        # Save uploaded files
        image_paths = []
        for file in files:
            ct = file.content_type or ""
            ext = (file.filename or "").lower().rsplit(".", 1)[-1]
            is_image = ct.startswith("image/") or ext in {"jpg", "jpeg", "png", "bmp", "webp"}
            if not is_image:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is not an image"
                )
            
            file_path = UPLOADS_DIR / f"{job_id}_{file.filename}"
            contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(contents)
            image_paths.append(file_path)
        
        # Process images
        result = processor.process_images(job_id, image_paths)
        return result
    
    except Exception as e:
        return ANPRResult(
            job_id=job_id,
            status="error",
            input_type="image",
            total_detections=0,
            processing_time=0,
            detections=[],
            error=str(e)
        )


@app.post("/api/process-video")
async def process_video(file: UploadFile = File(...)) -> ANPRResult:
    """
    Process a video file for ANPR
    
    Extracts frames and processes plates in each frame.
    Returns video file with bounding boxes drawn.
    
    Returns:
        ANPRResult with detected plates and path to output video
    """
    if not processor:
        return ANPRResult(
            job_id="",
            status="error",
            input_type="video",
            total_detections=0,
            processing_time=0,
            detections=[],
            error="Backend processor not initialized. Try again in a moment.",
        )
    
    job_id = str(uuid.uuid4())
    
    try:
        ct = file.content_type or ""
        ext = (file.filename or "").lower().rsplit(".", 1)[-1]
        is_video = ct.startswith("video/") or ext in {"mp4", "webm", "avi", "mov", "mkv"}
        if not is_video:
            raise HTTPException(
                status_code=400,
                detail="File is not a video"
            )
        
        # Save video file
        video_path = UPLOADS_DIR / f"{job_id}_{file.filename}"
        contents = await file.read()
        with open(video_path, "wb") as f:
            f.write(contents)
        
        # Process video
        result = processor.process_video(job_id, video_path)
        return result
    
    except Exception as e:
        return ANPRResult(
            job_id=job_id,
            status="error",
            input_type="video",
            total_detections=0,
            processing_time=0,
            detections=[],
            error=str(e)
        )


@app.get("/api/results/{job_id}")
async def get_results(job_id: str) -> ANPRResult:  # noqa: F811
    """
    Retrieve results for a completed job
    """
    result_file = RESULTS_DIR / f"{job_id}.json"
    
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    
    import json
    with open(result_file) as f:
        result_data = json.load(f)
    
    return ANPRResult(**result_data)


# ============================================================================
# Vehicle Registry — In-Memory Store (seeded with demo data)
# ============================================================================

from datetime import datetime as _dt, date as _date  # noqa: E402

def _now() -> _dt:
    return _dt.utcnow()

_vehicles: dict[str, VehicleRecord] = {}


def _seed_vehicles() -> None:
    rows = [
        # Clear vehicles
        dict(plate_number="TS32T2514", vehicle_make="Maruti Suzuki", vehicle_model="Swift",
             vehicle_year=2019, vehicle_color="White", vehicle_type="Car",
             owner_name="Ravi Kumar", owner_phone="9849123456",
             owner_email="ravi.kumar@email.com", owner_address="Flat 3B, Hitech City, Hyderabad",
             registered_rto_state="Telangana", registered_rto_code="TS32",
             chassis_number="MA3FJEB1S00123456", engine_number="G10B1234567",
             registration_date=_date(2019, 6, 15), registration_expiry=_date(2034, 6, 14),
             insurance_expiry=_date(2025, 6, 14), status="Clear"),
        dict(plate_number="AP28AL4708", vehicle_make="Hyundai", vehicle_model="i20",
             vehicle_year=2021, vehicle_color="Silver", vehicle_type="Car",
             owner_name="Priya Reddy", owner_phone="9550234567",
             owner_email="priya.reddy@email.com", owner_address="House 15, Gachibowli, Hyderabad",
             registered_rto_state="Andhra Pradesh", registered_rto_code="AP28",
             chassis_number="MALA851CBLM012345", engine_number="G4FA567890",
             registration_date=_date(2021, 3, 10), registration_expiry=_date(2036, 3, 9),
             insurance_expiry=_date(2025, 3, 9), status="Clear"),
        dict(plate_number="MH04CE8821", vehicle_make="Honda", vehicle_model="Activa 6G",
             vehicle_year=2022, vehicle_color="Blue", vehicle_type="Bike",
             owner_name="Amit Shah", owner_phone="9820345678",
             registered_rto_state="Maharashtra", registered_rto_code="MH04",
             registration_date=_date(2022, 1, 20), registration_expiry=_date(2037, 1, 19),
             insurance_expiry=_date(2025, 1, 19), status="Clear"),
        dict(plate_number="KA05MN3301", vehicle_make="Tata", vehicle_model="Nexon",
             vehicle_year=2023, vehicle_color="Red", vehicle_type="Car",
             owner_name="Suresh Nair", owner_phone="9880456789",
             owner_email="suresh.nair@email.com", owner_address="12 MG Road, Bangalore",
             registered_rto_state="Karnataka", registered_rto_code="KA05",
             registration_date=_date(2023, 8, 5), registration_expiry=_date(2038, 8, 4),
             insurance_expiry=_date(2025, 8, 4), status="Clear"),
        dict(plate_number="DL09CAB5521", vehicle_make="Toyota", vehicle_model="Innova Crysta",
             vehicle_year=2020, vehicle_color="Black", vehicle_type="Car",
             owner_name="Meera Sharma", owner_phone="9711567890",
             owner_email="meera.sharma@email.com", owner_address="45 Dwarka, New Delhi",
             registered_rto_state="Delhi", registered_rto_code="DL09",
             registration_date=_date(2020, 11, 12), registration_expiry=_date(2035, 11, 11),
             insurance_expiry=_date(2025, 11, 11), status="Clear"),
        # Stolen vehicles
        dict(plate_number="TN22CK1193", vehicle_make="Bajaj", vehicle_model="Pulsar 150",
             vehicle_year=2018, vehicle_color="Black", vehicle_type="Bike",
             owner_name="Arjun Krishnamurthy", owner_phone="9444678901",
             registered_rto_state="Tamil Nadu", registered_rto_code="TN22",
             registration_date=_date(2018, 4, 22), registration_expiry=_date(2033, 4, 21),
             insurance_expiry=_date(2024, 4, 21), status="Stolen/Missing",
             police_complaint_id="FIR/2024/TN22/001",
             missing_date=_dt(2024, 7, 15, 14, 30)),
        dict(plate_number="RJ14GH7742", vehicle_make="Maruti Suzuki", vehicle_model="Alto 800",
             vehicle_year=2017, vehicle_color="White", vehicle_type="Car",
             owner_name="Sunita Verma", owner_phone="9414789012",
             registered_rto_state="Rajasthan", registered_rto_code="RJ14",
             registration_date=_date(2017, 9, 3), registration_expiry=_date(2032, 9, 2),
             insurance_expiry=_date(2024, 9, 2), status="Stolen/Missing",
             police_complaint_id="FIR/2024/RJ14/015",
             missing_date=_dt(2024, 10, 2, 19, 45)),
        dict(plate_number="UP16BT4490", vehicle_make="Mahindra", vehicle_model="Scorpio",
             vehicle_year=2019, vehicle_color="Brown", vehicle_type="Car",
             owner_name="Rahul Mishra", owner_phone="9935890123",
             registered_rto_state="Uttar Pradesh", registered_rto_code="UP16",
             registration_date=_date(2019, 12, 18), registration_expiry=_date(2034, 12, 17),
             insurance_expiry=_date(2024, 12, 17), status="Stolen/Missing",
             police_complaint_id="FIR/2024/UP16/032",
             missing_date=_dt(2024, 11, 8, 8, 0)),
        # Recovered vehicles
        dict(plate_number="GJ01BC2288", vehicle_make="Ford", vehicle_model="EcoSport",
             vehicle_year=2016, vehicle_color="Grey", vehicle_type="Car",
             owner_name="Kiran Patel", owner_phone="9898901234",
             registered_rto_state="Gujarat", registered_rto_code="GJ01",
             registration_date=_date(2016, 7, 30), registration_expiry=_date(2031, 7, 29),
             insurance_expiry=_date(2024, 7, 29), status="Recovered",
             police_complaint_id="FIR/2023/GJ01/008",
             missing_date=_dt(2023, 5, 10, 22, 0),
             recovery_date=_dt(2023, 7, 25, 11, 30)),
        dict(plate_number="PB10HD9934", vehicle_make="Hero", vehicle_model="Splendor Plus",
             vehicle_year=2020, vehicle_color="Red", vehicle_type="Bike",
             owner_name="Gurpreet Singh", owner_phone="9815012345",
             registered_rto_state="Punjab", registered_rto_code="PB10",
             registration_date=_date(2020, 2, 14), registration_expiry=_date(2035, 2, 13),
             insurance_expiry=_date(2025, 2, 13), status="Recovered",
             police_complaint_id="FIR/2024/PB10/005",
             missing_date=_dt(2024, 3, 1, 6, 0),
             recovery_date=_dt(2024, 4, 18, 15, 0)),
    ]
    now = _now()
    for r in rows:
        r.setdefault("owner_phone", None)
        r.setdefault("owner_email", None)
        r.setdefault("owner_address", None)
        r.setdefault("chassis_number", None)
        r.setdefault("engine_number", None)
        r.setdefault("registration_date", None)
        r.setdefault("registration_expiry", None)
        r.setdefault("insurance_expiry", None)
        r.setdefault("police_complaint_id", None)
        r.setdefault("missing_date", None)
        r.setdefault("recovery_date", None)
        r.setdefault("vehicle_year", None)
        rec = VehicleRecord(**r, created_at=now, updated_at=now)
        _vehicles[rec.plate_number] = rec


_seed_vehicles()


# ============================================================================
# Vehicle Registry Endpoints
# ============================================================================

@app.get("/api/vehicles", response_model=list[VehicleRecord])
async def list_vehicles(status: str = "", search: str = ""):
    """List vehicles with optional status filter and text search."""
    results = list(_vehicles.values())
    if status:
        results = [v for v in results if v.status == status]
    if search:
        q = search.lower()
        results = [
            v for v in results
            if q in v.plate_number.lower()
            or q in v.owner_name.lower()
            or q in v.registered_rto_state.lower()
            or q in v.vehicle_make.lower()
            or q in v.vehicle_model.lower()
        ]
    return results


@app.post("/api/vehicles", response_model=ActionResponse, status_code=201)
async def create_vehicle(payload: VehicleCreate):
    """Add a new vehicle to the registry."""
    plate = payload.plate_number
    if plate in _vehicles:
        raise HTTPException(status_code=409, detail=f"Vehicle {plate} already exists")
    now = _now()
    rec = VehicleRecord(
        **payload.model_dump(),
        status="Clear",
        created_at=now,
        updated_at=now,
    )
    _vehicles[plate] = rec
    return ActionResponse(success=True, message=f"Vehicle {plate} added successfully", vehicle=rec)


@app.get("/api/vehicles/{plate}", response_model=VehicleRecord)
async def get_vehicle(plate: str):
    """Get a single vehicle by plate number."""
    plate = plate.upper()
    if plate not in _vehicles:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return _vehicles[plate]


@app.put("/api/vehicles/{plate}", response_model=ActionResponse)
async def update_vehicle(plate: str, payload: VehicleUpdate):
    """Update fields of an existing vehicle."""
    plate = plate.upper()
    if plate not in _vehicles:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    existing = _vehicles[plate]
    updated_data = existing.model_dump()
    for field, value in payload.model_dump(exclude_none=True).items():
        updated_data[field] = value
    updated_data["updated_at"] = _now()
    rec = VehicleRecord(**updated_data)
    _vehicles[plate] = rec
    return ActionResponse(success=True, message=f"Vehicle {plate} updated", vehicle=rec)


@app.delete("/api/vehicles/{plate}", response_model=ActionResponse)
async def delete_vehicle(plate: str):
    """Remove a vehicle from the registry."""
    plate = plate.upper()
    if plate not in _vehicles:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    rec = _vehicles.pop(plate)
    return ActionResponse(success=True, message=f"Vehicle {plate} deleted", vehicle=rec)


@app.post("/api/vehicles/{plate}/file-complaint", response_model=ActionResponse)
async def file_complaint(plate: str, payload: ComplaintRequest):
    """Mark a vehicle as Stolen/Missing and attach FIR details."""
    plate = plate.upper()
    if plate not in _vehicles:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    existing = _vehicles[plate]
    now = _now()
    updated_data = existing.model_dump()
    updated_data.update(
        status="Stolen/Missing",
        police_complaint_id=payload.complaint_id,
        missing_date=now,
        updated_at=now,
    )
    rec = VehicleRecord(**updated_data)
    _vehicles[plate] = rec
    return ActionResponse(
        success=True,
        message=f"Police complaint {payload.complaint_id} filed for vehicle {plate}",
        vehicle=rec,
    )


@app.post("/api/vehicles/{plate}/mark-recovered", response_model=ActionResponse)
async def mark_recovered(plate: str, payload: RecoveryRequest):
    """Mark a stolen vehicle as Recovered."""
    plate = plate.upper()
    if plate not in _vehicles:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    existing = _vehicles[plate]
    now = _now()
    updated_data = existing.model_dump()
    updated_data.update(
        status="Recovered",
        recovery_date=now,
        updated_at=now,
    )
    rec = VehicleRecord(**updated_data)
    _vehicles[plate] = rec
    return ActionResponse(
        success=True,
        message=f"Vehicle {plate} marked as recovered",
        vehicle=rec,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
