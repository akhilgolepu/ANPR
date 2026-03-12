"""
ANPR FastAPI Backend
Handles image/video uploads, plate detection, and OCR recognition
"""

import csv
import io
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import List

from fastapi import BackgroundTasks, FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response

from schemas import (
    ANPRResult, PlateDetection,
    VehicleRecord, VehicleCreate, VehicleUpdate,
    ComplaintRequest, RecoveryRequest, ActionResponse,
    VehicleMatchSummary, DetectionCorrectionRequest, BulkImportResponse,
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


_jobs: dict[str, dict] = {}


def _normalise_plate(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


def _vehicle_match(record: VehicleRecord) -> VehicleMatchSummary:
    return VehicleMatchSummary(
        plate_number=record.plate_number,
        status=record.status,
        vehicle_make=record.vehicle_make,
        vehicle_model=record.vehicle_model,
        owner_name=record.owner_name,
        registered_rto_state=record.registered_rto_state,
        registered_rto_code=record.registered_rto_code,
        police_complaint_id=record.police_complaint_id,
    )


def _write_result(result: ANPRResult) -> None:
    result_file = RESULTS_DIR / f"{result.job_id}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(mode="json"), f, indent=2)


def _read_result(job_id: str) -> ANPRResult | None:
    result_file = RESULTS_DIR / f"{job_id}.json"
    if not result_file.exists():
        return None
    with open(result_file, encoding="utf-8") as f:
        return _enrich_result(ANPRResult(**json.load(f)))


def _set_job(job_id: str, **updates) -> None:
    job = _jobs.setdefault(job_id, {
        "job_id": job_id,
        "status": "processing",
        "input_type": "video",
        "total_detections": 0,
        "processing_time": 0.0,
        "detections": [],
        "progress": 0.0,
        "stage": "Queued",
        "alert_count": 0,
        "review_count": 0,
        "error": None,
        "output_file_url": None,
    })
    job.update(updates)


def _job_result(job_id: str) -> ANPRResult:
    job = _jobs[job_id]
    return ANPRResult(**job)


def _enrich_result(result: ANPRResult) -> ANPRResult:
    alert_count = 0
    review_count = 0
    for detection in result.detections:
        effective_plate = _normalise_plate(detection.human_corrected_text or detection.plate_text)
        matched = _vehicles.get(effective_plate)
        detection.registry_match = _vehicle_match(matched) if matched else None
        detection.review_required = (
            False if detection.human_verified else (
                detection.confidence < 0.75
                or detection.format_score < 0.70
                or len(effective_plate) < 8
            )
        )
        if detection.review_required:
            review_count += 1
        if matched and matched.status == "Stolen/Missing":
            alert_count += 1
    result.alert_count = alert_count
    result.review_count = review_count
    result.progress = 100.0 if result.status == "completed" else result.progress
    return result


def _filter_vehicles(status: str = "", search: str = "", vehicle_type: str = "", state_code: str = "") -> list[VehicleRecord]:
    results = list(_vehicles.values())
    if status:
        results = [v for v in results if v.status == status]
    if vehicle_type:
        results = [v for v in results if v.vehicle_type == vehicle_type]
    if state_code:
        code = state_code.upper().strip()
        results = [v for v in results if v.registered_rto_code.upper().startswith(code)]
    if search:
        q = search.lower()
        results = [
            v for v in results
            if q in v.plate_number.lower()
            or q in v.owner_name.lower()
            or q in v.registered_rto_state.lower()
            or q in v.registered_rto_code.lower()
            or q in v.vehicle_make.lower()
            or q in v.vehicle_model.lower()
        ]
    return results


def _run_video_job(job_id: str, video_path: Path) -> None:
    try:
        _set_job(job_id, stage="Preparing video", progress=2.0)
        result = processor.process_video(
            job_id,
            video_path,
            progress_callback=lambda progress, stage: _set_job(
                job_id,
                progress=round(min(progress, 99.0), 1),
                stage=stage,
            ),
        )
        result = _enrich_result(result)
        _write_result(result)
        _set_job(
            job_id,
            status="completed",
            stage="Completed",
            progress=100.0,
            total_detections=result.total_detections,
            processing_time=result.processing_time,
            detections=result.model_dump(mode="json")["detections"],
            output_file_url=result.output_file_url,
            alert_count=result.alert_count,
            review_count=result.review_count,
        )
    except Exception as exc:
        error_result = ANPRResult(
            job_id=job_id,
            status="error",
            input_type="video",
            total_detections=0,
            processing_time=0.0,
            detections=[],
            error=str(exc),
            progress=100.0,
            stage="Failed",
        )
        _write_result(error_result)
        _set_job(job_id, status="error", progress=100.0, stage="Failed", error=str(exc))


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
        result = _enrich_result(processor.process_images(job_id, image_paths))
        _write_result(result)
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
async def process_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> ANPRResult:
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
        
        _set_job(job_id, input_type="video", stage="Queued video job", progress=0.0)
        background_tasks.add_task(_run_video_job, job_id, video_path)
        return _job_result(job_id)
    
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
    result = _read_result(job_id)
    if result is not None:
        return result
    if job_id in _jobs:
        return _job_result(job_id)
    raise HTTPException(status_code=404, detail="Job not found")


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
async def list_vehicles(status: str = "", search: str = "", vehicle_type: str = "", state_code: str = ""):
    """List vehicles with optional filters and text search."""
    return _filter_vehicles(status=status, search=search, vehicle_type=vehicle_type, state_code=state_code)


@app.get("/api/vehicles/export")
async def export_vehicles(status: str = "", search: str = "", vehicle_type: str = "", state_code: str = ""):
    rows = _filter_vehicles(status=status, search=search, vehicle_type=vehicle_type, state_code=state_code)
    output = io.StringIO()
    fields = list(VehicleRecord.model_fields.keys())
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(row.model_dump(mode="json"))
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="vehicle-registry.csv"'},
    )


@app.post("/api/vehicles/import", response_model=BulkImportResponse)
async def import_vehicles(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    content = (await file.read()).decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(content))
    imported = 0
    updated = 0
    errors: list[str] = []

    for idx, row in enumerate(reader, start=2):
        try:
            data = {key: (value.strip() if isinstance(value, str) else value) for key, value in row.items()}
            payload = {
                "plate_number": _normalise_plate(data.get("plate_number", "")),
                "vehicle_make": data.get("vehicle_make") or "Unknown",
                "vehicle_model": data.get("vehicle_model") or "Unknown",
                "vehicle_year": int(data["vehicle_year"]) if data.get("vehicle_year") else None,
                "vehicle_color": data.get("vehicle_color") or "Unknown",
                "vehicle_type": data.get("vehicle_type") or "Other",
                "owner_name": data.get("owner_name") or "Unknown",
                "owner_phone": data.get("owner_phone") or None,
                "owner_email": data.get("owner_email") or None,
                "owner_address": data.get("owner_address") or None,
                "registered_rto_state": data.get("registered_rto_state") or "Unknown",
                "registered_rto_code": data.get("registered_rto_code") or "",
                "chassis_number": data.get("chassis_number") or None,
                "engine_number": data.get("engine_number") or None,
                "registration_date": data.get("registration_date") or None,
                "registration_expiry": data.get("registration_expiry") or None,
                "insurance_expiry": data.get("insurance_expiry") or None,
                "status": data.get("status") or "Clear",
                "police_complaint_id": data.get("police_complaint_id") or None,
                "missing_date": data.get("missing_date") or None,
                "recovery_date": data.get("recovery_date") or None,
                "created_at": data.get("created_at") or _now(),
                "updated_at": _now(),
            }
            if not payload["plate_number"]:
                raise ValueError("plate_number is required")
            exists = payload["plate_number"] in _vehicles
            rec = VehicleRecord(**payload)
            _vehicles[rec.plate_number] = rec
            if exists:
                updated += 1
            else:
                imported += 1
        except Exception as exc:
            errors.append(f"Line {idx}: {exc}")

    return BulkImportResponse(success=not errors, imported=imported, updated=updated, errors=errors)


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


@app.put("/api/results/{job_id}/detections/{detection_index}", response_model=ANPRResult)
async def correct_detection(job_id: str, detection_index: int, payload: DetectionCorrectionRequest):
    result = _read_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if detection_index < 0 or detection_index >= len(result.detections):
        raise HTTPException(status_code=404, detail="Detection not found")

    detection = result.detections[detection_index]
    detection.human_corrected_text = payload.corrected_text
    detection.human_verified = True
    result = _enrich_result(result)
    _write_result(result)
    if job_id in _jobs and _jobs[job_id].get("status") == "completed":
        _set_job(
            job_id,
            detections=result.model_dump(mode="json")["detections"],
            alert_count=result.alert_count,
            review_count=result.review_count,
        )
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
