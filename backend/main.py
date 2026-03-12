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

from schemas import ANPRResult, PlateDetection
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
            if not file.content_type.startswith("image/"):
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
        if not file.content_type.startswith("video/"):
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
async def get_results(job_id: str) -> ANPRResult:
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
