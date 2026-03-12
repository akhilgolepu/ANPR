# ANPR System - Setup & Running

## Project Structure

```
/home/akhil/3-2/
├── backend/                 # FastAPI backend server
│   ├── main.py             # FastAPI app & API endpoints
│   ├── processor.py        # Detection & OCR processing logic
│   ├── schemas.py          # Pydantic data models
│   └── requirements.txt     # Backend dependencies
│
├── website/                # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── PipelineDemo.tsx         # Upload & processing UI
│   │   │   ├── ANPRResults.tsx          # Results display
│   │   │   ├── MonitoringDashboard.tsx  # System metrics
│   │   │   └── ...
│   │   └── lib/
│   │       └── anpr-api.ts              # API client
│   └── package.json
│
├── models/weights/
│   └── yolov8s_license_plate_best.pt   # YOLO detection model
│
├── scripts/ocr/
│   └── smart_ocr_pipeline.py           # OCR processing (already tested)
│
└── outputs/
    └── api_uploads/                    # Temporary upload storage
       crops/                           # Plate/vehicle crops (served as static)
       api_results/                     # Result JSON files
```

---

## Setup Instructions

### 1. Install Backend Dependencies

```bash
cd /home/akhil/3-2/backend
conda run -n ml_workspace pip install -r requirements.txt
```

### 2. Install Frontend Dependencies (if not already done)

```bash
cd /home/akhil/3-2/website
npm install
```

---

## Running the Application

### Option A: Run Backend & Frontend in Separate Terminals

**Terminal 1 - Start Backend API:**
```bash
cd /home/akhil/3-2/backend
conda run -n ml_workspace python main.py
```

**Terminal 2 - Start Frontend:**
```bash
cd /home/akhil/3-2/website
npm run dev
```

The frontend will be available at: `http://localhost:5173`

### Option B: Run Both Together

Create a script file `/home/akhil/3-2/start-all.sh`:

```bash
#!/bin/bash
set -e

echo "Starting ANPR System..."

# Start backend in background
echo "Starting Backend API on port 8000..."
cd /home/akhil/3-2/backend
conda run -n ml_workspace python main.py &
BACKEND_PID=$!

# Give backend time to start
sleep 3

# Start frontend
echo "Starting Frontend on port 5173..."
cd /home/akhil/3-2/website
npm run dev

# Kill backend when frontend exits
kill $BACKEND_PID
```

Then run:
```bash
chmod +x /home/akhil/3-2/start-all.sh
bash /home/akhil/3-2/start-all.sh
```

---

## Testing the System

### 1. Check Backend Health
```bash
curl http://localhost:8000/api/health
# Expected response: {"status":"ok","service":"ANPR API"}
```

### 2. Upload Image via API (using curl)
```bash
curl -X POST "http://localhost:8000/api/process-images" \
  -F "files=@/path/to/image.jpg"
```

### 3. Upload Video via API
```bash
curl -X POST "http://localhost:8000/api/process-video" \
  -F "file=@/path/to/video.mp4"
```

### 4. Use Web Interface
- Open `http://localhost:5173`
- Click "Try the Model" button
- Drag & drop or select images/videos
- View results with license plate detections and OCR text

---

## Key Features

✅ **Image Processing**
- Upload single or multiple images
- Real-time plate detection (YOLOv8s, 99.48% accuracy)
- OCR recognition (Smart pipeline, 22% accuracy)
- Interactive results display with crops

✅ **Video Processing**
- Upload video files (MP4, WebM)
- Frame-by-frame processing
- Draws bounding boxes and detected text
- Returns output video with annotations

✅ **Results Export**
- Download results as JSON
- Download results as CSV
- View detection crops (vehicle & plate)
- Download processed video

✅ **System Metrics**
- Real-time processing statistics
- Performance monitoring dashboard
- Pipeline architecture overview

---

## Backend API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/process-images` | Process image(s) |
| POST | `/api/process-video` | Process video |
| GET | `/api/results/{job_id}` | Retrieve results |

---

## Configuration

### Backend Settings (in `main.py`)

- **Upload directory**: `outputs/api_uploads/`
- **Crops directory**: `outputs/api_results/crops/`
- **Results directory**: `outputs/api_results/`
- **CORS**: Enabled for all origins (restrict in production)
- **Port**: 8000

### Frontend Settings (in `lib/anpr-api.ts`)

- **API Base URL**: `http://localhost:8000/api`

---

## Troubleshooting

### Backend won't start
```bash
# Check if port 8000 is already in use
lsof -i :8000

# Kill existing process if needed
kill -9 <PID>

# Try a different port
# Edit main.py: uvicorn.run(app, host="0.0.0.0", port=9000)
```

### Frontend can't connect to backend
- Ensure backend is running on port 8000
- Check CORS is properly configured
- Clear browser cache and reload
- Check browser console for errors

### YOLO model not found
- Ensure model exists at: `models/weights/yolov8s_license_plate_best.pt`
- OR update `processor.py` with correct path to your model

### Out of GPU memory
- Edit `processor.py`: `SmartOCR(gpu=False)` to use CPU
- Or adjust detection `imgsz` parameter (smaller value = faster but less accurate)

---

## Production Deployment

For production, consider:

1. **Restrict CORS** to your frontend domain
2. **Add authentication** to API endpoints
3. **Implement rate limiting**
4. **Use proper error logging**
5. **Docker containerization** for deployment
6. **Load balancing** for multiple requests
7. **Move to Uvicorn production server** (gunicorn with uvicorn workers)

Example Uvicorn production command:
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

---

## Performance Metrics

- **Detection Speed**: ~300ms per image (GPU)
- **OCR Speed**: ~100ms per plate
- **Video Processing**: ~15 FPS with detection & OCR
- **Memory**: ~2-3GB (including models)
- **Detection Accuracy**: 99.48% mAP
- **OCR Accuracy**: 22.0% (Smart pipeline)

---

For issues or questions, check the project logs in `outputs/api_results/` directory.
