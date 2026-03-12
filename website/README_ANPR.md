# ANPR Web Application

Complete web interface for Automatic Number Plate Recognition with real-time processing capabilities.

## 🌟 Features

- **Multi-format Support**: Process single images, multiple images, or video files
- **Real-time Processing**: Live progress updates with WebSocket-like polling
- **Modern UI**: Dark theme with glass morphism and orange accent colors
- **GPU Optimized**: Backend leverages your RTX 5060 8GB GPU for fast processing
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Quick Start

### Method 1: Automatic Start (Recommended)
```bash
# From project root
./start-web-app.sh
```

This will automatically:
- Set up Python virtual environment for backend
- Install all dependencies 
- Start backend on http://localhost:8000
- Start frontend on http://localhost:5173

### Method 2: Manual Start

**Backend:**
```bash
cd website/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host localhost --port 8000 --reload
```

**Frontend:**
```bash
cd website
npm install
npm run dev
```

## 📊 Usage

1. **Open the web app**: Navigate to http://localhost:5173
2. **Upload content**: Drag & drop or click to select:
   - Single image (JPG, PNG)
   - Multiple images for batch processing
   - Video file (MP4, AVI)
3. **Process**: Click "Process" to start ANPR analysis
4. **View results**: See detected license plates with vehicle images

## 🎯 Results Display

The results show:
- **Vehicle image (left)**: Cropped image of the detected vehicle
- **License plate text (center)**: OCR-extracted text with confidence score
- **Plate image (right)**: Cropped license plate for verification

## 🛠 API Endpoints

The backend provides RESTful API endpoints:

- `POST /upload/single-image` - Process single image
- `POST /upload/multiple-images` - Process image batch  
- `POST /upload/video` - Process video file
- `GET /status/{job_id}` - Check processing status
- `GET /results/{job_id}` - Get final results
- `GET /health` - Backend health check

## 🔧 Configuration

### Backend Configuration
- **Model path**: `models/weights/yolov8s_license_plate_best.pt`
- **Upload directory**: `website/backend/uploads/`
- **Results directory**: `website/backend/results/`

### Frontend Configuration  
- **API URL**: http://localhost:8000 (configurable in `lib/anpr-api.ts`)
- **File size limit**: 10MB per file
- **Supported formats**: Images (PNG, JPG), Videos (MP4, AVI)

## 🔍 Architecture

```
ANPR Web App
├── Frontend (React + TypeScript + Vite)
│   ├── Upload interface with drag & drop
│   ├── Real-time processing status
│   └── Results visualization
├── Backend (FastAPI + Python)
│   ├── File upload handling
│   ├── ANPR processing pipeline
│   └── RESTful API endpoints
└── ANPR Engine
    ├── YOLOv8 vehicle/plate detection
    ├── EasyOCR text recognition
    └── GPU optimization
```

## 🎨 Design System

- **Primary color**: Orange (#FF6600)
- **Theme**: Dark mode with glass morphism
- **Typography**: Space Grotesk font
- **Components**: shadcn/ui component library

## 📱 Mobile Support

The interface is fully responsive and works on:
- Desktop browsers
- Tablet devices  
- Mobile phones

## 🔧 Development

### Frontend Development
```bash
cd website
npm run dev              # Start dev server
npm run build           # Build for production
npm run preview         # Preview production build
```

### Backend Development
```bash
cd website/backend
source venv/bin/activate
uvicorn main:app --reload --host localhost --port 8000
```

### Adding Features
- **New upload types**: Extend `anpr-api.ts` service
- **UI components**: Add to `components/` directory
- **API endpoints**: Add to `backend/main.py`

## 🤝 Integration

This web interface integrates with your existing ANPR pipeline:
- Uses `scripts/realtime_anpr.py` for processing
- Leverages `src/` modules for detection and OCR
- Maintains compatibility with training scripts

## 📸 Screenshots

The web app displays:
1. Upload zone with drag & drop support
2. Processing status with real-time updates
3. Results with vehicle images and detected plates
4. Clean, modern dark theme interface

---

**Ready to use!** Start the application and begin processing vehicle images through the web interface.