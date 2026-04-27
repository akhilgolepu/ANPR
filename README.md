# ANPR System - Automatic Number Plate Recognition

Intelligent ANPR system for Indian license plates using **YOLOv8** for plate detection and **TrOCR** (`microsoft/trocr-base-printed`) for text recognition.

## Pipeline

1. **YOLOv8s** — detects license plate bounding boxes (99.48% mAP)
2. **Phase-2 preprocessing** — CLAHE contrast enhancement + bilateral denoising
3. **TrOCR** — Transformer-based OCR reads the plate text (~95.5% avg confidence)
4. **Indian plate postprocessing** — state-code anchoring, zone-aware OCR-confusion fixes
5. **FastAPI backend + React frontend** — full web interface with vehicle registry

## Quick Start

### Backend

```bash
pip install -r backend/requirements.txt
cd backend
python -m uvicorn main:app --host localhost --port 8000
```

### Frontend

```bash
cd website
npm install
npm run dev
# Opens at http://localhost:8080
```

### Test Pipeline (test_images/)

```bash
python test_pipeline.py
# Annotated results → outputs/test_results/
```

## Dependencies

| Package                  | Purpose                 |
| ------------------------ | ----------------------- |
| `fastapi` + `uvicorn`    | Backend API server      |
| `ultralytics`            | YOLOv8 plate detection  |
| `transformers` + `torch` | TrOCR recognition       |
| `opencv-python`          | Image/video processing  |
| `Pillow`                 | Image format conversion |
| `python-multipart`       | File upload support     |

Frontend: React + Vite + Tailwind + shadcn/ui

## Project Structure

```
3-2/
├── config/                      # Configuration files
│   ├── dataset.yaml            # Dataset preparation config
│   └── training/                # Training configs
│       └── plate_detection_train.yaml
├── src/                         # Source code
│   ├── dataset/                # Dataset preparation utilities
│   ├── detection/              # Detection models (vehicle, plate)
│   ├── ocr/                    # OCR models (license plate recognition)
│   └── utils/                  # Utility functions
├── scripts/                     # Executable scripts
│   ├── prepare_dataset.py      # Dataset preparation pipeline
│   └── train/                  # Training scripts
│       └── train_plate_detection.py
├── datasets/                    # Data storage
│   ├── raw/                    # Raw datasets (VOC XML, images)
│   └── processed/              # Processed datasets (YOLO format)
│       ├── vehicle_detection/
│       ├── plate_detection/
│       └── ocr_dataset/
├── models/                      # Saved model weights
│   └── weights/
├── runs/                        # Training outputs (logs, checkpoints)
│   └── plate_detection/         # Plate detection training runs
│       └── yolov8s_license_plate/
│           └── weights/
│               └── best.pt
├── outputs/                     # Inference outputs (gitignored)
│   └── plate_crops/             # Extracted plate crops
├── notebooks/                   # Jupyter notebooks for experiments
└── requirements.txt             # Python dependencies
```

## Quick Start

### 1. Setup Environment

```bash
conda activate ml_workspace
pip install -r requirements.txt
```

### 2. Prepare Datasets

```bash
python scripts/prepare_dataset.py
```

This creates:

- **Vehicle Detection**: `datasets/processed/vehicle_detection/` (YOLO format)
- **Plate Detection**: `datasets/processed/plate_detection/` (YOLO format, 80/10/10 split)
- **OCR Dataset**: `datasets/processed/ocr_dataset/` (plate crops + labels)

### 3. Train License Plate Detection Model

```bash
python scripts/train/train_plate_detection.py
```

Output: `runs/plate_detection/yolov8s_license_plate/weights/best.pt`

### 4. Extract Plate Crops

```bash
python scripts/extract_plate_crops.py --source path/to/images --out outputs/crops
```

### 5. Recognize Plate Text (OCR)

```bash
python scripts/recognize_plates.py --source outputs/crops/crops/ --output results.csv
```

See [README_PROCESS.md](README_PROCESS.md) for detailed workflow.

## Models

### Model 1: License Plate Detection (YOLOv8-s)

- **Input**: 640×640 images
- **Output**: Bounding boxes for license plates
- **Classes**: `license_plate` (1 class)
- **Augmentations**: Motion blur, brightness/contrast, Gaussian noise, perspective transform
- **Metrics**: mAP@0.5, Precision, Recall

### Model 2: License Plate Recognition (TrOCR)

- **Engine**: `microsoft/trocr-base-printed` – ViT encoder + autoregressive decoder
- **Preprocessing**: CLAHE contrast enhancement + bilateral denoising (Phase 2 pipeline)
- **Confidence**: Derived from beam-search sequence probability
- **Model caching**: Saved locally to `models/trocr/` after first download
- **Output**: Plate text (A-Z, 0-9), position-corrected for Indian format

### Model 3: Vehicle Detection (Pretrained YOLO)

- Uses COCO-pretrained YOLO
- Classes: car, bus, tempo, vehicle_truck, two_wheelers

## Dataset Statistics

**Plate Detection Dataset** (after adding dataset-3 crops):

- Train: ~3,841 images
- Val: ~480 images
- Test: ~481 images
- Total: ~4,802 images

**Vehicle Detection Dataset**:

- Train: ~62 images
- Val: ~7 images
- Test: ~9 images

## Configuration

- **Dataset config**: `config/dataset.yaml`
- **Training config**: `config/training/plate_detection_train.yaml`

## Dependencies

- **ml_workspace**: PyTorch 2.11, torchvision, numpy, pillow (already installed)
- **Additional**: `pip install -r requirements.txt`

See `requirements.txt` for full list.

## OCR Engine

**License Plate Recognition (TrOCR – `microsoft/trocr-base-printed`)**:

- Transformer-based architecture: ViT image encoder + autoregressive text decoder
- Phase 2 preprocessing: CLAHE contrast enhancement + bilateral denoising
- Confidence derived from beam-search sequence scores (4 beams)
- Model saved locally to `models/trocr/` after first download (no re-download on restart)
- Indian plate format post-processing via position-aware character correction

Previous engine: EasyOCR (22% accuracy) — replaced by TrOCR for higher accuracy on printed text.
