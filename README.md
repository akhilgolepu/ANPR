# ANPR System - Automatic Number Plate Recognition

Intelligent ANPR system for Indian license plates with vehicle detection, plate detection, and OCR.

## Documentation

- **[README_PROCESS.md](README_PROCESS.md)** - Complete workflow guide (detection → cropping → OCR)
- **[README_DATASET.md](README_DATASET.md)** - Dataset preparation and splitting guide
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Detailed project structure

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

### Model 2: License Plate Recognition (CRNN) - TODO
- **Input**: Plate crops (variable width, fixed height)
- **Output**: Plate text (A-Z, 0-9)
- **Loss**: CTC Loss

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

## OCR Performance Metrics

**License Plate Recognition (EasyOCR)**:
- **Exact Match Rate**: 15.87% (40/252 on validation set)
- **Average Character Accuracy**: 51.67%
- **Average OCR Confidence**: 35.86%

See [OCR_METRICS.md](OCR_METRICS.md) for detailed evaluation results and recommendations.

## License

[Your License Here]
