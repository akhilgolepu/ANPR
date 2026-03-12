#!/usr/bin/env python3
"""
Real-time ANPR for images, video files, and webcam.

Supports multiple vehicles/plates per frame.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detection.plate_cropper import find_latest_best_pt
from src.ocr.plate_recognizer import recognize_plate_text


def process_image_anpr(image_path: str, conf: float = 0.5, iou: float = 0.4) -> Dict[str, Any]:
    """
    Process single image and return structured results for backend API.
    
    Args:
        image_path: Path to input image
        conf: Detection confidence threshold  
        iou: IoU threshold for NMS
        
    Returns:
        Dictionary with detections and metadata
    """
    from ultralytics import YOLO
    
    start_time = time.time()
    
    # Load model
    weights_path = _resolve_weights(None)
    model = YOLO(str(weights_path))
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Run detection
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=640,
        verbose=False
    )[0]
    
    detections = []
    
    # Process each detection
    if results.boxes is not None:
        for i, box in enumerate(results.boxes.data):
            x1, y1, x2, y2, det_conf, class_id = box.cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Extract plate region
            plate_crop = image[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue
                
            # Run OCR
            plate_text, ocr_conf = recognize_plate_text(plate_crop)
            
            # Create expanded vehicle crop (larger area around plate)
            h, w = image.shape[:2]
            vehicle_margin = 100
            v_x1 = max(0, x1 - vehicle_margin)
            v_y1 = max(0, y1 - vehicle_margin)
            v_x2 = min(w, x2 + vehicle_margin)
            v_y2 = min(h, y2 + vehicle_margin)
            vehicle_crop = image[v_y1:v_y2, v_x1:v_x2]
            
            detections.append({
                "plate_text": plate_text or "UNKNOWN",
                "confidence": float(ocr_conf) if ocr_conf else 0.0,
                "bbox": [x1, y1, x2, y2],
                "vehicle_image": vehicle_crop,
                "plate_image": plate_crop
            })
    
    processing_time = time.time() - start_time
    
    return {
        "detections": detections,
        "processing_time": processing_time,
        "input_path": image_path
    }


def process_video_anpr(video_path: str, conf: float = 0.5, iou: float = 0.4, 
                       output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Process video file and return structured results for backend API.
    
    Args:
        video_path: Path to input video
        conf: Detection confidence threshold
        iou: IoU threshold for NMS  
        output_path: Optional path for annotated video output
        
    Returns:
        Dictionary with detections and metadata
    """
    from ultralytics import YOLO
    
    start_time = time.time()
    
    # Load model
    weights_path = _resolve_weights(None)
    model = YOLO(str(weights_path))
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output video writer if needed
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    detections = []
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process every 5th frame to speed up (adjust as needed)
            if frame_count % 5 != 0:
                if writer:
                    writer.write(frame)
                continue
                
            # Run detection on frame
            results = model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                imgsz=640,
                verbose=False
            )[0]
            
            frame_detections = []
            
            # Process detections in this frame
            if results.boxes is not None:
                for box in results.boxes.data:
                    x1, y1, x2, y2, det_conf, class_id = box.cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Extract plate region
                    plate_crop = frame[y1:y2, x1:x2]
                    if plate_crop.size == 0:
                        continue
                        
                    # Run OCR
                    plate_text, ocr_conf = recognize_plate_text(plate_crop)
                    
                    # Skip low confidence detections
                    if not plate_text or (ocr_conf and ocr_conf < 0.3):
                        continue
                    
                    # Draw detection on frame
                    _draw_plate(frame, x1, y1, x2, y2, det_conf, plate_text, ocr_conf or 0.0)
                    
                    # Create vehicle crop
                    h, w = frame.shape[:2]
                    vehicle_margin = 80
                    v_x1 = max(0, x1 - vehicle_margin)
                    v_y1 = max(0, y1 - vehicle_margin)
                    v_x2 = min(w, x2 + vehicle_margin)
                    v_y2 = min(h, y2 + vehicle_margin)
                    vehicle_crop = frame[v_y1:v_y2, v_x1:v_x2]
                    
                    frame_detections.append({
                        "plate_text": plate_text,
                        "confidence": float(ocr_conf) if ocr_conf else 0.0,
                        "bbox": [x1, y1, x2, y2],
                        "vehicle_image": vehicle_crop,
                        "plate_image": plate_crop,
                        "frame_number": frame_count,
                        "timestamp": frame_count / fps
                    })
            
            detections.extend(frame_detections)
            
            # Write annotated frame
            if writer:
                writer.write(frame)
                
    finally:
        cap.release()
        if writer:
            writer.release()
    
    processing_time = time.time() - start_time
    
    return {
        "detections": detections,
        "processing_time": processing_time,
        "input_path": video_path,
        "output_video": output_path if output_path else None,
        "total_frames": total_frames,
        "fps": fps
    }


def _draw_plate(frame, x1, y1, x2, y2, det_conf, plate_text, ocr_conf):
    """Draw bounding box and label on frame."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
    label = f"{plate_text or 'UNKNOWN'} | d:{det_conf:.2f} o:{ocr_conf:.2f}"
    y_text = max(20, y1 - 10)
    cv2.putText(frame, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)


def _resolve_weights(user_weights: Path | None) -> Path:
    if user_weights is not None:
        if not user_weights.exists():
            raise FileNotFoundError(f"Weights not found: {user_weights}")
        return user_weights

    candidates = [ROOT / "runs"]
    for root in candidates:
        if root.exists():
            best = find_latest_best_pt(root)
            if best is not None and best.exists():
                return best
    raise FileNotFoundError("No trained best.pt found under anpr_v2/runs or runs")


def _process_frame(model, frame, args):
    result = model.predict(
        source=frame,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        device=args.device,
        verbose=False,
    )[0]

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return frame, 0

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
    detections = 0

    h, w = frame.shape[:2]
    for i, b in enumerate(xyxy):
        x1, y1, x2, y2 = int(max(0, b[0])), int(max(0, b[1])), int(min(w, b[2])), int(min(h, b[3]))
        if x2 <= x1 or y2 <= y1:
            continue

        det_conf = float(confs[i]) if confs is not None else 0.0
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        ocr = recognize_plate_text(crop, engine=args.engine)
        _draw_plate(frame, x1, y1, x2, y2, det_conf, ocr.text, ocr.confidence)
        detections += 1

    return frame, detections


def run_image(model, source: Path, args) -> None:
    frame = cv2.imread(str(source))
    if frame is None:
        raise RuntimeError(f"Could not read image: {source}")

    out, detections = _process_frame(model, frame, args)
    print(f"Detected plates: {detections}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out), out)
        print(f"Saved: {args.out}")

    if not args.no_show:
        cv2.imshow("ANPR", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video(model, source: str, args) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.out), fourcc, fps, (width, height))

    total = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        out, detections = _process_frame(model, frame, args)
        total += detections

        if writer is not None:
            writer.write(out)

        if not args.no_show:
            cv2.imshow("ANPR", out)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Total detected plates: {total}")
    if args.out:
        print(f"Saved: {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time ANPR (multi-vehicle, multi-plate)")
    parser.add_argument("--source", required=True, help="Image path, video path, or webcam index (e.g. 0)")
    parser.add_argument("--weights", type=Path, default=None, help="Plate detection model weights")
    parser.add_argument("--engine", default="easyocr", choices=["easyocr", "paddleocr", "tesseract"])
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=50)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", type=Path, default=None, help="Optional output image/video path")
    parser.add_argument("--no-show", action="store_true", help="Disable live display window")
    args = parser.parse_args()

    from ultralytics import YOLO

    weights = _resolve_weights(args.weights)
    print(f"Using weights: {weights}")
    model = YOLO(str(weights))

    source = args.source
    source_path = Path(source)

    if source_path.exists() and source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        run_image(model, source_path, args)
        return

    if source.isdigit():
        run_video(model, int(source), args)
        return

    run_video(model, source, args)


if __name__ == "__main__":
    main()
