from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2


@dataclass(frozen=True)
class PlateCrop:
    source_path: str
    crop_path: str
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int
    frame: int | None = None


def find_latest_best_pt(search_root: Path) -> Path | None:
    """
    Return most recently modified `best.pt` under search_root.
    
    Searches in order:
    1. runs/plate_detection/*/weights/best.pt (new structure)
    2. runs/detect/runs/plate_detection/*/weights/best.pt (legacy nested structure)
    3. Any best.pt recursively
    """
    # Try new structure first
    plate_detection_dir = search_root / "plate_detection"
    if plate_detection_dir.exists():
        best_files = list(plate_detection_dir.rglob("weights/best.pt"))
        if best_files:
            best_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return best_files[0]
    
    # Try legacy nested structure
    legacy_dir = search_root / "detect" / "runs" / "plate_detection"
    if legacy_dir.exists():
        best_files = list(legacy_dir.rglob("weights/best.pt"))
        if best_files:
            best_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return best_files[0]
    
    # Fallback: search all
    best_files = list(search_root.rglob("best.pt"))
    if not best_files:
        return None
    best_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return best_files[0]


def _clamp_xyxy(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[int, int, int, int] | None:
    x1i = max(0, min(w - 1, int(x1)))
    y1i = max(0, min(h - 1, int(y1)))
    x2i = max(0, min(w, int(x2)))
    y2i = max(0, min(h, int(y2)))
    if x2i <= x1i or y2i <= y1i:
        return None
    return x1i, y1i, x2i, y2i


def extract_plate_crops(
    *,
    weights: Path,
    source: Path,
    out_dir: Path,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str = "0",
    max_det: int = 50,
) -> list[PlateCrop]:
    """
    Run YOLO plate detector and save cropped license plates.

    Saves:
      - `out_dir/crops/*.jpg`
      - `out_dir/crops.csv` (metadata)
    """
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))

    records: list[PlateCrop] = []

    # stream=True yields results progressively for dirs/videos
    results = model.predict(
        source=str(source),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        max_det=max_det,
        stream=True,
        verbose=False,
    )

    for r in results:
        orig = r.orig_img  # numpy array (BGR)
        if orig is None:
            continue
        h, w = orig.shape[:2]

        boxes = getattr(r, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None

        src_path = str(getattr(r, "path", source))
        frame = getattr(r, "frame", None)

        src_stem = Path(src_path).stem
        frame_part = f"_f{int(frame):06d}" if frame is not None else ""

        for i, b in enumerate(xyxy):
            c = float(confs[i]) if confs is not None else 0.0
            clamped = _clamp_xyxy(b[0], b[1], b[2], b[3], w=w, h=h)
            if clamped is None:
                continue
            x1, y1, x2, y2 = clamped
            crop = orig[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_name = f"{src_stem}{frame_part}_plate_{i:02d}_conf_{c:.2f}.jpg"
            crop_path = crops_dir / crop_name
            cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            records.append(
                PlateCrop(
                    source_path=src_path,
                    crop_path=str(crop_path),
                    conf=c,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    frame=int(frame) if frame is not None else None,
                )
            )

    # Write CSV
    csv_path = out_dir / "crops.csv"
    with open(csv_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["source_path", "crop_path", "frame", "conf", "x1", "y1", "x2", "y2"])
        for rec in records:
            wtr.writerow([rec.source_path, rec.crop_path, rec.frame, f"{rec.conf:.6f}", rec.x1, rec.y1, rec.x2, rec.y2])

    return records

