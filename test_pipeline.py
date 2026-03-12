"""
Full Pipeline Test — test_images/
Runs YOLOv8 plate detection + TrOCR recognition on all images in test_images/,
saves annotated outputs to outputs/test_results/, and prints an accuracy summary.
"""

import sys
import json
import math
import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# ── Directories ───────────────────────────────────────────────────────────────
TEST_DIR    = PROJECT_ROOT / "test_images"
OUT_DIR     = PROJECT_ROOT / "outputs" / "test_results"
CROP_DIR    = OUT_DIR / "crops"
ANNOT_DIR   = OUT_DIR / "annotated"
TROCR_CACHE = PROJECT_ROOT / "models" / "trocr"

for d in (OUT_DIR, CROP_DIR, ANNOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── YOLO model path ───────────────────────────────────────────────────────────
def find_yolo_weights() -> Path:
    candidates = [
        PROJECT_ROOT / "runs" / "plate_detection" / "yolov8s_6402" / "weights" / "best.pt",
        PROJECT_ROOT / "runs" / "plate_detection" / "yolov8s_640"  / "weights" / "best.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = list(PROJECT_ROOT.glob("**/plate_detection**/weights/best.pt"))
    if matches:
        return max(matches, key=lambda x: x.stat().st_mtime)
    raise FileNotFoundError("YOLO weights not found. Checked: " + str(candidates))


# ── TrOCR loader ──────────────────────────────────────────────────────────────
def load_trocr():
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    MODEL_ID = "microsoft/trocr-base-printed"
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TROCR_CACHE.mkdir(parents=True, exist_ok=True)
    if (TROCR_CACHE / "config.json").exists():
        print(f"  [TrOCR] Loading from local cache: {TROCR_CACHE}")
        proc  = TrOCRProcessor.from_pretrained(str(TROCR_CACHE))
        model = VisionEncoderDecoderModel.from_pretrained(str(TROCR_CACHE))
    else:
        print(f"  [TrOCR] First run — downloading {MODEL_ID} …")
        proc  = TrOCRProcessor.from_pretrained(MODEL_ID)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
        proc.save_pretrained(str(TROCR_CACHE))
        model.save_pretrained(str(TROCR_CACHE))
        print(f"  [TrOCR] Saved to {TROCR_CACHE}")

    model.to(device).eval()
    print(f"  [TrOCR] Ready on {'GPU ✓' if device.type == 'cuda' else 'CPU'}")
    return proc, model, device


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Phase-2: CLAHE contrast + bilateral denoising."""
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    cleaned  = cv2.bilateralFilter(contrast, 11, 17, 17)
    return cleaned


# ── TrOCR inference ───────────────────────────────────────────────────────────
def trocr_recognize(proc, model, device, crop_bgr: np.ndarray):
    import torch
    cleaned   = preprocess(crop_bgr)
    pil_img   = PILImage.fromarray(cleaned).convert("RGB")
    pv        = proc(images=pil_img, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pv,
            num_beams=4,
            max_new_tokens=32,
            return_dict_in_generate=True,
            output_scores=True,
        )

    raw_text   = proc.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    clean_text = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()

    seq_scores = getattr(outputs, "sequences_scores", None)
    if seq_scores is not None:
        score      = seq_scores[0].item()
        confidence = float(min(1.0, math.exp(max(-10.0, score))))
    else:
        confidence = 0.75

    return clean_text, confidence, raw_text


# ── Confidence label helper ───────────────────────────────────────────────────
def conf_label(c: float) -> str:
    if c >= 0.80: return "HIGH  "
    if c >= 0.60: return "MEDIUM"
    return "LOW   "


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═"*60)
    print("  ANPR FULL PIPELINE TEST")
    print("  YOLOv8 Plate Detection  +  TrOCR Recognition")
    print("═"*60)

    # ── Load YOLO ─────────────────────────────────────────────────────────────
    from ultralytics import YOLO
    weights = find_yolo_weights()
    print(f"\n[YOLO] Loading weights: {weights}")
    detector = YOLO(str(weights))
    print("[YOLO] Ready ✓")

    # ── Load TrOCR ────────────────────────────────────────────────────────────
    print("\n[TrOCR] Initialising …")
    proc, model, device = load_trocr()

    # ── Collect images ────────────────────────────────────────────────────────
    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in TEST_DIR.iterdir() if p.suffix.lower() in exts)
    if not images:
        print(f"\n[ERROR] No images found in {TEST_DIR}")
        return

    print(f"\n[INFO] Found {len(images)} test image(s) in {TEST_DIR}\n")

    # ── Per-image results storage ─────────────────────────────────────────────
    all_results  = []
    all_confs    = []
    total_plates = 0
    pipeline_t0  = time.time()

    # ── Process each image ────────────────────────────────────────────────────
    for img_path in images:
        print("─"*60)
        print(f"  Image : {img_path.name}")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Cannot read image — skipping")
            continue

        h, w = img.shape[:2]
        annotated = img.copy()

        t0       = time.time()
        yolo_res = detector.predict(img, conf=0.25, iou=0.7, verbose=False)
        det_ms   = (time.time() - t0) * 1000

        boxes = yolo_res[0].boxes.xyxy.cpu().numpy() if yolo_res else []

        print(f"  Size  : {w}×{h}  |  Detection: {det_ms:.0f} ms  |  Plates found: {len(boxes)}")

        img_detections = []

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Skip degenerate crops
            if (x2 - x1) < 10 or (y2 - y1) < 5:
                continue

            plate_crop = img[y1:y2, x1:x2]

            t1           = time.time()
            text, conf, raw = trocr_recognize(proc, model, device, plate_crop)
            ocr_ms       = (time.time() - t1) * 1000

            all_confs.append(conf)
            total_plates += 1

            # ── Save plate crop ──────────────────────────────────────────────
            crop_name = f"{img_path.stem}_plate_{idx+1}.jpg"
            cv2.imwrite(str(CROP_DIR / crop_name), plate_crop)

            # ── Annotate image ───────────────────────────────────────────────
            colour = (0, 200, 0) if conf >= 0.80 else (0, 165, 255) if conf >= 0.60 else (0, 0, 220)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
            label = f"{text}  {conf*100:.0f}%"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(annotated, (x1, max(0, y1-lh-8)), (x1+lw+4, y1), colour, -1)
            cv2.putText(annotated, label, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            det_record = {
                "plate_index"  : idx + 1,
                "plate_text"   : text,
                "raw_ocr_text" : raw,
                "confidence"   : round(conf, 4),
                "conf_band"    : conf_label(conf).strip(),
                "bbox"         : [x1, y1, x2, y2],
                "ocr_ms"       : round(ocr_ms, 1),
                "crop_saved"   : crop_name,
            }
            img_detections.append(det_record)

            print(f"  Plate {idx+1}: [{conf_label(conf)}] {conf*100:.1f}%  "
                  f"Text: \"{text}\"  Raw: \"{raw}\"  OCR: {ocr_ms:.0f} ms")

        if not img_detections:
            print("  (no plates detected above threshold)")

        # ── Save annotated image ─────────────────────────────────────────────
        annot_name = f"{img_path.stem}_annotated.jpg"
        cv2.imwrite(str(ANNOT_DIR / annot_name), annotated)

        all_results.append({
            "image"      : img_path.name,
            "size"       : f"{w}x{h}",
            "detect_ms"  : round(det_ms, 1),
            "detections" : img_detections,
            "annotated"  : annot_name,
        })

    total_time = time.time() - pipeline_t0

    # ── Accuracy / confidence summary ─────────────────────────────────────────
    print("\n" + "═"*60)
    print("  TrOCR ACCURACY SUMMARY")
    print("═"*60)
    if all_confs:
        avg    = sum(all_confs) / len(all_confs)
        high   = sum(1 for c in all_confs if c >= 0.80)
        medium = sum(1 for c in all_confs if 0.60 <= c < 0.80)
        low    = sum(1 for c in all_confs if c < 0.60)
        print(f"  Images processed     : {len(all_results)}")
        print(f"  Total plates found   : {total_plates}")
        print(f"  Avg confidence       : {avg*100:.1f}%")
        print(f"  High   (≥80%)        : {high}/{total_plates}")
        print(f"  Medium (60–79%)      : {medium}/{total_plates}")
        print(f"  Low    (<60%)        : {low}/{total_plates}")
        print(f"  Total pipeline time  : {total_time:.1f}s")
        print(f"  Avg time per image   : {total_time/max(1,len(all_results))*1000:.0f} ms")
    else:
        print("  No plates were detected across all test images.")
    print("═"*60)

    # ── Save JSON summary ─────────────────────────────────────────────────────
    summary = {
        "total_images"      : len(images),
        "total_plates"      : total_plates,
        "avg_confidence_pct": round(sum(all_confs)/max(1,len(all_confs))*100, 2),
        "pipeline_time_s"   : round(total_time, 2),
        "ocr_engine"        : "trocr-base-printed",
        "results"           : all_results,
    }
    json_path = OUT_DIR / "test_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Annotated images  → {ANNOT_DIR}")
    print(f"  Plate crops       → {CROP_DIR}")
    print(f"  JSON summary      → {json_path}")
    print()


if __name__ == "__main__":
    main()
