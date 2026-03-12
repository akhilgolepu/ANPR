"""
Dataset Preparation Pipeline
- Converts VOC XML → YOLO format (vehicle + plate detection)
- Extracts plate crops for OCR training
- Creates train/val/test splits
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any

from .voc_parser import Annotation, BBox, parse_voc_xml, find_image_for_xml, iter_voc_annotations


def _iter_plate_crop_images(raw_root: Path, sources: list[str]) -> list[tuple[Path, Annotation]]:
    """Yield (img_path, ann) for plate crop sources. Each image is a full plate (bbox = full image)."""
    try:
        from PIL import Image
    except ImportError:
        return []
    seen: set[Path] = set()
    out: list[tuple[Path, Annotation]] = []
    for src in sources:
        src_path = raw_root / src.replace("\\", "/")
        if not src_path.exists():
            continue
        for img_path in src_path.rglob("*.png"):
            if img_path.resolve() in seen:
                continue
            seen.add(img_path.resolve())
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                continue
            if w < 20 or h < 8:
                continue
            box = BBox(xmin=0, ymin=0, xmax=float(w), ymax=float(h))
            ann = Annotation(filename=img_path.name, folder="", width=w, height=h, objects=[("plate", box)])
            out.append((img_path, ann))
        for img_path in src_path.rglob("*.jpg"):
            if img_path.resolve() in seen:
                continue
            seen.add(img_path.resolve())
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                continue
            if w < 20 or h < 8:
                continue
            box = BBox(xmin=0, ymin=0, xmax=float(w), ymax=float(h))
            ann = Annotation(filename=img_path.name, folder="", width=w, height=h, objects=[("plate", box)])
            out.append((img_path, ann))
    return out


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config. Requires PyYAML (pip install pyyaml)."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def bbox_to_yolo(box: BBox, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert VOC (xmin,ymin,xmax,ymax) to YOLO (x_center, y_center, width, height) normalized [0,1]."""
    if img_w <= 0 or img_h <= 0:
        return 0.0, 0.0, 0.0, 0.0
    xc = (box.xmin + box.xmax) / 2 / img_w
    yc = (box.ymin + box.ymax) / 2 / img_h
    w = box.width / img_w
    h = box.height / img_h
    xc = max(0, min(1, xc))
    yc = max(0, min(1, yc))
    w = max(0, min(1, w))
    h = max(0, min(1, h))
    return xc, yc, w, h


def prepare_vehicle_dataset(
    raw_root: Path,
    sources: list[str],
    out_root: Path,
    class_map: dict[str, int],
    splits: dict[str, float],
    seed: int,
) -> dict[str, int]:
    """Prepare YOLO-format vehicle detection dataset from VOC sources."""
    all_items: list[tuple[Path, Path, Annotation]] = []
    for src in sources:
        src_path = raw_root / src.replace("\\", "/")
        if not src_path.exists():
            continue
        for xml_path, img_path in iter_voc_annotations(src_path):
            ann = parse_voc_xml(xml_path)
            if ann is None or not ann.objects:
                continue
            valid_objs = [(n, b) for n, b in ann.objects if n.strip().lower() in {k.lower() for k in class_map} or n.strip().lower() in {"auto", "tractor", "bicycle"}]
            if not valid_objs:
                continue
            all_items.append((xml_path, img_path, ann))

    if not all_items:
        return {}

    random.seed(seed)
    shuffled = all_items[:]
    random.shuffle(shuffled)

    total = len(shuffled)
    t_ratio = splits.get("train", 0.75)
    v_ratio = splits.get("val", 0.15)
    te_ratio = splits.get("test", 0.10)
    t_end = int(total * t_ratio)
    v_end = t_end + int(total * v_ratio)
    train_items = shuffled[:t_end]
    val_items = shuffled[t_end:v_end]
    test_items = shuffled[v_end:]

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "images").mkdir(exist_ok=True)
    (out_root / "labels").mkdir(exist_ok=True)

    for split_name, items in [("train", train_items), ("val", val_items), ("test", test_items)]:
        (out_root / "images" / split_name).mkdir(exist_ok=True)
        (out_root / "labels" / split_name).mkdir(exist_ok=True)

    label_dir = out_root / "labels"
    images_dir = out_root / "images"

    def _class_id(n: str) -> int | None:
        n = n.strip().lower()
        cid = class_map.get(n)
        if cid is not None:
            return cid
        if n == "auto":
            return class_map.get("tempo", 2)
        if n == "tractor":
            return class_map.get("vehicle_truck", 3)
        if n == "bicycle":
            return class_map.get("two_wheelers", 4)
        return None

    def process_split(split_name: str, items: list) -> int:
        count = 0
        for i, (xml_path, img_path, ann) in enumerate(items):
            base = f"{split_name}_{xml_path.stem}_{i}"
            img_dst = images_dir / split_name / f"{base}.jpg"
            lbl_dst = label_dir / split_name / f"{base}.txt"
            try:
                shutil.copy2(img_path, img_dst)
            except Exception:
                continue
            lines = []
            for name, box in ann.objects:
                cid = _class_id(name)
                if cid is None:
                    continue
                if not box.is_valid():
                    continue
                xc, yc, w, h = bbox_to_yolo(box, ann.width, ann.height)
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            lbl_dst.write_text("\n".join(lines) + "\n" if lines else "")
            count += 1
        return count

    n_train = process_split("train", train_items)
    n_val = process_split("val", val_items)
    n_test = process_split("test", test_items)

    # data.yaml for YOLO
    class_names = [""] * (max(class_map.values()) + 1)
    for name, idx in class_map.items():
        if name not in ("auto", "tractor", "bicycle"):
            class_names[idx] = name
    data_yaml = f"""# Vehicle Detection Dataset - YOLO format
path: {out_root.absolute()}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}
"""
    (out_root / "data.yaml").write_text(data_yaml)
    return {"train": n_train, "val": n_val, "test": n_test}


def prepare_plate_dataset(
    raw_root: Path,
    sources: list[str],
    out_root: Path,
    splits: dict[str, float],
    seed: int,
    crop_sources: list[str] | None = None,
) -> dict[str, int]:
    """Prepare YOLO-format plate detection dataset. Single class 'license_plate'.
    Sources: VOC XML (full images with plate bbox). crop_sources: full-image plate crops."""
    all_items: list[tuple[Path, Path, Annotation]] = []
    for src in sources:
        src_path = raw_root / src.replace("\\", "/")
        if not src_path.exists():
            continue
        for xml_path, img_path in iter_voc_annotations(src_path):
            ann = parse_voc_xml(xml_path)
            if ann is None or not ann.objects:
                continue
            all_items.append((xml_path, img_path, ann))
    crop_items = _iter_plate_crop_images(raw_root, crop_sources or [])
    for img_path, ann in crop_items:
        all_items.append((img_path, img_path, ann))

    if not all_items:
        return {}

    random.seed(seed)
    shuffled = all_items[:]
    random.shuffle(shuffled)

    total = len(shuffled)
    t_ratio = splits.get("train", 0.80)
    v_ratio = splits.get("val", 0.10)
    te_ratio = splits.get("test", 0.10)
    t_end = int(total * t_ratio)
    v_end = t_end + int(total * v_ratio)
    train_items = shuffled[:t_end]
    val_items = shuffled[t_end:v_end]
    test_items = shuffled[v_end:]

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "images").mkdir(exist_ok=True)
    (out_root / "labels").mkdir(exist_ok=True)

    for split_name in ("train", "val", "test"):
        (out_root / "images" / split_name).mkdir(exist_ok=True)
        (out_root / "labels" / split_name).mkdir(exist_ok=True)

    label_dir = out_root / "labels"
    images_dir = out_root / "images"
    plate_class = 0

    def process_split(split_name: str, items: list) -> int:
        """
        Process a dataset split: copy images and create YOLO labels.
        
        Image naming convention: {split}_{source_dataset}_{sequential_id}.jpg
        Example: train_dataset1_000001.jpg, val_dataset3_000042.jpg
        
        Steps:
        1. Extract source dataset name from file path
        2. Generate consistent filename with zero-padded sequential ID
        3. Copy image to destination
        4. Create YOLO format label file
        """
        count = 0
        for i, (name_src, img_path, ann) in enumerate(items):
            # Consistent naming: split_source_sequential_id
            # Extract source dataset name from path for better organization
            # name_src can be XML path or image path (for crop sources)
            if name_src.suffix.lower() in ('.xml',):
                # VOC XML source - use parent directory name
                source_name = name_src.parent.name if name_src.parent.name else "dataset"
            else:
                # Image path (crop source) - use parent directory name
                source_name = name_src.parent.name if name_src.parent.name else "dataset"
            
            # Clean source name (remove spaces, special chars)
            source_name = "".join(c if c.isalnum() or c == "_" else "_" for c in source_name)[:20]
            base = f"{split_name}_{source_name}_{i:06d}"  # 6-digit zero-padded ID
            ext = img_path.suffix.lower()
            img_dst = images_dir / split_name / f"{base}{ext if ext in ('.jpg', '.jpeg', '.png') else '.jpg'}"
            lbl_dst = label_dir / split_name / f"{base}.txt"
            try:
                shutil.copy2(img_path, img_dst)
            except Exception:
                continue
            lines = []
            for _, box in ann.objects:
                if not box.is_valid(min_w=5, min_h=3):
                    continue
                xc, yc, w, h = bbox_to_yolo(box, ann.width, ann.height)
                lines.append(f"{plate_class} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            lbl_dst.write_text("\n".join(lines) + "\n" if lines else "")
            count += 1
        return count

    n_train = process_split("train", train_items)
    n_val = process_split("val", val_items)
    n_test = process_split("test", test_items)

    data_yaml = f"""# Plate Detection Dataset - YOLO format
path: {out_root.absolute()}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['license_plate']
"""
    (out_root / "data.yaml").write_text(data_yaml)
    return {"train": n_train, "val": n_val, "test": n_test}


def prepare_ocr_dataset(
    raw_root: Path,
    sources: list[str],
    out_root: Path,
    min_w: int = 20,
    min_h: int = 8,
    padding_ratio: float = 0.1,
    splits: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[str, int]:
    """Extract plate crops for OCR. Each crop saved with plate text as filename/label."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL/Pillow required for OCR crop extraction. pip install pillow")

    all_crops: list[tuple[Path, str, BBox, int, int]] = []  # (img_path, plate_text, box, w, h)
    for src in sources:
        src_path = raw_root / src.replace("\\", "/")
        if not src_path.exists():
            continue
        for xml_path, img_path in iter_voc_annotations(src_path):
            ann = parse_voc_xml(xml_path)
            if ann is None or not ann.objects:
                continue
            for name, box in ann.objects:
                text = (name or "").strip()
                if len(text) < 4:
                    continue
                if box.width < min_w or box.height < min_h:
                    continue
                all_crops.append((img_path, text, box, ann.width, ann.height))

    if not all_crops:
        return {}

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "images").mkdir(exist_ok=True)
    (out_root / "labels").mkdir(exist_ok=True)

    if splits:
        random.seed(seed)
        shuffled = all_crops[:]
        random.shuffle(shuffled)
        total = len(shuffled)
        t_ratio = splits.get("train", 0.75)
        v_ratio = splits.get("val", 0.15)
        t_end = int(total * t_ratio)
        v_end = t_end + int(total * v_ratio)
        train_crops = shuffled[:t_end]
        val_crops = shuffled[t_end:v_end]
        test_crops = shuffled[v_end:]
    else:
        train_crops = all_crops
        val_crops = []
        test_crops = []

    def safe_filename(t: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in t)[:64]

    def extract_crops(crops: list, split: str) -> int:
        split_img = out_root / "images" / split
        split_lbl = out_root / "labels" / split
        split_img.mkdir(parents=True, exist_ok=True)
        split_lbl.mkdir(parents=True, exist_ok=True)
        count = 0
        for i, (img_path, text, box, img_w, img_h) in enumerate(crops):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            pad_w = max(1, int(box.width * padding_ratio))
            pad_h = max(1, int(box.height * padding_ratio))
            x1 = max(0, int(box.xmin) - pad_w)
            y1 = max(0, int(box.ymin) - pad_h)
            x2 = min(img.width, int(box.xmax) + pad_w)
            y2 = min(img.height, int(box.ymax) + pad_h)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img.crop((x1, y1, x2, y2))
            if crop.width < 5 or crop.height < 3:
                continue
            fname = safe_filename(text) + f"_{i}.jpg"
            dst = split_img / fname
            crop.save(dst, "JPEG", quality=95)
            (split_lbl / (dst.stem + ".txt")).write_text(text)
            count += 1
        return count

    n_train = extract_crops(train_crops, "train") if train_crops else 0
    n_val = extract_crops(val_crops, "val") if val_crops else 0
    n_test = extract_crops(test_crops, "test") if test_crops else 0

    return {"train": n_train, "val": n_val, "test": n_test}


def run(config_path: Path | None = None) -> None:
    """Run full dataset preparation pipeline."""
    base = Path(__file__).resolve().parent.parent.parent
    cfg_path = config_path or base / "config" / "dataset.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = load_config(cfg_path)
    raw_root = base / cfg.get("paths", {}).get("raw_root", "datasets/raw")
    proc_root = base / cfg.get("paths", {}).get("processed_root", "datasets/processed")
    splits = cfg.get("splits", {"train": 0.75, "val": 0.15, "test": 0.10})
    seed = int(splits.get("seed", 42))

    plate_sources = cfg.get("paths", {}).get("plate_sources", ["dataset-1/State-wise_OLX", "dataset-1/video_images"])
    plate_crop_sources = cfg.get("paths", {}).get("plate_crop_sources", [])
    vehicle_sources = cfg.get("paths", {}).get("vehicle_sources", ["dataset--2"])

    out_cfg = cfg.get("outputs", {})
    vehicle_out = base / out_cfg.get("vehicle_detection", "datasets/processed/vehicle_detection")
    plate_out = base / out_cfg.get("plate_detection", "datasets/processed/plate_detection")
    ocr_out = base / out_cfg.get("ocr_dataset", "datasets/processed/ocr_dataset")

    vehicle_classes = cfg.get("vehicle_classes", {
        "car": 0, "bus": 1, "tempo": 2, "vehicle_truck": 3, "two_wheelers": 4,
        "auto": 2, "tractor": 3, "bicycle": 4,
    })
    plate_crop = cfg.get("plate_crop", {"min_width": 20, "min_height": 8, "padding_ratio": 0.1})

    print("Preparing Vehicle Detection dataset...")
    v_counts = prepare_vehicle_dataset(
        raw_root, vehicle_sources, vehicle_out, vehicle_classes, splits, seed
    )
    print(f"  Vehicle: train={v_counts.get('train',0)}, val={v_counts.get('val',0)}, test={v_counts.get('test',0)}")

    print("Preparing Plate Detection dataset...")
    p_counts = prepare_plate_dataset(
        raw_root, plate_sources, plate_out, splits, seed, crop_sources=plate_crop_sources
    )
    print(f"  Plate: train={p_counts.get('train',0)}, val={p_counts.get('val',0)}, test={p_counts.get('test',0)}")

    print("Preparing OCR dataset (plate crops)...")
    ocr_counts = prepare_ocr_dataset(
        raw_root, plate_sources, ocr_out,
        min_w=plate_crop.get("min_width", 20),
        min_h=plate_crop.get("min_height", 8),
        padding_ratio=float(plate_crop.get("padding_ratio", 0.1)),
        splits=splits,
        seed=seed,
    )
    print(f"  OCR: train={ocr_counts.get('train',0)}, val={ocr_counts.get('val',0)}, test={ocr_counts.get('test',0)}")

    print("\nDone. Outputs:")
    print(f"  - {vehicle_out}")
    print(f"  - {plate_out}")
    print(f"  - {ocr_out}")
