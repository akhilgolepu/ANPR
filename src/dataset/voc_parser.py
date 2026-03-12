"""
VOC XML Parser - Supports both annotation formats:
  - Format A: object/name = plate text (dataset-1)
  - Format B: object/name = vehicle type (dataset-2, may have attributes/occluded)
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class BBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return max(0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0, self.ymax - self.ymin)

    def is_valid(self, min_w: float = 1, min_h: float = 1) -> bool:
        return self.width >= min_w and self.height >= min_h


@dataclass
class Annotation:
    filename: str
    folder: str
    width: int
    height: int
    objects: list[tuple[str, BBox]]


def parse_voc_xml(xml_path: Path) -> Annotation | None:
    """Parse VOC format XML, handling both schema variants."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return None

    # Size
    size_el = root.find("size")
    if size_el is None:
        return None

    width_el = size_el.find("width")
    height_el = size_el.find("height")
    width = int(width_el.text or 0) if width_el is not None else 0
    height = int(height_el.text or 0) if height_el is not None else 0
    if width <= 0 or height <= 0:
        return None

    # Filename
    filename_el = root.find("filename")
    folder_el = root.find("folder")
    filename = (filename_el.text or "").strip() if filename_el is not None else ""
    folder = (folder_el.text or "").strip() if folder_el is not None else ""
    if not filename:
        filename = xml_path.with_suffix(".jpg").name

    objects: list[tuple[str, BBox]] = []

    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None or not (name_el.text or "").strip():
            continue

        name = (name_el.text or "").strip()
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = _float(bndbox.find("xmin"))
        ymin = _float(bndbox.find("ymin"))
        xmax = _float(bndbox.find("xmax"))
        ymax = _float(bndbox.find("ymax"))
        if xmin is None or ymin is None or xmax is None or ymax is None:
            continue
        if xmax <= xmin or ymax <= ymin:
            continue

        box = BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        objects.append((name, box))

    return Annotation(filename=filename, folder=folder, width=width, height=height, objects=objects)


def _float(el: ET.Element | None) -> float | None:
    if el is None or el.text is None:
        return None
    try:
        return float(el.text.strip())
    except ValueError:
        return None


def is_plate_annotation(annotation: Annotation) -> bool:
    """
    Heuristic: object names that look like Indian license plates
    (e.g. MH02FN2783, AP05BY7799, OD01Q4668) vs vehicle types (car, bus).
    """
    plate_pattern = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$", re.I)
    vehicle_names = {"car", "bus", "tempo", "vehicle_truck", "two_wheelers", "auto", "tractor", "bicycle"}

    for name, _ in annotation.objects:
        n = name.strip()
        if n.lower() in vehicle_names:
            return False
        if plate_pattern.match(n) or (len(n) >= 8 and n[:2].isalpha() and any(c.isdigit() for c in n)):
            return True
    # Default: if any object name looks like plate (alphanumeric, 8+ chars), treat as plate
    for name, _ in annotation.objects:
        if len(name) >= 8 and name.isalnum():
            return True
    return False


def find_image_for_xml(xml_path: Path, search_root: Path | None = None) -> Path | None:
    """Resolve image path from XML. Checks same dir, then filename from XML."""
    ann = parse_voc_xml(xml_path)
    parent = xml_path.parent

    candidates = [
        xml_path.with_suffix(".jpg"),
        xml_path.with_suffix(".jpeg"),
        xml_path.with_suffix(".png"),
    ]
    if ann is not None and ann.filename:
        candidates.extend([
            parent / ann.filename,
            parent / ann.filename.replace(".jpeg", ".jpg").replace(".png", ".jpg"),
        ])
    if search_root is not None and ann is not None:
        candidates.append(search_root / ann.filename)

    for p in candidates:
        if p.exists():
            return p
    return None


def iter_voc_annotations(root: Path) -> Iterator[tuple[Path, Path]]:
    """Yield (xml_path, image_path) for all VOC annotations under root."""
    for xml_path in root.rglob("*.xml"):
        img_path = find_image_for_xml(xml_path, search_root=root)
        if img_path is not None:
            yield xml_path, img_path
