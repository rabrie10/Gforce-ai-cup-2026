"""Build and validate the 960x540 Level-0 YOLO training set.

The hosted evaluator downsamples the 3840x2160 source frame with OpenCV
INTER_AREA before sending Level 0.  This script reproduces that operation,
converts every source-pixel annotation to a YOLO label, and writes tracked
validation summaries plus a few visual overlays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import cv2


DRONE_ROOT = Path(__file__).resolve().parents[1]
if str(DRONE_ROOT) not in sys.path:
    sys.path.insert(0, str(DRONE_ROOT))

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH, OBJECT_CLASSES, TRANSMITTED_VIEW_SIZE


EXPECTED_FRAMES = 25
EXPECTED_ANNOTATIONS = 259
SOURCE = DRONE_ROOT / "src" / "helsinki"
GENERATED = Path(__file__).resolve().parent / "generated" / "l0_helsinki"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
OVERLAY_FRAMES = (0, 12, 24)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_dataset() -> dict:
    images_out = GENERATED / "images" / "train"
    labels_out = GENERATED / "labels" / "train"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    image_paths = sorted((SOURCE / "images").glob("frame_*.png"))
    annotation_paths = sorted((SOURCE / "annotations").glob("frame_*.json"))
    if len(image_paths) != EXPECTED_FRAMES or len(annotation_paths) != EXPECTED_FRAMES:
        raise ValueError(
            f"Expected {EXPECTED_FRAMES} source images and annotations; got "
            f"{len(image_paths)} and {len(annotation_paths)}"
        )

    class_to_index = {name: index for index, name in enumerate(OBJECT_CLASSES)}
    class_measurements: dict[str, list[dict]] = defaultdict(list)
    frame_records = []
    annotation_count = 0
    target_width, target_height = TRANSMITTED_VIEW_SIZE
    scale_x, scale_y = target_width / IMAGE_WIDTH, target_height / IMAGE_HEIGHT

    for image_path, annotation_path in zip(image_paths, annotation_paths):
        if image_path.stem != annotation_path.stem:
            raise ValueError(f"Image/annotation mismatch: {image_path} vs {annotation_path}")
        frame = int(image_path.stem.rsplit("_", 1)[1])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
            shape = None if image is None else image.shape
            raise ValueError(f"Unexpected source image {image_path}: {shape}")
        l0_image = cv2.resize(
            image, (target_width, target_height), interpolation=cv2.INTER_AREA
        )
        output_image = images_out / image_path.name
        if not cv2.imwrite(str(output_image), l0_image):
            raise OSError(f"Could not write {output_image}")

        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        if payload.get("frame") != frame:
            raise ValueError(f"Frame field mismatch in {annotation_path}")
        label_lines = []
        overlay = l0_image.copy()
        for annotation in payload["annotations"]:
            object_id = annotation["object_id"]
            if object_id not in class_to_index:
                raise ValueError(f"Unknown class {object_id!r} in {annotation_path}")
            x1, y1, x2, y2 = map(float, annotation["bbox"])
            x1 = min(max(x1, 0.0), float(IMAGE_WIDTH))
            x2 = min(max(x2, 0.0), float(IMAGE_WIDTH))
            y1 = min(max(y1, 0.0), float(IMAGE_HEIGHT))
            y2 = min(max(y2, 0.0), float(IMAGE_HEIGHT))
            if not x1 < x2 or not y1 < y2:
                raise ValueError(f"Zero/negative-area box in {annotation_path}: {annotation}")

            # Uniform 1/4 downsampling means normalized YOLO coordinates are
            # unchanged, but compute through L0 pixels so the geometry is explicit.
            lx1, ly1, lx2, ly2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
            center_x = ((lx1 + lx2) / 2) / target_width
            center_y = ((ly1 + ly2) / 2) / target_height
            width = (lx2 - lx1) / target_width
            height = (ly2 - ly1) / target_height
            values = (center_x, center_y, width, height)
            if not all(0.0 <= value <= 1.0 for value in values) or width <= 0 or height <= 0:
                raise ValueError(f"Illegal normalized label in {annotation_path}: {values}")
            label_lines.append(
                f"{class_to_index[object_id]} " + " ".join(f"{value:.8f}" for value in values)
            )
            pixel_width, pixel_height = lx2 - lx1, ly2 - ly1
            class_measurements[object_id].append(
                {
                    "width_px": pixel_width,
                    "height_px": pixel_height,
                    "area_px2": pixel_width * pixel_height,
                }
            )
            annotation_count += 1
            if frame in OVERLAY_FRAMES:
                p1 = (round(lx1), round(ly1))
                p2 = (round(lx2), round(ly2))
                cv2.rectangle(overlay, p1, p2, (0, 255, 255), 1)
                cv2.putText(
                    overlay,
                    object_id,
                    (p1[0], max(10, p1[1] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        label_path = labels_out / f"{image_path.stem}.txt"
        label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        if frame in OVERLAY_FRAMES:
            overlay_path = ARTIFACTS / f"l0_overlay_frame_{frame:06d}.jpg"
            if not cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"Could not write {overlay_path}")
        frame_records.append(
            {
                "frame": frame,
                "annotations": len(label_lines),
                "source_image_sha256": sha256(image_path),
                "generated_image_sha256": sha256(output_image),
                "label_sha256": sha256(label_path),
            }
        )

    if annotation_count != EXPECTED_ANNOTATIONS:
        raise ValueError(f"Expected {EXPECTED_ANNOTATIONS} annotations, got {annotation_count}")
    missing = [name for name in OBJECT_CLASSES if not class_measurements[name]]
    if missing:
        raise ValueError(f"Classes absent after conversion: {missing}")

    yaml_lines = [
        f"path: {GENERATED.as_posix()}",
        "train: images/train",
        # Deliberately the same 25 frames: this is a fit diagnostic only. The
        # hosted evaluator is the external validation set.
        "val: images/train",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(OBJECT_CLASSES))
    dataset_yaml = GENERATED / "dataset.yaml"
    dataset_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    class_summary = []
    for index, name in enumerate(OBJECT_CLASSES):
        rows = class_measurements[name]
        widths = [row["width_px"] for row in rows]
        heights = [row["height_px"] for row in rows]
        areas = [row["area_px2"] for row in rows]
        class_summary.append(
            {
                "class_index": index,
                "object_id": name,
                "count": len(rows),
                "width_px_min": min(widths),
                "width_px_median": statistics.median(widths),
                "width_px_p95": percentile(widths, 0.95),
                "width_px_max": max(widths),
                "height_px_min": min(heights),
                "height_px_median": statistics.median(heights),
                "height_px_p95": percentile(heights, 0.95),
                "height_px_max": max(heights),
                "area_px2_median": statistics.median(areas),
            }
        )

    csv_path = ARTIFACTS / "l0_class_size_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_summary[0]))
        writer.writeheader()
        writer.writerows(class_summary)

    all_widths = [row["width_px"] for rows in class_measurements.values() for row in rows]
    all_heights = [row["height_px"] for rows in class_measurements.values() for row in rows]
    summary = {
        "status": "PASS",
        "source_geometry": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "l0_geometry": [target_width, target_height],
        "resize_interpolation": "cv2.INTER_AREA",
        "frames": len(frame_records),
        "annotations": annotation_count,
        "class_mapping": {str(i): name for i, name in enumerate(OBJECT_CLASSES)},
        "all_boxes": {
            "width_px_min": min(all_widths),
            "width_px_median": statistics.median(all_widths),
            "width_px_p95": percentile(all_widths, 0.95),
            "width_px_max": max(all_widths),
            "height_px_min": min(all_heights),
            "height_px_median": statistics.median(all_heights),
            "height_px_p95": percentile(all_heights, 0.95),
            "height_px_max": max(all_heights),
        },
        "class_summary": class_summary,
        "frames_detail": frame_records,
        "overlay_frames": list(OVERLAY_FRAMES),
        "training_policy": "all 25 sequential frames; no random adjacent-frame split",
        "validation_policy": "same-frame fit diagnostic only; hosted evaluation is external",
        "dataset_yaml": str(dataset_yaml),
    }
    summary_path = ARTIFACTS / "l0_data_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    summary = build_dataset()
    print(json.dumps({key: summary[key] for key in ("status", "frames", "annotations")}, indent=2))
    print(f"Dataset: {GENERATED / 'dataset.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
