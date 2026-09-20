"""Ultralytics inference isolated from the Drone Flyby wire protocol."""

from __future__ import annotations

import os
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from ultralytics import YOLO

from dtos import OBJECT_CLASSES, DroneFlybyPredictionDto, DroneFlybyPredictRequestDto
from utils import clip_bbox_to_frame, view_bbox_to_global


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "models" / "drone_yolo11n_l0.pt"
DEFAULT_IMGSZ = 960
DEFAULT_CONFIDENCE = 0.01
DEFAULT_IOU = 0.50
DEFAULT_MAX_DETECTIONS = 100


@dataclass(frozen=True)
class RawDetection:
    class_index: int
    confidence: float
    xyxy: tuple[float, float, float, float]


_metrics_lock = threading.Lock()
_metrics: list[dict] = []
_detector_lock = threading.Lock()
_detector_instance = None
_detector_error = None


def reset_runtime_metrics() -> None:
    with _metrics_lock:
        _metrics.clear()


def runtime_metrics() -> dict:
    with _metrics_lock:
        rows = list(_metrics)
    if not rows:
        return {"calls": 0, "rows": []}

    def summarize(key: str) -> dict:
        values = sorted(float(row[key]) for row in rows)
        position = 0.95 * (len(values) - 1)
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        p95 = values[lower] + (position - lower) * (values[upper] - values[lower])
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": p95,
            "max": max(values),
        }

    return {
        "calls": len(rows),
        "model_predict_ms": summarize("model_predict_ms"),
        "model_inference_ms": summarize("model_inference_ms"),
        "rows": rows,
    }


class LocalYoloDetector:
    """One process-wide YOLO model, loaded from a deterministic local path."""

    def __init__(
        self,
        weights: Path,
        imgsz: int = DEFAULT_IMGSZ,
        confidence: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU,
    ) -> None:
        if not weights.is_file():
            raise FileNotFoundError(f"Detector weight is missing: {weights}")
        self.weights = weights
        self.imgsz = imgsz
        self.confidence = confidence
        self.iou = iou
        # Match the four-vCPU deployment budget during local measurements.
        torch.set_num_threads(int(os.environ.get("DRONE_TORCH_THREADS", "4")))
        self.model = YOLO(str(weights))
        names = tuple(self.model.names[index] for index in sorted(self.model.names))
        if names != OBJECT_CLASSES:
            raise ValueError(f"Model class mapping does not match OBJECT_CLASSES: {names}")

    def infer(self, image: np.ndarray) -> list[RawDetection]:
        started = time.perf_counter()
        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=self.iou,
            max_det=DEFAULT_MAX_DETECTIONS,
            device="cpu",
            agnostic_nms=True,
            verbose=False,
        )
        model_predict_ms = (time.perf_counter() - started) * 1000.0
        result = results[0]
        rows: list[RawDetection] = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confidences = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy()
            for box, confidence, class_index in zip(xyxy, confidences, classes):
                rows.append(
                    RawDetection(
                        class_index=int(class_index),
                        confidence=float(confidence),
                        xyxy=tuple(float(value) for value in box),
                    )
                )
        with _metrics_lock:
            _metrics.append(
                {
                    "model_predict_ms": model_predict_ms,
                    "model_inference_ms": float(result.speed.get("inference", 0.0)),
                    "predictions": len(rows),
                }
            )
        return rows

    def warmup(self) -> None:
        self.infer(np.zeros((540, 960, 3), dtype=np.uint8))
        reset_runtime_metrics()


def detections_to_annotations(
    detections: Iterable[RawDetection],
    image_shape: tuple[int, ...],
    request: DroneFlybyPredictRequestDto,
) -> list[DroneFlybyPredictionDto]:
    """Clip, deduplicate, and lift local pixel detections to global DTO boxes."""
    height, width = image_shape[:2]
    annotations: list[DroneFlybyPredictionDto] = []
    seen = set()
    for detection in detections:
        if detection.class_index < 0 or detection.class_index >= len(OBJECT_CLASSES):
            continue
        x1, y1, x2, y2 = detection.xyxy
        x1, x2 = max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))
        y1, y2 = max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        local_box = (x1 / width, y1 / height, x2 / width, y2 / height)
        global_box = clip_bbox_to_frame(
            view_bbox_to_global(
                local_box,
                request.view.source_region_xyxy,
                request.original_width,
                request.original_height,
            )
        )
        if global_box is None:
            continue
        object_id = OBJECT_CLASSES[detection.class_index]
        key = (object_id,) + tuple(round(value, 7) for value in global_box)
        if key in seen:
            continue
        seen.add(key)
        annotations.append(
            DroneFlybyPredictionDto(
                object_id=object_id,
                bbox=list(global_box),
                confidence=max(0.0, min(1.0, float(detection.confidence))),
            )
        )
    return annotations


def get_detector() -> LocalYoloDetector:
    global _detector_instance, _detector_error
    if _detector_instance is not None:
        return _detector_instance
    if _detector_error is not None:
        raise RuntimeError("Detector failed during its one startup load") from _detector_error
    with _detector_lock:
        if _detector_instance is None and _detector_error is None:
            try:
                path = Path(os.environ.get("DRONE_MODEL_PATH", DEFAULT_WEIGHTS))
                _detector_instance = LocalYoloDetector(path)
            except Exception as exc:
                _detector_error = exc
                raise
    return _detector_instance
