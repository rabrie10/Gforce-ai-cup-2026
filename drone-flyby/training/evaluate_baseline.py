"""Run oracle, offline, and realtime evaluation and preserve exact metrics."""

from __future__ import annotations

import dataclasses
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import uvicorn
import cv2


DRONE_ROOT = Path(__file__).resolve().parents[1]
if str(DRONE_ROOT) not in sys.path:
    sys.path.insert(0, str(DRONE_ROOT))

import local_evaluator
from api import app, endpoint_metrics, reset_endpoint_metrics
from detector import reset_runtime_metrics, runtime_metrics
from dtos import OBJECT_CLASSES


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
HOST = "127.0.0.1"
PORT = 9054
URL = f"http://{HOST}:{PORT}/predict"
OVERLAY_FRAMES = (0, 12, 24)


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    p95 = ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": p95,
        "max": max(values),
    }


def render_overlays(predictions: dict[int, list[dict]]) -> None:
    """Draw yellow GT and magenta predictions for qualitative review."""
    for frame in OVERLAY_FRAMES:
        image = local_evaluator.load_frame(frame, "helsinki")
        image = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)
        for row in local_evaluator.load_annotations(frame, "helsinki"):
            x1, y1, x2, y2 = (round(float(value) / 4) for value in row["bbox"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.putText(image, f"GT {row['object_id']}", (x1, max(10, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
        for row in predictions.get(frame, []):
            x1, y1, x2, y2 = (round(float(value) / 4) for value in row["bbox"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 1)
            label = f"P {row['object_id']} {float(row['confidence']):.2f}"
            cv2.putText(image, label, (x1, min(535, y2 + 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1, cv2.LINE_AA)
        output = ARTIFACTS / f"prediction_overlay_frame_{frame:06d}.jpg"
        if not cv2.imwrite(str(output), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Could not write {output}")


def run_mode(realtime: bool) -> dict:
    reset_runtime_metrics()
    reset_endpoint_metrics()
    predictions, stats = local_evaluator.replay(URL, "helsinki", realtime, 0.0, False)
    score, per_class = local_evaluator.score("helsinki", predictions)
    frames = local_evaluator.frame_numbers("helsinki")
    counts = [len(predictions.get(frame, [])) for frame in frames]
    confidences = [
        float(row["confidence"])
        for detections in predictions.values()
        for row in detections
    ]
    return {
        "mode": "realtime" if realtime else "non_realtime",
        "map50": score,
        "per_class_ap50": {name: per_class.get(name) for name in OBJECT_CLASSES},
        "statistics": dataclasses.asdict(stats),
        "round_trip_ms": distribution(stats.round_trip_ms),
        "predictions_per_frame": distribution([float(value) for value in counts]),
        "confidence": distribution(confidences),
        "runtime": runtime_metrics(),
        "endpoint_processing_ms": endpoint_metrics(),
        "predictions": {str(frame): predictions.get(frame, []) for frame in frames},
    }


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    oracle_predictions = local_evaluator.oracle_predictions("helsinki")
    oracle_score, oracle_classes = local_evaluator.score("helsinki", oracle_predictions)
    if oracle_score != 1.0 or any(value != 1.0 for value in oracle_classes.values()):
        raise RuntimeError(f"Oracle sanity failed: {oracle_score}, {oracle_classes}")

    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(120):
        if server.started:
            break
        time.sleep(0.25)
    if not server.started:
        raise RuntimeError("Local FastAPI server did not start")
    try:
        non_realtime = run_mode(False)
        render_overlays({int(key): value for key, value in non_realtime["predictions"].items()})
        realtime = run_mode(True)
    finally:
        server.should_exit = True
        thread.join(timeout=30)

    result = {
        "status": "PASS",
        "oracle_map50": oracle_score,
        "oracle_per_class_ap50": oracle_classes,
        "non_realtime": non_realtime,
        "realtime": realtime,
    }
    output = ARTIFACTS / "evaluation_results.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
