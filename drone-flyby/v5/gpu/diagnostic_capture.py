"""Optional, lossy diagnostic capture for decoded V6 observations.

The capture path is deliberately independent of inference. ``submit`` performs
only bounded admission and a defensive image copy; PNG encoding and all disk
I/O happen on one background writer thread.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


_SAFE_PART = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_part(value: Any) -> str:
    text = _SAFE_PART.sub("_", str(value)).strip("._")
    return text or "unknown"


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _model_dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_model_dump(item) for item in value]
    return value


@dataclass(frozen=True)
class CaptureConfig:
    enabled: bool = False
    output_dir: Path = Path("/workspace/v6-diagnostics")
    queue_size: int = 8
    per_level_limit: int = 10
    frame_window_size: int = 50
    per_window_limit: int = 2
    quota_bytes: int = 50 * 1024 * 1024

    @classmethod
    def from_environment(cls) -> "CaptureConfig":
        return cls(
            enabled=os.getenv("V6_DIAGNOSTIC_CAPTURE", "0") == "1",
            output_dir=Path(os.getenv("V6_DIAGNOSTIC_DIR", "/workspace/v6-diagnostics")),
            queue_size=max(1, int(os.getenv("V6_DIAGNOSTIC_QUEUE_SIZE", "8"))),
            per_level_limit=max(0, int(os.getenv("V6_DIAGNOSTIC_PER_LEVEL", "10"))),
            frame_window_size=max(1, int(os.getenv("V6_DIAGNOSTIC_WINDOW_SIZE", "50"))),
            per_window_limit=max(0, int(os.getenv("V6_DIAGNOSTIC_PER_WINDOW", "2"))),
            quota_bytes=max(0, int(os.getenv("V6_DIAGNOSTIC_QUOTA_BYTES", str(50 * 1024 * 1024)))),
        )


@dataclass
class _CaptureItem:
    sequence_id: str
    request_id: str
    image: np.ndarray
    metadata: dict[str, Any]
    candidates: Any
    predictions: Any


class DiagnosticCapture:
    """Bounded asynchronous writer for actual decoded observations.

    The class is inert when disabled. Admission never waits for the writer and
    all writer failures are converted into counters rather than exceptions.
    """

    _STOP = object()

    def __init__(self, config: CaptureConfig | None = None):
        self.config = config or CaptureConfig.from_environment()
        self._queue: queue.Queue[_CaptureItem | object] = queue.Queue(self.config.queue_size)
        self._counts: dict[tuple[str, int], int] = {}
        self._reserved_bytes = 0
        self._lock = threading.Lock()
        self._stats = {"accepted": 0, "written": 0, "dropped_queue": 0,
                       "dropped_quota": 0, "dropped_error": 0}
        self._thread: threading.Thread | None = None
        if self.config.enabled:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(target=self._writer, name="v6-diagnostic-writer", daemon=True)
            self._thread.start()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def submit(self, request: Any, image: np.ndarray, candidates: Any, predictions: Any) -> bool:
        """Admit one actual decoded request without waiting for disk I/O."""
        if not self.config.enabled:
            return False
        view = request.view
        level = int(view.resolution_level)
        if level not in (1, 2):
            return False
        sequence_id = str(request.sequence_id)
        request_id = str(request.request_id)
        key = (sequence_id, level)
        window = int(request.frame_index) // self.config.frame_window_size
        window_key = (sequence_id, level, window)
        with self._lock:
            if self._counts.get(key, 0) >= self.config.per_level_limit:
                self._stats["dropped_quota"] += 1
                return False
            if self._counts.get(window_key, 0) >= self.config.per_window_limit:
                self._stats["dropped_quota"] += 1
                return False
            if self._queue.full():
                self._stats["dropped_queue"] += 1
                return False
            self._counts[key] = self._counts.get(key, 0) + 1
            self._counts[window_key] = self._counts.get(window_key, 0) + 1
            self._stats["accepted"] += 1
        try:
            copied = np.ascontiguousarray(image).copy()
            metadata = {
                "sequence_id": sequence_id,
                "request_id": request_id,
                "frame_index": int(request.frame_index),
                "frame": int(request.frame),
                "received_camera": {
                    "resolution_level": level,
                    "center_x": int(view.center_x),
                    "center_y": int(view.center_y),
                    "source_region_xyxy": [int(value) for value in view.source_region_xyxy],
                },
                "image": {
                    "width": int(copied.shape[1]),
                    "height": int(copied.shape[0]),
                    "channels": int(copied.shape[2]) if copied.ndim == 3 else 1,
                    "dtype": str(copied.dtype),
                    "sha256": hashlib.sha256(copied.tobytes()).hexdigest(),
                },
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "candidate_stage": "post_merge_detector_candidates",
            }
            item = _CaptureItem(sequence_id, request_id, copied, metadata,
                                _model_dump(candidates), _model_dump(predictions))
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            with self._lock:
                self._counts[key] = max(0, self._counts.get(key, 1) - 1)
                self._counts[window_key] = max(0, self._counts.get(window_key, 1) - 1)
                self._stats["dropped_queue"] += 1
            return False
        except Exception:
            with self._lock:
                self._stats["dropped_error"] += 1
            return False

    def close(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._queue.put(self._STOP, timeout=timeout)
        self._thread.join(timeout)

    def _writer(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                self._write(item)
            except Exception:
                with self._lock:
                    self._stats["dropped_error"] += 1
            finally:
                self._queue.task_done()

    def _write(self, item: _CaptureItem) -> None:
        ok, encoded = cv2.imencode(".png", item.image)
        if not ok:
            raise ValueError("PNG encoding failed")
        image_bytes = encoded.tobytes()
        metadata = dict(item.metadata)
        metadata["candidates"] = item.candidates
        metadata["predictions"] = item.predictions
        metadata_bytes = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode("utf-8")
        required = len(image_bytes) + len(metadata_bytes)
        with self._lock:
            current = self._reserved_bytes + self._directory_bytes()
            if current + required > self.config.quota_bytes:
                self._stats["dropped_quota"] += 1
                return
            self._reserved_bytes += required
        try:
            stem = f"{_safe_part(item.sequence_id)}__{_safe_part(item.request_id)}"
            image_path = self.config.output_dir / f"{stem}.png"
            metadata_path = self.config.output_dir / f"{stem}.json"
            image_tmp = image_path.with_suffix(".png.tmp")
            metadata_tmp = metadata_path.with_suffix(".json.tmp")
            image_tmp.write_bytes(image_bytes)
            metadata_tmp.write_bytes(metadata_bytes)
            os.replace(image_tmp, image_path)
            os.replace(metadata_tmp, metadata_path)
            with self._lock:
                self._stats["written"] += 1
        finally:
            with self._lock:
                self._reserved_bytes = max(0, self._reserved_bytes - required)

    def _directory_bytes(self) -> int:
        try:
            return sum(path.stat().st_size for path in self.config.output_dir.iterdir() if path.is_file())
        except OSError:
            return self.config.quota_bytes