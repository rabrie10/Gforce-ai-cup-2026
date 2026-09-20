"""Non-blocking validation capture.

The official specification says the Validation sequence may be recorded and
kept, and Baseline 1's audit was crippled by the absence of exactly this data:
with no hosted GT and no request trace, class-level, size-level and
localization-level attribution of a 0.0016 score were simply *not observable*.
This module exists so that the next hosted attempt produces a forensic record
instead of a single number.

Every design choice here serves one constraint: **it must not cost a frame.**
Frames arrive every 333 ms and a slow answer is a skipped frame scored as no
detections, so the capture path is a bounded queue and one daemon writer
thread. When the queue is full, records are dropped and counted; nothing waits.
Any failure inside the writer is swallowed and counted, because a telemetry bug
must never be able to invalidate a response.

Off by default. ``DRONE_TELEMETRY=1`` switches it on.
"""

from __future__ import annotations

import base64
import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from v2.config import CONFIG, TelemetryConfig


logger = logging.getLogger(__name__)


@dataclass
class TelemetryStats:
    accepted: int = 0
    dropped: int = 0
    written: int = 0
    errors: int = 0

    def snapshot(self) -> dict:
        return {
            'accepted': self.accepted,
            'dropped': self.dropped,
            'written': self.written,
            'errors': self.errors,
        }


class TelemetryRecorder:
    """A bounded queue and a background writer. Never blocks ``/predict``."""

    def __init__(self, config: Optional[TelemetryConfig] = None) -> None:
        self.config = config or CONFIG.telemetry
        self.stats = TelemetryStats()
        self._queue: Optional[queue.Queue] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._session: Optional[Path] = None
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        if not self.config.enabled or self._thread is not None:
            return
        try:
            root = Path(self.config.directory)
            requested = ''.join(
                char if char.isalnum() or char in '-_' else '_'
                for char in self.config.session_id.strip()
            )
            prefix = f'{requested}_' if requested else 'run_'
            self._session = root / (
                prefix + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
            )
            (self._session / 'frames').mkdir(parents=True, exist_ok=True)
            if self.config.store_images:
                (self._session / 'images').mkdir(parents=True, exist_ok=True)
            self._queue = queue.Queue(maxsize=max(8, self.config.queue_size))
            self._thread = threading.Thread(
                target=self._run, name='drone-telemetry', daemon=True
            )
            self._thread.start()
            logger.info('Telemetry capture started at %s', self._session)
        except Exception:
            # A capture path that cannot start must not stop the service.
            logger.exception('Telemetry could not start; continuing without it')
            self._queue = None
            self._thread = None

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._queue is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    @property
    def active(self) -> bool:
        return self._queue is not None and self._thread is not None

    # -- capture ------------------------------------------------------------ #

    def record(self, payload: Dict[str, Any], image_b64: Optional[str] = None) -> None:
        """Enqueue one frame record. Returns immediately, always."""
        if self._queue is None:
            return
        try:
            keep_image = (
                self.config.store_images
                and image_b64 is not None
                and self._should_store_image(payload)
            )
            item = (payload, image_b64 if keep_image else None)
            self._queue.put_nowait(item)
            with self._lock:
                self.stats.accepted += 1
        except queue.Full:
            with self._lock:
                self.stats.dropped += 1
        except Exception:
            with self._lock:
                self.stats.errors += 1

    def _should_store_image(self, payload: Dict[str, Any]) -> bool:
        stride = max(1, int(self.config.image_stride))
        if stride == 1:
            return True
        try:
            index = int(payload.get('frame_index', payload.get('frame', 0)))
        except (TypeError, ValueError):
            return False
        return index % stride == 0

    # -- writer ------------------------------------------------------------- #

    def _run(self) -> None:
        assert self._queue is not None and self._session is not None
        while not (self._stop.is_set() and self._queue.empty()):
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            payload, image_b64 = item
            try:
                self._write(payload, image_b64)
                with self._lock:
                    self.stats.written += 1
            except Exception:
                with self._lock:
                    self.stats.errors += 1
                logger.debug('Telemetry write failed', exc_info=True)
            finally:
                self._queue.task_done()

    def _write(self, payload: Dict[str, Any], image_b64: Optional[str]) -> None:
        assert self._session is not None
        frame_index = payload.get('frame_index', payload.get('frame', 0))
        request_id = ''.join(
            char if char.isalnum() or char in '-_' else '_'
            for char in str(payload.get('request_id', ''))
        )[-40:]
        sequence_id = ''.join(
            char if char.isalnum() or char in '-_' else '_'
            for char in str(payload.get('sequence_id', ''))
        )[-24:]
        suffix = '_'.join(part for part in (sequence_id, request_id) if part)
        name = f'{int(frame_index):05d}' + (f'_{suffix}' if suffix else '')
        path = self._session / 'frames' / f'{name}.json'
        path.write_text(
            json.dumps(payload, default=str, separators=(',', ':')), encoding='utf-8'
        )
        if image_b64:
            # Written as received. No ground truth is invented, and nothing
            # from the request's credentials or headers reaches this directory.
            (self._session / 'images' / f'{name}.png').write_bytes(
                base64.b64decode(image_b64)
            )

    def snapshot(self) -> dict:
        with self._lock:
            stats = self.stats.snapshot()
        return {
            'enabled': self.config.enabled,
            'active': self.active,
            'session': str(self._session) if self._session else None,
            **stats,
        }
