import json
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request as HttpRequest, urlopen
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtos import DroneFlybyPredictRequestDto  # noqa: E402
from utils import encode_image, source_region_for_view  # noqa: E402
from v5.gpu.diagnostic_capture import CaptureConfig, DiagnosticCapture  # noqa: E402
from test_v2 import constraints_for, request_at  # noqa: E402


class DiagnosticCaptureTests(unittest.TestCase):
    def request(self, level=1, frame=7, sequence_id="seq-a") -> DroneFlybyPredictRequestDto:
        request = request_at(level, 1440 if level else 1920, 810 if level else 1080,
                             frame=frame, frame_index=frame, sequence_id=sequence_id)
        request.request_id = f"{sequence_id}:{frame}:{level}"
        return request

    def wait_for(self, predicate, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        self.fail("diagnostic writer did not finish")

    def test_disabled_capture_is_inert(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = DiagnosticCapture(CaptureConfig(enabled=False, output_dir=Path(directory)))
            image = np.full((540, 960, 3), 17, dtype=np.uint8)
            self.assertFalse(capture.submit(self.request(), image, [], []))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_enabled_capture_preserves_pixels_metadata_candidates_and_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = DiagnosticCapture(CaptureConfig(enabled=True, output_dir=Path(directory), queue_size=2))
            image = np.zeros((540, 960, 3), dtype=np.uint8)
            image[12:34, 56:89] = (3, 7, 11)
            request = self.request(level=2, frame=12)
            candidates = [{"local_box": [1, 2, 30, 40], "source_box": [100, 200, 220, 360], "score": 0.75}]
            predictions = [{"object_id": "helicopter", "bbox": [0.1, 0.2, 0.3, 0.4], "confidence": 0.8}]
            self.assertTrue(capture.submit(request, image, candidates, predictions))
            capture.close()
            self.wait_for(lambda: capture.stats()["written"] == 1)
            json_path = next(Path(directory).glob("*.json"))
            image_path = next(Path(directory).glob("*.png"))
            metadata = json.loads(json_path.read_text())
            decoded = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertTrue(np.array_equal(decoded, image))
            self.assertEqual(metadata["sequence_id"], "seq-a")
            self.assertEqual(metadata["request_id"], request.request_id)
            self.assertEqual(metadata["frame_index"], 12)
            self.assertEqual(metadata["received_camera"]["resolution_level"], 2)
            self.assertEqual(metadata["received_camera"]["source_region_xyxy"], list(source_region_for_view(2, 1440, 810)))
            self.assertEqual(metadata["image"]["width"], 960)
            self.assertEqual(metadata["image"]["height"], 540)
            self.assertEqual(metadata["candidates"], candidates)
            self.assertEqual(metadata["predictions"], predictions)

    def test_limits_each_level_and_drops_queue_without_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = DiagnosticCapture(CaptureConfig(enabled=True, output_dir=Path(directory), queue_size=1, per_level_limit=10))
            started = threading.Event()
            release = threading.Event()
            def blocked_writer(item):
                started.set()
                release.wait(2.0)
            capture._write = blocked_writer
            image = np.zeros((540, 960, 3), dtype=np.uint8)
            self.assertTrue(capture.submit(self.request(level=1, frame=1), image, [], []))
            self.assertTrue(started.wait(1.0))
            self.assertTrue(capture.submit(self.request(level=1, frame=2), image, [], []))
            self.assertFalse(capture.submit(self.request(level=1, frame=3), image, [], []))
            release.set()
            capture.close()
            self.assertGreaterEqual(capture.stats()["dropped_queue"], 1)

    def test_disk_quota_drops_capture_without_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = DiagnosticCapture(CaptureConfig(enabled=True, output_dir=Path(directory), quota_bytes=1))
            image = np.zeros((540, 960, 3), dtype=np.uint8)
            self.assertTrue(capture.submit(self.request(), image, [], []))
            capture.close()
            self.assertEqual(capture.stats()["written"], 0)
            self.assertGreaterEqual(capture.stats()["dropped_quota"], 1)

    def test_synthetic_request_over_local_ephemeral_http_port(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = DiagnosticCapture(CaptureConfig(enabled=True, output_dir=Path(directory)))
            request = self.request(level=1, frame=31, sequence_id="http-seq")
            image = np.full((540, 960, 3), 23, dtype=np.uint8)

            class Handler(BaseHTTPRequestHandler):
                def do_POST(self):
                    accepted = capture.submit(request, image, [{"score": 0.4}], [])
                    self.send_response(200 if accepted else 503)
                    self.end_headers()
                    self.wfile.write(b"ok")

                def log_message(self, *_args):
                    return

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                response = urlopen(HttpRequest(f"http://127.0.0.1:{server.server_port}/predict", method="POST"))
                self.assertEqual(response.status, 200)
            finally:
                server.shutdown()
                thread.join(timeout=2.0)
                server.server_close()
                capture.close()
            self.assertEqual(capture.stats()["accepted"], 1)
            self.assertEqual(capture.stats()["written"], 1)


if __name__ == "__main__":
    unittest.main()