import json
import socket
import sys
import tempfile
import time
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from urllib.request import Request as HttpRequest, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtos import OBJECT_CLASSES  # noqa: E402
from test_v2 import request_at  # noqa: E402
from utils import source_region_for_view  # noqa: E402
from v5.gpu.diagnostic_capture import CaptureConfig, DiagnosticCapture  # noqa: E402
from v5.gpu.pipeline_v6 import V6Pipeline  # noqa: E402


class FakeExpert:
    def classify(self, image, boxes):
        results = []
        features = []
        for _ in boxes:
            scores = np.full(16, 0.01, dtype=np.float32)
            scores[OBJECT_CLASSES.index("helicopter")] = 0.99
            results.append({"target_probability": 0.9, "class_scores": scores.tolist()})
            features.append(np.zeros(4, dtype=np.float32))
        return results, features, None


def make_pipeline(capture):
    pipeline = V6Pipeline.__new__(V6Pipeline)
    pipeline.cfg = SimpleNamespace(
        detector="synthetic", device="cpu", assets="synthetic", max_tracks=300,
        ttl=8, active_camera=True, min_emit_conf=0.05, classify_min_px=22,
    )
    pipeline.dev = "cpu"
    pipeline.expert = FakeExpert()
    pipeline.ensemble = None
    pipeline.states = OrderedDict()
    pipeline.lock = __import__("threading").RLock()
    pipeline.capture = capture
    pipeline.last_diagnostics = {}
    pipeline.manifest = {"pipeline": "v6", "diagnostic_capture": capture.enabled}
    pipeline._detect = lambda image, region: [{
        "local_box": [100.0, 100.0, 180.0, 160.0],
        "source_box": [region[0] + 100.0, region[1] + 100.0, region[0] + 180.0, region[1] + 160.0],
        "score": 0.8,
        "src": "synthetic",
    }]
    return pipeline


class V6DiagnosticIntegrationTests(unittest.TestCase):
    def test_capture_on_preserves_predictions_camera_commands_and_request_identity(self):
        image = np.zeros((540, 960, 3), dtype=np.uint8)
        image[20:40, 30:70] = (5, 9, 13)
        requests = [
            request_at(0, 1920, 1080, image=image, frame=0, frame_index=0, sequence_id="same-seq"),
            request_at(1, 1440, 810, image=image, frame=1, frame_index=1, sequence_id="same-seq"),
            request_at(2, 1440, 810, image=image, frame=2, frame_index=2, sequence_id="same-seq"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            off = make_pipeline(DiagnosticCapture(CaptureConfig(enabled=False)))
            on_capture = DiagnosticCapture(CaptureConfig(enabled=True, output_dir=Path(directory), queue_size=8))
            on = make_pipeline(on_capture)
            try:
                off_results = [off.predict(request) for request in requests]
                on_results = [on.predict(request) for request in requests]
            finally:
                on.close()

            self.assertEqual(
                [result.model_dump(mode="json") for result in off_results],
                [result.model_dump(mode="json") for result in on_results],
            )
            records = sorted(Path(directory).glob("*.json"))
            self.assertEqual(len(records), 2)
            for record_path in records:
                record = json.loads(record_path.read_text())
                self.assertEqual(record["sequence_id"], "same-seq")
                self.assertIn(record["request_id"], {"request-1", "request-2"})
                self.assertIn(record["received_camera"]["resolution_level"], (1, 2))
                self.assertEqual(record["image"]["width"], 960)
                self.assertEqual(record["image"]["height"], 540)
                self.assertEqual(len(record["candidates"]), 1)
                self.assertEqual(record["candidates"][0]["src"], "synthetic")
                level = record["received_camera"]["resolution_level"]
                expected_region = list(source_region_for_view(level, 1440, 810))
                self.assertEqual(record["received_camera"]["source_region_xyxy"], expected_region)
                self.assertEqual(record["candidates"][0]["source_box"], [
                    expected_region[0] + 100.0, expected_region[1] + 100.0,
                    expected_region[0] + 180.0, expected_region[1] + 160.0,
                ])
                image_path = record_path.with_suffix(".png")
                decoded = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertTrue(np.array_equal(decoded, image))
                result_index = int(record["request_id"].split("-")[1])
                self.assertEqual(record["predictions"], [
                    prediction.model_dump(mode="json") for prediction in on_results[result_index].annotations
                ])

    def test_integrated_pipeline_serves_synthetic_request_on_ephemeral_port(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = make_pipeline(DiagnosticCapture(CaptureConfig(
                enabled=True, output_dir=Path(directory), queue_size=4,
            )))
            app = FastAPI()

            @app.post("/predict")
            async def predict(request: Request):
                from dtos import DroneFlybyPredictRequestDto
                parsed = DroneFlybyPredictRequestDto.model_validate(await request.json())
                return pipeline.predict(parsed).model_dump(mode="json")

            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
            thread = __import__("threading").Thread(target=server.run, daemon=True)
            thread.start()
            try:
                deadline = time.monotonic() + 5.0
                while not server.started and time.monotonic() < deadline:
                    time.sleep(0.01)
                request = request_at(1, 1440, 810, frame=1, frame_index=1, sequence_id="http-v6")
                body = json.dumps(request.model_dump(mode="json")).encode("utf-8")
                response = urlopen(HttpRequest(
                    f"http://127.0.0.1:{port}/predict", data=body,
                    headers={"Content-Type": "application/json"}, method="POST",
                ))
                payload = json.loads(response.read())
                self.assertEqual(payload["request_id"], request.request_id)
                self.assertIn("annotations", payload)
            finally:
                server.should_exit = True
                thread.join(timeout=5.0)
                pipeline.close()
            self.assertTrue(server.started)
            self.assertEqual(pipeline.capture.stats()["written"], 1)


if __name__ == "__main__":
    unittest.main()