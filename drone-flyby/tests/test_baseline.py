"""Focused protocol and inference-boundary tests for the minimal baseline."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import example
from detector import RawDetection, detections_to_annotations
from dtos import (
    ALLOWED_RESOLUTION_LEVELS,
    FULL_FRAME_CENTER,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAXIMUM_CENTER_DELTA_PIXELS,
    OBJECT_CLASSES,
    SOURCE_REGION_SIZES,
    CameraConstraintsDto,
    CameraLevelBoundsDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyViewDto,
)
from utils import describe_camera_rejection, encode_image, source_region_for_view


def request_at(level: int, center_x: int, center_y: int) -> DroneFlybyPredictRequestDto:
    allowed = ALLOWED_RESOLUTION_LEVELS[level]
    bounds = []
    for target in allowed:
        width, height = SOURCE_REGION_SIZES[target]
        bounds.append(
            CameraLevelBoundsDto(
                resolution_level=target,
                width=width,
                height=height,
                minimum_center_x=width // 2,
                maximum_center_x=IMAGE_WIDTH - width // 2,
                minimum_center_y=height // 2,
                maximum_center_y=IMAGE_HEIGHT - height // 2,
            )
        )
    view = DroneFlybyViewDto(
        resolution_level=level,
        center_x=center_x,
        center_y=center_y,
        view_id="test-view",
        image=encode_image(np.zeros((540, 960, 3), dtype=np.uint8)),
        image_media_type="image/png",
        width=960,
        height=540,
        source_region_xyxy=list(source_region_for_view(level, center_x, center_y)),
    )
    return DroneFlybyPredictRequestDto(
        sequence_id="test-sequence",
        frame=0,
        frame_index=0,
        request_id="test-request",
        frame_interval_ms=333,
        response_timeout_ms=3333,
        original_width=IMAGE_WIDTH,
        original_height=IMAGE_HEIGHT,
        view=view,
        camera_constraints=CameraConstraintsDto(
            maximum_center_delta=MAXIMUM_CENTER_DELTA_PIXELS[level],
            allowed_resolution_levels=list(allowed),
            center_bounds=bounds,
            full_view_reset_exempt_from_delta=True,
        ),
    )


class BaselineTests(unittest.TestCase):
    def test_dataset_class_mapping_matches_dto_exactly(self):
        summary_path = ROOT / "training" / "artifacts" / "l0_data_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        mapping = tuple(summary["class_mapping"][str(index)] for index in range(16))
        self.assertEqual(mapping, OBJECT_CLASSES)
        self.assertEqual(summary["annotations"], 259)

    def test_model_output_conversion_clips_deduplicates_and_rejects_degenerate(self):
        request = request_at(1, *FULL_FRAME_CENTER)
        detection = RawDetection(3, 0.8, (0.0, 0.0, 960.0, 540.0))
        rows = [
            detection,
            detection,
            RawDetection(3, 0.5, (-20.0, -20.0, 0.0, 10.0)),
            RawDetection(99, 0.5, (10.0, 10.0, 20.0, 20.0)),
        ]
        annotations = detections_to_annotations(rows, (540, 960, 3), request)
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0].object_id, "large_launcher")
        self.assertEqual(list(annotations[0].bbox), [0.25, 0.25, 0.75, 0.75])

    def test_always_l0_policy_holds_or_steps_legally(self):
        level0 = request_at(0, *FULL_FRAME_CENTER)
        self.assertIsNone(example.choose_next_view(level0))

        level1 = request_at(1, *FULL_FRAME_CENTER)
        command1 = example.choose_next_view(level1)
        self.assertEqual(command1.resolution_level, 0)
        self.assertIsNone(
            describe_camera_rejection(
                1,
                (level1.view.center_x, level1.view.center_y),
                command1.resolution_level,
                (command1.center_x, command1.center_y),
            )
        )

        level2 = request_at(2, 480, 270)
        command2 = example.choose_next_view(level2)
        self.assertEqual(command2.resolution_level, 1)
        self.assertIsNone(
            describe_camera_rejection(
                2,
                (level2.view.center_x, level2.view.center_y),
                command2.resolution_level,
                (command2.center_x, command2.center_y),
            )
        )

    def test_detector_exception_returns_valid_empty_response(self):
        request = request_at(0, *FULL_FRAME_CENTER)
        with mock.patch.object(example, "detect", side_effect=RuntimeError("test failure")):
            response = example.predict(request)
        self.assertEqual(response.request_id, request.request_id)
        self.assertEqual(response.frame, request.frame)
        self.assertEqual(response.annotations, [])
        self.assertIsNone(response.requested_view)

    def test_every_emitted_annotation_is_protocol_valid(self):
        request = request_at(0, *FULL_FRAME_CENTER)
        rows = [
            RawDetection(index, 1.2, (10.0 + index, 20.0, 30.0 + index, 50.0))
            for index in range(len(OBJECT_CLASSES))
        ]
        annotations = detections_to_annotations(rows, (540, 960, 3), request)
        self.assertEqual(len(annotations), len(OBJECT_CLASSES))
        for annotation in annotations:
            x1, y1, x2, y2 = annotation.bbox
            self.assertTrue(0 <= x1 < x2 <= 1)
            self.assertTrue(0 <= y1 < y2 <= 1)
            self.assertTrue(0 <= annotation.confidence <= 1)


if __name__ == "__main__":
    unittest.main()
