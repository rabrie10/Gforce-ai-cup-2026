"""Stateless current-frame detector and always-Level-0 camera policy."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from detector import detections_to_annotations, get_detector
from dtos import (
    FULL_FRAME_CENTER,
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
    RequestedViewDto,
)
from utils import decode_view


logger = logging.getLogger(__name__)


def detect(
    image: np.ndarray,
    request: DroneFlybyPredictRequestDto,
) -> list[DroneFlybyPredictionDto]:
    return detections_to_annotations(get_detector().infer(image), image.shape, request)


def choose_next_view(request: DroneFlybyPredictRequestDto) -> Optional[RequestedViewDto]:
    """Hold L0, or legally step L2 to L1 to L0 using supplied constraints."""
    current = request.view
    constraints = request.camera_constraints
    if current.resolution_level == 0:
        return None

    if current.resolution_level == 1 and 0 in constraints.allowed_resolution_levels:
        if constraints.bounds_for_level(0) is None:
            return None
        return RequestedViewDto(
            resolution_level=0,
            center_x=FULL_FRAME_CENTER[0],
            center_y=FULL_FRAME_CENTER[1],
        )

    # L2 cannot jump directly to L0. Keeping the current center where possible
    # and clamping to L1 bounds moves at most 550.73 px, inside the L2 limit.
    if current.resolution_level == 2 and 1 in constraints.allowed_resolution_levels:
        bounds = constraints.bounds_for_level(1)
        if bounds is None:
            return None
        center_x = min(max(current.center_x, bounds.minimum_center_x), bounds.maximum_center_x)
        center_y = min(max(current.center_y, bounds.minimum_center_y), bounds.maximum_center_y)
        return RequestedViewDto(resolution_level=1, center_x=center_x, center_y=center_y)
    return None


def warmup_detector() -> None:
    get_detector().warmup()


def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    """Return detections from only the current observation; never temporal state."""
    try:
        image = decode_view(request.view)
        annotations = detect(image, request)
    except Exception:
        logger.exception("Detector failed on frame %s", request.frame)
        annotations = []
    return DroneFlybyPredictResponseDto(
        request_id=request.request_id,
        frame=request.frame,
        annotations=annotations,
        requested_view=choose_next_view(request),
    )
