"""Adapter between the wire protocol and the Architecture V2 pipeline.

``api.py`` imports ``predict`` and ``warmup_detector`` from here and owns the
transport; everything else lives in ``v2/``. Baseline 1's stateless
detect-and-answer version is preserved at commit ``129564f`` for rollback.
"""

from __future__ import annotations

import logging

from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from v2.pipeline import get_pipeline


logger = logging.getLogger(__name__)


def warmup_detector() -> None:
    """Load every model and run each one once, before the first scored frame."""
    get_pipeline().warmup()


def describe() -> dict:
    """What is actually loaded, for the health endpoint and deployment checks."""
    return get_pipeline().describe()


def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    """Answer one frame with the whole-frame state, not just this crop."""
    try:
        return get_pipeline().predict(request)
    except Exception:
        # An exception here means no response at all and zero detections for
        # this frame. An empty but valid answer is strictly better, and it
        # keeps the camera where it is rather than leaving it uncommanded.
        logger.exception('Pipeline failed on frame %s', request.frame)
        return DroneFlybyPredictResponseDto(
            request_id=request.request_id,
            frame=request.frame,
            annotations=[],
            requested_view=None,
        )
