"""Image-derived global motion compensation, in source coordinates.

D-007 selected **affine** as the first image-derived GMC target, with similarity
kept as the lower-complexity fallback and independent constant velocity below
that. Homography is deliberately not implemented; reopening that decision needs
a new measured problem, not a new idea.

The one structural trick here is that the *known* part of the geometry is
removed analytically instead of being left for the estimator to discover. Two
consecutive requests may arrive from different camera levels and centres, and a
feature matcher asked to absorb a 2x zoom on top of the scene motion will simply
fail. So each received image is first resampled into a shared low-resolution
source-referenced canvas using its own ``source_region_xyxy``; whatever motion
is left in that canvas is scene motion, and nothing else.

The estimate is then either trusted completely or discarded completely. A
half-trusted global transform applied at reduced weight is the mechanism by
which one bad frame corrupts every track at once, so the quality gates are
hard gates.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH
from v2.config import CONFIG, GmcConfig
from v2.geometry import IDENTITY_AFFINE, decompose_affine, rescale_affine


logger = logging.getLogger(__name__)


@dataclass
class MotionEstimate:
    """A source-space affine plus everything needed to judge it."""

    matrix: np.ndarray = field(default_factory=lambda: IDENTITY_AFFINE.copy())
    model: str = 'identity'
    ok: bool = False
    quality: float = 0.0
    inliers: int = 0
    candidates: int = 0
    inlier_ratio: float = 0.0
    reason: str = 'no-previous-frame'
    elapsed_ms: float = 0.0

    def summary(self) -> dict:
        return {
            'model': self.model,
            'ok': self.ok,
            'quality': round(self.quality, 4),
            'inliers': self.inliers,
            'candidates': self.candidates,
            'inlier_ratio': round(self.inlier_ratio, 4),
            'reason': self.reason,
            'elapsed_ms': round(self.elapsed_ms, 2),
            'translation': [round(float(self.matrix[0, 2]), 2), round(float(self.matrix[1, 2]), 2)],
        }


class GlobalMotionEstimator:
    """Frame-to-frame shared motion with an explicit fallback ladder."""

    def __init__(self, config: Optional[GmcConfig] = None) -> None:
        self.config = config or CONFIG.gmc
        self.canvas_scale = self.config.work_width / float(IMAGE_WIDTH)
        self.canvas_size = (
            int(round(IMAGE_WIDTH * self.canvas_scale)),
            int(round(IMAGE_HEIGHT * self.canvas_scale)),
        )
        self._previous_canvas: Optional[np.ndarray] = None
        self._previous_mask: Optional[np.ndarray] = None

    # -- canvas ------------------------------------------------------------- #

    def _to_canvas(
        self, image: np.ndarray, region: Sequence[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Place a received image into the shared source-referenced canvas."""
        canvas = np.zeros((self.canvas_size[1], self.canvas_size[0]), dtype=np.uint8)
        mask = np.zeros_like(canvas)

        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        x1 = int(round(region[0] * self.canvas_scale))
        y1 = int(round(region[1] * self.canvas_scale))
        x2 = int(round(region[2] * self.canvas_scale))
        y2 = int(round(region[3] * self.canvas_scale))
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(self.canvas_size[0], x2)
        y2 = min(self.canvas_size[1], y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return canvas, mask

        interpolation = (
            cv2.INTER_AREA if (x2 - x1) < grey.shape[1] else cv2.INTER_LINEAR
        )
        canvas[y1:y2, x1:x2] = cv2.resize(
            grey, (x2 - x1, y2 - y1), interpolation=interpolation
        )
        mask[y1:y2, x1:x2] = 255
        return canvas, mask

    # -- gates -------------------------------------------------------------- #

    def _accept(self, matrix: np.ndarray, model: str) -> Tuple[bool, str]:
        """Reject any transform a 3 fps survey flight could not have produced."""
        if matrix is None or not np.isfinite(matrix).all():
            return False, 'non-finite-transform'
        parts = decompose_affine(matrix)
        if parts['scale_x'] <= 1e-6 or parts['scale_y'] <= 1e-6:
            return False, 'degenerate-scale'
        for axis in ('scale_x', 'scale_y'):
            if abs(parts[axis] - 1.0) > self.config.max_scale_deviation:
                return False, f'{axis}-out-of-range'
        if model == 'affine' and abs(parts['shear']) > self.config.max_shear:
            return False, 'shear-out-of-range'
        limit = self.config.max_translation_fraction * self.canvas_size[0]
        if abs(parts['translation'][0]) > limit or abs(parts['translation'][1]) > limit:
            return False, 'translation-out-of-range'
        return True, 'accepted'

    # -- estimation --------------------------------------------------------- #

    def estimate(
        self, image: np.ndarray, region: Sequence[float]
    ) -> MotionEstimate:
        """Return the source-space motion from the previous frame to this one.

        The previous canvas is always replaced, whether or not the estimate is
        accepted, so one unusable frame costs one frame rather than stranding
        the estimator on a stale reference.
        """
        started = time.perf_counter()
        canvas, mask = self._to_canvas(image, region)
        previous_canvas, previous_mask = self._previous_canvas, self._previous_mask
        self._previous_canvas, self._previous_mask = canvas, mask

        if not self.config.enabled:
            return MotionEstimate(reason='disabled', elapsed_ms=self._ms(started))
        if previous_canvas is None:
            return MotionEstimate(reason='no-previous-frame', elapsed_ms=self._ms(started))

        overlap = cv2.bitwise_and(previous_mask, mask)
        if int(np.count_nonzero(overlap)) < 400:
            return MotionEstimate(reason='insufficient-overlap', elapsed_ms=self._ms(started))

        corners = cv2.goodFeaturesToTrack(
            previous_canvas,
            maxCorners=self.config.max_corners,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance,
            mask=overlap,
            blockSize=3,
        )
        if corners is None or len(corners) < self.config.min_inliers:
            return MotionEstimate(
                reason='insufficient-features',
                candidates=0 if corners is None else len(corners),
                elapsed_ms=self._ms(started),
            )

        # The pyramid has to span the largest displacement the canvas can
        # show. A 3 fps survey flight moves about 70 source pixels per frame,
        # which is only ~12 canvas pixels at the default width - but a frame
        # arriving after several skipped ones moves several times that, and a
        # pyramid too shallow for it does not degrade, it simply fails.
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_canvas,
            canvas,
            corners,
            None,
            winSize=(21, 21),
            maxLevel=self.config.pyramid_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if tracked is None or status is None:
            return MotionEstimate(reason='flow-failed', elapsed_ms=self._ms(started))

        keep = status.reshape(-1) == 1
        source_points = corners.reshape(-1, 2)[keep]
        target_points = tracked.reshape(-1, 2)[keep]
        candidates = int(len(source_points))
        if candidates < self.config.min_inliers:
            return MotionEstimate(
                reason='too-few-tracked-points',
                candidates=candidates,
                elapsed_ms=self._ms(started),
            )

        # Only points inside the overlap of both masks can carry real motion;
        # a point that lands on canvas padding is measuring an image border.
        inside = self._inside_mask(target_points, mask)
        source_points, target_points = source_points[inside], target_points[inside]
        candidates = int(len(source_points))
        if candidates < self.config.min_inliers:
            return MotionEstimate(
                reason='points-left-overlap',
                candidates=candidates,
                elapsed_ms=self._ms(started),
            )

        ladder = (
            ('affine', cv2.estimateAffine2D),
            ('similarity', cv2.estimateAffinePartial2D),
        )
        if self.config.model == 'similarity':
            ladder = ladder[1:]

        for model, estimator in ladder:
            matrix, inlier_flags = estimator(
                source_points,
                target_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_threshold,
                maxIters=2000,
                confidence=0.995,
            )
            if matrix is None or inlier_flags is None:
                continue
            inliers = int(inlier_flags.sum())
            ratio = inliers / max(1, candidates)
            if inliers < self.config.min_inliers or ratio < self.config.min_inlier_ratio:
                continue
            accepted, reason = self._accept(matrix, model)
            if not accepted:
                continue
            return MotionEstimate(
                matrix=rescale_affine(matrix, 1.0 / self.canvas_scale),
                model=model,
                ok=True,
                quality=float(ratio * min(1.0, inliers / 60.0)),
                inliers=inliers,
                candidates=candidates,
                inlier_ratio=float(ratio),
                reason=reason,
                elapsed_ms=self._ms(started),
            )

        # Last resort inside the image path: a robust median translation. It is
        # reported as its own model so telemetry can tell it apart from a real
        # affine fit, and so the confidence policy can discount it.
        deltas = target_points - source_points
        median = np.median(deltas, axis=0)
        spread = float(np.median(np.abs(deltas - median)))
        if spread <= 2.0:
            matrix = np.array(
                [[1.0, 0.0, float(median[0])], [0.0, 1.0, float(median[1])]],
                dtype=np.float64,
            )
            accepted, reason = self._accept(matrix, 'translation')
            if accepted:
                return MotionEstimate(
                    matrix=rescale_affine(matrix, 1.0 / self.canvas_scale),
                    model='translation',
                    ok=True,
                    quality=0.35,
                    inliers=candidates,
                    candidates=candidates,
                    inlier_ratio=1.0,
                    reason=reason,
                    elapsed_ms=self._ms(started),
                )

        return MotionEstimate(
            reason='all-models-rejected',
            candidates=candidates,
            elapsed_ms=self._ms(started),
        )

    @staticmethod
    def _inside_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
        columns = np.clip(np.round(points[:, 0]).astype(int), 0, mask.shape[1] - 1)
        rows = np.clip(np.round(points[:, 1]).astype(int), 0, mask.shape[0] - 1)
        return mask[rows, columns] > 0

    @staticmethod
    def _ms(started: float) -> float:
        return (time.perf_counter() - started) * 1000.0

    def reset(self) -> None:
        self._previous_canvas = None
        self._previous_mask = None
