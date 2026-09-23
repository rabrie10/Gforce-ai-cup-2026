"""Coordinate, box and transform maths shared by the pipeline and the probes.

Everything the V2 pipeline reasons about lives in **source pixels**
(0..3840 x 0..2160). Received-image coordinates are converted on the way in and
normalized frame-global coordinates on the way out, and nowhere else. That one
rule removes the whole class of "local crop box returned as a global box" bugs
the protocol notes warn about.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH, SOURCE_REGION_SIZES, TRANSMITTED_VIEW_SIZE


Box = Tuple[float, float, float, float]

# Source pixels per received pixel, by resolution level.
LEVEL_DOWNSCALE = {0: 4.0, 1: 2.0, 2: 1.0}


# --------------------------------------------------------------------------- #
# Boxes
# --------------------------------------------------------------------------- #

def box_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_center(box: Sequence[float]) -> Tuple[float, float]:
    return (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))


def box_short_side(box: Sequence[float]) -> float:
    return min(box[2] - box[0], box[3] - box[1])


def box_diagonal(box: Sequence[float]) -> float:
    return math.hypot(box[2] - box[0], box[3] - box[1])


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union. The evaluator's matching rule at 0.50."""
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = box_area(a) + box_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def contains(outer: Sequence[float], inner: Sequence[float]) -> bool:
    """True when ``inner`` lies completely inside ``outer``."""
    return (
        inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def clamp_box_to_frame(box: Sequence[float]) -> Optional[Box]:
    """Clip a source-pixel box to the frame, or None if nothing survives."""
    x1 = max(0.0, min(float(IMAGE_WIDTH), float(box[0])))
    x2 = max(0.0, min(float(IMAGE_WIDTH), float(box[2])))
    y1 = max(0.0, min(float(IMAGE_HEIGHT), float(box[1])))
    y2 = max(0.0, min(float(IMAGE_HEIGHT), float(box[3])))
    if x2 - x1 < 1e-3 or y2 - y1 < 1e-3:
        return None
    return (x1, y1, x2, y2)


def expand_box(box: Sequence[float], factor: float, minimum_side: float = 0.0) -> Box:
    """Scale a box about its centre, optionally enforcing a minimum side."""
    cx, cy = box_center(box)
    half_width = max(0.5 * (box[2] - box[0]) * factor, 0.5 * minimum_side)
    half_height = max(0.5 * (box[3] - box[1]) * factor, 0.5 * minimum_side)
    return (cx - half_width, cy - half_height, cx + half_width, cy + half_height)


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #

def source_region(level: int, center_x: int, center_y: int) -> Box:
    """The source rectangle a camera position covers."""
    width, height = SOURCE_REGION_SIZES[level]
    return (
        float(center_x - width // 2),
        float(center_y - height // 2),
        float(center_x + width // 2),
        float(center_y + height // 2),
    )


def view_to_source(box: Sequence[float], region: Sequence[float]) -> Box:
    """Map a received-image pixel box into source pixels.

    ``box`` is in the 960x540 received image; ``region`` is
    ``request.view.source_region_xyxy``.
    """
    scale_x = (region[2] - region[0]) / TRANSMITTED_VIEW_SIZE[0]
    scale_y = (region[3] - region[1]) / TRANSMITTED_VIEW_SIZE[1]
    return (
        region[0] + box[0] * scale_x,
        region[1] + box[1] * scale_y,
        region[0] + box[2] * scale_x,
        region[1] + box[3] * scale_y,
    )


def source_to_view(box: Sequence[float], region: Sequence[float]) -> Box:
    """Inverse of :func:`view_to_source`."""
    scale_x = TRANSMITTED_VIEW_SIZE[0] / (region[2] - region[0])
    scale_y = TRANSMITTED_VIEW_SIZE[1] / (region[3] - region[1])
    return (
        (box[0] - region[0]) * scale_x,
        (box[1] - region[1]) * scale_y,
        (box[2] - region[0]) * scale_x,
        (box[3] - region[1]) * scale_y,
    )


def source_to_global(box: Sequence[float], width: int, height: int) -> Box:
    """Normalize a source-pixel box against the full frame, for the response."""
    return (box[0] / width, box[1] / height, box[2] / width, box[3] / height)


def clamp_center(level: int, center_x: float, center_y: float) -> Tuple[int, int]:
    """Clamp a desired centre into the legal range for a level, as ints."""
    width, height = SOURCE_REGION_SIZES[level]
    minimum_x, maximum_x = width // 2, IMAGE_WIDTH - width // 2
    minimum_y, maximum_y = height // 2, IMAGE_HEIGHT - height // 2
    x = int(round(min(max(center_x, minimum_x), maximum_x)))
    y = int(round(min(max(center_y, minimum_y), maximum_y)))
    return x, y


# --------------------------------------------------------------------------- #
# Rendering a level view (probe-side only; the evaluator does this for us)
# --------------------------------------------------------------------------- #

def render_level_view(frame: np.ndarray, level: int, center_x: int, center_y: int) -> np.ndarray:
    """Reproduce, bit for bit, the 960x540 image the evaluator would transmit.

    Used by the offline probes and by the gallery builder so that reference
    crops carry exactly the detail loss a live crop carries. INTER_AREA at an
    exact integer factor is plain box averaging, which is what the harness does.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in source_region(level, center_x, center_y))
    view = frame[y1:y2, x1:x2]
    if (view.shape[1], view.shape[0]) != TRANSMITTED_VIEW_SIZE:
        view = cv2.resize(view, TRANSMITTED_VIEW_SIZE, interpolation=cv2.INTER_AREA)
    return view


def render_object_at_level(
    frame: np.ndarray,
    box: Sequence[float],
    level: int,
    crop_size: int,
    context: float,
) -> Optional[np.ndarray]:
    """Render one object the way the recognizer would see it at ``level``.

    The source region is downsampled by the level's factor first and only then
    resized to the recognizer's input, so the crop carries the information the
    camera would actually have transmitted. Upsampling afterwards cannot invent
    detail that the level never carried, which is exactly the property the
    oracle resolution probe is measuring.
    """
    downscale = LEVEL_DOWNSCALE[level]
    region = expand_box(box, context, minimum_side=downscale * 2.0)
    x1, y1, x2, y2 = (int(round(v)) for v in region)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    patch = frame[y1:y2, x1:x2]

    if downscale > 1.0:
        target = (
            max(1, int(round(patch.shape[1] / downscale))),
            max(1, int(round(patch.shape[0] / downscale))),
        )
        patch = cv2.resize(patch, target, interpolation=cv2.INTER_AREA)

    interpolation = cv2.INTER_AREA if patch.shape[0] > crop_size else cv2.INTER_LINEAR
    return cv2.resize(patch, (crop_size, crop_size), interpolation=interpolation)


def crop_from_view(
    view: np.ndarray,
    box_in_view: Sequence[float],
    crop_size: int,
    context: float,
) -> Optional[np.ndarray]:
    """Cut an object-centred square out of a received image, with context.

    Border-replicated rather than clipped, so an object near the crop edge
    still produces a well-formed, centred query instead of a lopsided one.
    """
    cx, cy = box_center(box_in_view)
    half = 0.5 * max(
        box_in_view[2] - box_in_view[0], box_in_view[3] - box_in_view[1]
    ) * context
    half = max(half, 3.0)
    x1, y1 = int(math.floor(cx - half)), int(math.floor(cy - half))
    x2, y2 = int(math.ceil(cx + half)), int(math.ceil(cy + half))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    pad_left, pad_top = max(0, -x1), max(0, -y1)
    pad_right = max(0, x2 - view.shape[1])
    pad_bottom = max(0, y2 - view.shape[0])
    patch = view[max(0, y1):min(view.shape[0], y2), max(0, x1):min(view.shape[1], x2)]
    if patch.size == 0:
        return None
    if pad_left or pad_top or pad_right or pad_bottom:
        patch = cv2.copyMakeBorder(
            patch, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE
        )
    interpolation = cv2.INTER_AREA if patch.shape[0] > crop_size else cv2.INTER_LINEAR
    return cv2.resize(patch, (crop_size, crop_size), interpolation=interpolation)


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #

IDENTITY_AFFINE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def apply_affine_to_box(matrix: np.ndarray, box: Sequence[float]) -> Box:
    """Push a box through a 2x3 affine, keeping it axis-aligned.

    All four corners are transformed and re-bounded rather than only the centre,
    so a transform carrying rotation or anisotropic scale changes the box shape
    the way it changes the object.
    """
    corners = np.array(
        [
            [box[0], box[1], 1.0],
            [box[2], box[1], 1.0],
            [box[2], box[3], 1.0],
            [box[0], box[3], 1.0],
        ],
        dtype=np.float64,
    )
    mapped = corners @ np.asarray(matrix, dtype=np.float64).T
    return (
        float(mapped[:, 0].min()),
        float(mapped[:, 1].min()),
        float(mapped[:, 0].max()),
        float(mapped[:, 1].max()),
    )


def apply_affine_to_point(matrix: np.ndarray, point: Sequence[float]) -> Tuple[float, float]:
    vector = np.array([point[0], point[1], 1.0], dtype=np.float64)
    mapped = np.asarray(matrix, dtype=np.float64) @ vector
    return float(mapped[0]), float(mapped[1])


def rescale_affine(matrix: np.ndarray, scale: float) -> np.ndarray:
    """Re-express an affine estimated at one pixel scale in another.

    A transform fitted on a downscaled working image has the same linear part
    but a translation measured in working pixels; moving to source pixels is a
    conjugation by a uniform scaling, which only touches the translation column.
    """
    out = np.array(matrix, dtype=np.float64, copy=True)
    out[:, 2] *= scale
    return out


def compose_affine(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Return the affine equivalent to applying ``inner`` then ``outer``."""
    a = np.vstack([np.asarray(outer, dtype=np.float64), [0.0, 0.0, 1.0]])
    b = np.vstack([np.asarray(inner, dtype=np.float64), [0.0, 0.0, 1.0]])
    return (a @ b)[:2, :]


def affine_from_similarity(matrix: np.ndarray) -> np.ndarray:
    """Widen a 2x3 similarity estimate into the affine representation."""
    return np.asarray(matrix, dtype=np.float64).reshape(2, 3)


def decompose_affine(matrix: np.ndarray) -> dict:
    """Report the scale, shear and rotation a 2x3 affine carries.

    The quality gates need interpretable quantities rather than raw matrix
    entries: a 3 fps survey flight at fixed altitude cannot plausibly scale by
    30% or shear noticeably between two consecutive frames, so a fit that says
    it did is a bad fit rather than a surprising flight.
    """
    linear = np.asarray(matrix, dtype=np.float64)[:, :2]
    determinant = float(np.linalg.det(linear))
    scale_x = float(math.hypot(linear[0, 0], linear[1, 0]))
    scale_y_numerator = determinant
    scale_y = float(scale_y_numerator / scale_x) if scale_x > 1e-9 else 0.0
    shear = (
        float((linear[0, 0] * linear[0, 1] + linear[1, 0] * linear[1, 1]) / (scale_x ** 2))
        if scale_x > 1e-9
        else 0.0
    )
    rotation = float(math.atan2(linear[1, 0], linear[0, 0]))
    return {
        'determinant': determinant,
        'scale_x': scale_x,
        'scale_y': scale_y,
        'shear': shear,
        'rotation': rotation,
        'translation': (float(matrix[0, 2]), float(matrix[1, 2])),
    }


def merge_boxes(boxes: Iterable[Sequence[float]], weights: Optional[Sequence[float]] = None) -> Box:
    """Confidence-weighted average of overlapping boxes."""
    rows = np.array([list(box) for box in boxes], dtype=np.float64)
    if weights is None:
        return tuple(rows.mean(axis=0))
    weight_vector = np.asarray(weights, dtype=np.float64)
    total = weight_vector.sum()
    if total <= 0:
        return tuple(rows.mean(axis=0))
    return tuple((rows * weight_vector[:, None]).sum(axis=0) / total)
