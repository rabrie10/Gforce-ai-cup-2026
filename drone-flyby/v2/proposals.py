"""Class-agnostic discovery: where should the camera look, not what is there.

Baseline 1 established that forcing a 16-way decision at L0, where most objects
are between two and fifteen pixels across, produces neither identity nor
localization: all sixty GT appearances under 8 px at L0 went completely
unlocalized. Discovery is therefore separated from identity. This module only
answers "something object-like is here", and is tuned for **recall under a
bounded proposal budget**, never for local 16-way AP.

Two backends, unioned by default:

``yolo`` (default)
    An existing detector with its class head ignored and its boxes pooled into
    a single objectness class. Localization was the part of Baseline 1 that
    worked - 77/78 correct at >=16 px - so it is a usable source of proposals
    even though its identity output is not trusted at all.

``saliency`` (measured and rejected as a default)
    A learning-free multi-scale local-contrast band-pass. Attractive in
    principle because nothing fitted it to this scene and it therefore cannot
    memorize one. **Probe C measured its centre recall at 0.046 (L0) and 0.024
    (L1) with an 80-proposal budget** - it ranks terrain texture above
    man-made objects. Kept, off by default, because the idea is sound and a
    scale-space blob detector with a proper significance test may yet work;
    the current implementation does not.

Level-matched input scale
-------------------------
Probe C also found the detector recalled 0.467 at L0 but only 0.200 at L1. It
was trained on L0 renders, so at L1 every object is twice the apparent size it
ever saw. The fix is to feed it the received image at ``960 >> level``, which
puts objects back at their trained apparent size - and costs proportionally
less at the higher levels. Recognition still uses the full-detail crop; only
the *detector's* view is downscaled.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from v2.config import CONFIG, ProposalConfig
from v2.geometry import Box, box_short_side, iou


logger = logging.getLogger(__name__)


@dataclass
class Proposal:
    """One candidate region, in received-image pixel coordinates."""

    box: Box
    score: float
    source: str


# --------------------------------------------------------------------------- #
# Saliency backend
# --------------------------------------------------------------------------- #

class SaliencyProposer:
    """Multi-scale morphological contrast blobs."""

    name = 'saliency'

    def __init__(self, config: ProposalConfig) -> None:
        self.config = config
        self._windows = tuple(2 * scale + 1 for scale in config.saliency_scales)

    @staticmethod
    def _robust_spread(response: np.ndarray) -> float:
        """Median plus a MAD, estimated on a subsample.

        The statistic only sets a threshold, so a sixteenth of the pixels gives
        the same answer to well within its own noise and costs a sixteenth of
        the time. On a 333 ms frame budget that difference is not cosmetic.
        """
        sample = response[::4, ::4].reshape(-1)
        median = float(np.median(sample))
        return median + 1.4826 * float(np.median(np.abs(sample - median)))

    def _response(self, grey: np.ndarray) -> np.ndarray:
        """Largest normalized local-contrast response over all scales.

        The contrast operator is a box-filter band-pass - the pixel minus its
        local mean - rather than a morphological top-hat. Morphology with a
        63x63 structuring element costs O(k^2) per pixel and measured 1690 ms
        per frame here, five times the entire frame interval. ``cv2.blur`` is
        separable and O(1) in the window size, and on high-contrast man-made
        objects sitting on natural terrain the two pick out the same blobs.

        Each scale is divided by its own robust spread before the maximum, so a
        scale that only ever sees texture noise cannot outvote a scale that is
        actually resolving an object.
        """
        combined = np.zeros(grey.shape, dtype=np.float32)
        for window in self._windows:
            local_mean = cv2.blur(grey, (window, window), borderType=cv2.BORDER_REFLECT)
            response = cv2.absdiff(grey, local_mean)
            spread = self._robust_spread(response)
            if spread < 1e-3:
                continue
            np.maximum(combined, response / spread, out=combined)
        return combined

    def propose(self, image: np.ndarray, budget: int, level: int = 0) -> List[Proposal]:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        response = self._response(grey)
        if not np.isfinite(response).any() or response.max() <= 0:
            return []

        threshold = float(np.percentile(response, self.config.saliency_percentile))
        mask = (response >= max(threshold, 1e-3)).astype(np.uint8)
        # One dilation closes the gap between an object's bright and dark side,
        # which otherwise splits a single object into two components.
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        proposals: List[Proposal] = []
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if area <= 1:
                continue
            side = min(width, height)
            if side < self.config.saliency_min_side:
                continue
            if max(width, height) > self.config.saliency_max_side:
                continue
            patch = response[y:y + height, x:x + width]
            proposals.append(
                Proposal(
                    box=(float(x), float(y), float(x + width), float(y + height)),
                    score=float(patch.max()),
                    source=self.name,
                )
            )
        proposals.sort(key=lambda proposal: proposal.score, reverse=True)
        return proposals[:budget]


# --------------------------------------------------------------------------- #
# Detector backend
# --------------------------------------------------------------------------- #

class OnnxYoloProposer:
    """The exported detector, driven directly by onnxruntime.

    Going through ultralytics for ONNX inference measured *worse* end to end
    than PyTorch, despite the detector itself being faster: proposals fell from
    185 ms to 104 ms while the recognizer that runs immediately afterwards rose
    from 76 ms to 249 ms. The cause is ORT's intra-op thread pool, which keeps
    its workers spinning after a call and starves torch on a four-core box.
    Owning the session lets us turn spinning off and pin the thread count, and
    lets us feed the real 960x540 aspect instead of a square letterbox.

    The output decode is the standard YOLO head layout: ``(1, 4 + nc, anchors)``
    with xywh boxes in input-pixel space. Classes are pooled to a single
    objectness score, which is the whole point - identity is not this module's
    job.
    """

    name = 'yolo'

    def __init__(self, config: ProposalConfig) -> None:
        import onnxruntime

        self.config = config
        weights = Path(config.yolo_weights)
        if not weights.is_file():
            raise FileNotFoundError(f'ONNX proposal weights missing: {weights}')

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = max(1, int(config.onnx_threads))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        # The reason this class exists. Without it the pool spins between
        # frames and the recognizer pays for it.
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        self.session = onnxruntime.InferenceSession(
            str(weights), options, providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.weights = weights

    @staticmethod
    def _letterbox(image: np.ndarray, target_long_side: int, stride: int = 32):
        """Scale to fit, then pad right/bottom to a stride multiple.

        Top-left alignment keeps the inverse mapping a plain scale with no
        offset, which is one fewer place for a coordinate bug to hide.
        """
        height, width = image.shape[:2]
        scale = min(target_long_side / width, target_long_side / height)
        new_width = max(stride, int(round(width * scale)))
        new_height = max(stride, int(round(height * scale)))
        resized = cv2.resize(
            image, (new_width, new_height),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        padded_width = int(math.ceil(new_width / stride) * stride)
        padded_height = int(math.ceil(new_height / stride) * stride)
        canvas = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
        canvas[:new_height, :new_width] = resized
        return canvas, float(new_width) / width, float(new_height) / height

    def propose(self, image: np.ndarray, budget: int, level: int = 0) -> List[Proposal]:
        target = max(64, int(self.config.yolo_imgsz) >> max(0, int(level)))
        canvas, scale_x, scale_y = self._letterbox(image, target)
        tensor = canvas[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        outputs = self.session.run(None, {self.input_name: np.ascontiguousarray(tensor)})

        prediction = outputs[0]
        if prediction.ndim != 3:
            return []
        rows = prediction[0]
        if rows.shape[0] < rows.shape[1]:
            rows = rows.T                       # (anchors, 4 + nc)
        boxes_xywh = rows[:, :4]
        # Pooled objectness: the strongest class response, class identity
        # discarded on purpose.
        scores = rows[:, 4:].max(axis=1)

        keep = scores >= self.config.yolo_confidence
        if not keep.any():
            return []
        boxes_xywh, scores = boxes_xywh[keep], scores[keep]

        half_w = boxes_xywh[:, 2] * 0.5
        half_h = boxes_xywh[:, 3] * 0.5
        x1 = (boxes_xywh[:, 0] - half_w) / scale_x
        y1 = (boxes_xywh[:, 1] - half_h) / scale_y
        x2 = (boxes_xywh[:, 0] + half_w) / scale_x
        y2 = (boxes_xywh[:, 1] + half_h) / scale_y

        rects = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        indices = cv2.dnn.NMSBoxes(
            rects.tolist(), scores.astype(float).tolist(),
            float(self.config.yolo_confidence), float(self.config.yolo_iou),
        )
        if indices is None or len(indices) == 0:
            return []
        order = np.array(indices).reshape(-1)

        height, width = image.shape[:2]
        proposals: List[Proposal] = []
        for index in order:
            box = (
                float(max(0.0, x1[index])), float(max(0.0, y1[index])),
                float(min(width, x2[index])), float(min(height, y2[index])),
            )
            if box[2] - box[0] < 1.0 or box[3] - box[1] < 1.0:
                continue
            proposals.append(Proposal(box=box, score=float(scores[index]), source=self.name))
        proposals.sort(key=lambda proposal: proposal.score, reverse=True)
        return proposals[:budget]


class UltralyticsYoloProposer:
    """Detector boxes with the class head deliberately discarded."""

    name = 'yolo'

    def __init__(self, config: ProposalConfig) -> None:
        from ultralytics import YOLO

        self.config = config
        weights = Path(config.yolo_weights)
        if not weights.is_file():
            fallback = Path(config.yolo_fallback_weights)
            if not fallback.is_file():
                raise FileNotFoundError(
                    f'Proposal detector weight is missing: {weights} (and no '
                    f'fallback at {fallback})'
                )
            logger.warning('Proposal weights %s missing; falling back to %s',
                           weights, fallback)
            weights = fallback
        self.weights = weights
        self.model = YOLO(str(weights))

    def propose(self, image: np.ndarray, budget: int, level: int = 0) -> List[Proposal]:
        # Objects at level L are 2^L times their L0 apparent size, and the
        # detector only ever saw L0. Shrinking its input by the same factor
        # restores the trained scale; the boxes come back in received-image
        # coordinates either way, because ultralytics rescales them for us.
        imgsz = max(64, int(self.config.yolo_imgsz) >> max(0, int(level)))
        results = self.model.predict(
            source=image,
            imgsz=imgsz,
            conf=self.config.yolo_confidence,
            iou=self.config.yolo_iou,
            max_det=max(budget, 100),
            device='cpu',
            agnostic_nms=True,
            verbose=False,
        )
        result = results[0]
        proposals: List[Proposal] = []
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            confidences = result.boxes.conf.detach().cpu().numpy()
            for box, confidence in zip(boxes, confidences):
                proposals.append(
                    Proposal(
                        box=tuple(float(value) for value in box),
                        score=float(confidence),
                        source=self.name,
                    )
                )
        proposals.sort(key=lambda proposal: proposal.score, reverse=True)
        return proposals[:budget]


# --------------------------------------------------------------------------- #
# Merging and the public entry point
# --------------------------------------------------------------------------- #

def merge_proposals(proposals: Sequence[Proposal], merge_iou: float, budget: int) -> List[Proposal]:
    """Greedy overlap merge, highest score first, preserving provenance.

    Scores from different backends are not commensurable, so proposals are
    interleaved by rank rather than compared numerically: the saliency
    detector's best blob and the detector's best box both survive.
    """
    by_source: dict = {}
    for proposal in proposals:
        by_source.setdefault(proposal.source, []).append(proposal)
    for rows in by_source.values():
        rows.sort(key=lambda proposal: proposal.score, reverse=True)

    interleaved: List[Proposal] = []
    sources = sorted(by_source)
    position = 0
    while len(interleaved) < sum(len(rows) for rows in by_source.values()):
        added = False
        for source in sources:
            rows = by_source[source]
            if position < len(rows):
                interleaved.append(rows[position])
                added = True
        if not added:
            break
        position += 1

    kept: List[Proposal] = []
    for proposal in interleaved:
        if len(kept) >= budget:
            break
        duplicate = False
        for existing in kept:
            if iou(existing.box, proposal.box) >= merge_iou:
                duplicate = True
                break
        if not duplicate:
            kept.append(proposal)
    return kept


class ProposalEngine:
    """The discovery stage, assembled once and reused for every frame."""

    def __init__(self, config: Optional[ProposalConfig] = None) -> None:
        self.config = config or CONFIG.proposals
        self.backends = []
        wanted = self.config.backend
        if wanted in ('saliency', 'hybrid'):
            self.backends.append(SaliencyProposer(self.config))
        if wanted in ('yolo', 'hybrid'):
            backend = self._build_detector()
            if backend is not None:
                self.backends.append(backend)
        if not self.backends:
            self.backends.append(SaliencyProposer(self.config))

    def _build_detector(self):
        """Prefer the ONNX session, fall back to ultralytics, then to nothing.

        Each step down is a real degradation and is logged as one. A silent
        fallback to no detector would make the endpoint look healthy while
        emitting nothing.
        """
        if str(self.config.yolo_weights).endswith('.onnx'):
            try:
                return OnnxYoloProposer(self.config)
            except Exception:
                logger.exception('ONNX proposal backend unavailable; trying ultralytics')
        for weights in (self.config.yolo_weights, self.config.yolo_fallback_weights):
            try:
                return UltralyticsYoloProposer(
                    replace(self.config, yolo_weights=weights)
                )
            except Exception:
                logger.exception('Ultralytics proposal backend failed for %s', weights)
        logger.error('No detector proposal backend could be loaded')
        return None

    def propose(
        self, image: np.ndarray, budget: Optional[int] = None, level: int = 0
    ) -> List[Proposal]:
        limit = budget or self.config.budget
        collected: List[Proposal] = []
        for backend in self.backends:
            try:
                collected.extend(backend.propose(image, limit, level))
            except Exception:
                logger.exception('Proposal backend %s failed', backend.name)
        return merge_proposals(collected, self.config.merge_iou, limit)

    def warmup(self) -> None:
        blank = np.zeros((540, 960, 3), dtype=np.uint8)
        for level in (0, 1, 2):
            self.propose(blank, budget=8, level=level)
