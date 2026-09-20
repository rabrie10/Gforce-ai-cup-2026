"""The Architecture V2 closed loop.

    L0 discovery / motion anchor
        -> global track bank
        -> camera scheduler
        -> L1 routine identity refresh (selective L2)
        -> transfer-oriented crop recognizer
        -> identity evidence fusion
        -> affine GMC / temporal state
        -> full-frame global outputs
        (repeat)

Each stage lives in its own module and is reachable through this one class, so
a component that fails to earn its place can be switched off by environment
variable without touching the loop. Nothing here talks to the network, and the
only per-frame allocations are the proposals and their crops.

The loop is also **defensive by construction**: a failure in any stage degrades
that stage to a neutral behaviour and the frame still produces a valid,
well-formed response. An exception escaping ``predict`` means no response at
all and zero detections for the frame, which is the most expensive failure in
the protocol.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from dtos import (
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
)
from utils import decode_view
from v2.config import CONFIG, Config
from v2.geometry import clamp_box_to_frame, crop_from_view, view_to_source
from v2.gmc import GlobalMotionEstimator, MotionEstimate
from v2.output import OutputBuilder
from v2.proposals import ProposalEngine
from v2.recognizer import CropRecognizer
from v2.scheduler import CameraDecision, CameraScheduler
from v2.telemetry import TelemetryRecorder
from v2.tracks import Observation, TrackBank


logger = logging.getLogger(__name__)


class Stopwatch:
    """Per-component timing, reported on every frame."""

    def __init__(self) -> None:
        self.marks: Dict[str, float] = {}
        self._started: Dict[str, float] = {}

    def start(self, name: str) -> None:
        self._started[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        began = self._started.pop(name, None)
        if began is None:
            return 0.0
        elapsed = (time.perf_counter() - began) * 1000.0
        self.marks[name] = round(elapsed, 2)
        return elapsed

    def set(self, name: str, value: float) -> None:
        self.marks[name] = round(value, 2)


class LatencyLedger:
    """Rolling per-component latency, so the CPU budget is observable.

    The deployment target is a four-vCPU burstable VM and frames arrive every
    333 ms. A component that quietly drifts over budget shows up here as a
    p95, not as a mysterious drop in score three attempts later.
    """

    def __init__(self, capacity: int = 512) -> None:
        self._rows: Dict[str, List[float]] = {}
        self._capacity = capacity
        self._lock = threading.Lock()

    def add(self, marks: Dict[str, float]) -> None:
        with self._lock:
            for name, value in marks.items():
                row = self._rows.setdefault(name, [])
                row.append(float(value))
                if len(row) > self._capacity:
                    del row[0]

    def reset(self) -> None:
        with self._lock:
            self._rows.clear()

    def report(self) -> dict:
        with self._lock:
            rows = {name: list(values) for name, values in self._rows.items()}
        summary = {}
        for name, values in sorted(rows.items()):
            if not values:
                continue
            ordered = sorted(values)
            position = 0.95 * (len(ordered) - 1)
            lower = int(position)
            upper = min(lower + 1, len(ordered) - 1)
            summary[name] = {
                'calls': len(ordered),
                'mean': round(sum(ordered) / len(ordered), 2),
                'median': round(ordered[len(ordered) // 2], 2),
                'p95': round(
                    ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower]), 2
                ),
                'max': round(ordered[-1], 2),
            }
        return summary


class DroneFlybyPipeline:
    """One process-wide instance, loaded and warmed at startup."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or CONFIG
        self.proposals = ProposalEngine(self.config.proposals)
        self.recognizer = CropRecognizer(self.config.recognizer)
        self.bank = TrackBank(self.config.tracks)
        self.motion = GlobalMotionEstimator(self.config.gmc)
        self.scheduler = CameraScheduler(self.config.scheduler)
        self.output = OutputBuilder(self.config.output)
        self.telemetry = TelemetryRecorder(self.config.telemetry)
        self.latency = LatencyLedger()
        self._lock = threading.Lock()
        self._sequence_id: Optional[str] = None
        self._last_frame_index: int = -1
        self._last_observations: List[Observation] = []
        self.last_diagnostics: Dict[str, object] = {}
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------- #

    def load(self) -> None:
        if self._loaded:
            return
        started = time.perf_counter()
        self.recognizer.load()
        self.telemetry.start()
        self._loaded = True
        logger.info(
            'Pipeline loaded in %.0f ms: recognizer=%s proposals=%s',
            (time.perf_counter() - started) * 1000.0,
            self.recognizer.describe(),
            [backend.name for backend in self.proposals.backends],
        )

    def warmup(self) -> None:
        """Run every model once so the first scored frame is not the cold one."""
        self.load()
        started = time.perf_counter()
        blank = np.zeros((540, 960, 3), dtype=np.uint8)
        try:
            self.proposals.warmup()
            self.recognizer.warmup()
            self.motion.estimate(blank, (0, 0, 3840, 2160))
            self.motion.reset()
        except Exception:
            logger.exception('Warmup encountered an error; service continues')
        logger.info('Pipeline warmed in %.0f ms', (time.perf_counter() - started) * 1000.0)

    def describe(self) -> dict:
        return {
            'recognizer': self.recognizer.describe(),
            'proposal_backends': [backend.name for backend in self.proposals.backends],
            'discovery_backend': self.config.proposals.discovery_backend,
            'proposal_budget': self.config.proposals.budget,
            'gmc_model': self.config.gmc.model if self.config.gmc.enabled else 'disabled',
            'working_level': self.config.scheduler.working_level,
            'scheduler_enabled': self.config.scheduler.enabled,
            'telemetry': self.telemetry.snapshot(),
            'latency_ms': self.latency.report(),
        }

    def reset_sequence(self, sequence_id: str) -> None:
        self.bank.reset()
        self.motion.reset()
        self.scheduler.reset()
        self._sequence_id = sequence_id
        self._last_frame_index = -1
        logger.info('Sequence state reset for %s', sequence_id)

    # -- the loop ----------------------------------------------------------- #

    def predict(self, request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
        """Answer one frame. Never raises."""
        with self._lock:
            return self._predict_locked(request)

    def _predict_locked(
        self, request: DroneFlybyPredictRequestDto
    ) -> DroneFlybyPredictResponseDto:
        timing = Stopwatch()
        timing.start('total')
        notes: Dict[str, object] = {}

        if not self._loaded:
            self.load()
        if request.sequence_id != self._sequence_id:
            self.reset_sequence(request.sequence_id)

        view = request.view
        region = tuple(float(value) for value in view.source_region_xyxy)
        level = int(view.resolution_level)

        # 1. Decode.
        timing.start('decode')
        try:
            image = decode_view(view)
        except Exception:
            logger.exception('Could not decode frame %s', request.frame)
            image = None
        timing.stop('decode')

        motion = MotionEstimate(reason='no-image')
        proposals = []
        observations: List[Observation] = []
        recognition_notes: Dict[str, object] = {}

        # Always consume the frame index, even on a frame we cannot decode.
        # Leaving it unconsumed would make the *next* frame believe the gap was
        # larger than it was and over-propagate every track by the difference.
        skipped = self._frames_skipped(request)

        if image is not None:
            # 2. Shared motion, estimated before the bank is advanced.
            timing.start('gmc')
            try:
                motion = self.motion.estimate(image, region)
            except Exception:
                logger.exception('GMC failed on frame %s', request.frame)
                motion = MotionEstimate(reason='exception')
            timing.stop('gmc')

            # 3. Advance every track, including the ones outside this crop.
            timing.start('predict_state')
            for step in range(max(1, skipped)):
                # A skipped frame still moved the world. Re-applying the one
                # transform we have is a better guess than pretending the gap
                # did not happen, but it is only trusted for the first step.
                self.bank.predict(
                    motion.matrix if motion.ok else None,
                    motion.quality if step == 0 else motion.quality * 0.5,
                )
            timing.stop('predict_state')

            # 4. Discovery.
            timing.start('proposals')
            try:
                proposals = self.proposals.propose(image, level=level)
            except Exception:
                logger.exception('Proposals failed on frame %s', request.frame)
                proposals = []
            timing.stop('proposals')

            # 5. Identity on object-centred crops.
            timing.start('recognize')
            observations, recognition_notes = self._recognize(image, proposals, region, level)
            timing.stop('recognize')
            # Kept only so the offline diagnostic can attribute a score to the
            # stage that caused it; nothing in the response path reads it.
            self._last_observations = observations

            # 6. Fuse.
            timing.start('update_state')
            notes['bank'] = self.bank.update(observations, request.frame, region)
            timing.stop('update_state')
        else:
            for _ in range(max(1, skipped)):
                self.bank.predict(None, 0.0)

        # 7. Camera.
        timing.start('schedule')
        self.scheduler.observe(level, view.center_x, view.center_y)
        try:
            decision = self.scheduler.decide(
                level,
                (int(view.center_x), int(view.center_y)),
                request.camera_constraints,
                self.bank,
            )
        except Exception:
            logger.exception('Scheduler failed on frame %s', request.frame)
            decision = CameraDecision(requested=None, reason='scheduler-exception')
        timing.stop('schedule')

        # 8. Global outputs.
        timing.start('emit')
        try:
            annotations, output_notes = self.output.build(
                self.bank, request.original_width, request.original_height
            )
        except Exception:
            logger.exception('Output build failed on frame %s', request.frame)
            annotations, output_notes = [], {'error': 'output-failed'}
        timing.stop('emit')

        timing.stop('total')
        self.latency.add(timing.marks)
        self.last_diagnostics = {
            'proposals': dict(self.proposals.last_stats),
            'recognition': recognition_notes,
            'observations': len(observations),
            'bank': notes.get('bank', {}),
            'track_bank_size': len(self.bank.tracks),
            'track_bank_saturated': (
                self.config.tracks.max_tracks > 0
                and len(self.bank.tracks) >= self.config.tracks.max_tracks
            ),
            'emitted_annotations': len(annotations),
            'emitted_classes': [annotation.object_id for annotation in annotations],
            'emitted_confidences': [float(annotation.confidence) for annotation in annotations],
            'timing_ms': dict(timing.marks),
        }
        self._capture(
            request, view, motion, decision, proposals, observations,
            recognition_notes, annotations, output_notes, timing, notes,
        )
        return DroneFlybyPredictResponseDto(
            request_id=request.request_id,
            frame=request.frame,
            annotations=annotations,
            requested_view=decision.requested,
        )

    # -- stages ------------------------------------------------------------- #

    def _frames_skipped(self, request: DroneFlybyPredictRequestDto) -> int:
        """How many emitted frames went unanswered before this one.

        ``frame_index`` gaps are the protocol's own record of frames we were too
        slow for. The world kept moving through them, so the state has to as
        well, or every skipped frame becomes a permanent offset in every box.
        """
        index = int(request.frame_index)
        gap = 1 if self._last_frame_index < 0 else max(1, index - self._last_frame_index)
        self._last_frame_index = index
        return min(gap, 6)

    def _recognize(
        self,
        image: np.ndarray,
        proposals,
        region: Tuple[float, float, float, float],
        level: int,
    ) -> Tuple[List[Observation], Dict[str, object]]:
        """Crop every proposal, score the batch, lift the boxes to source."""
        if not proposals:
            return [], {'crops': 0}

        crop_size = self.recognizer.crop_size
        context = self.recognizer.crop_context
        crops: List[np.ndarray] = []
        kept = []
        for proposal in proposals:
            crop = crop_from_view(image, proposal.box, crop_size, context)
            if crop is None:
                continue
            crops.append(crop)
            kept.append(proposal)

        results, elapsed_ms = self.recognizer.recognize(crops, level)
        threshold = self.config.recognizer.objectness_threshold
        observations: List[Observation] = []
        rejected = 0
        for proposal, result in zip(kept, results):
            # The reject option, applied before anything reaches the bank. A
            # proposal that matches terrain better than any class prototype is
            # not a weak object, it is not an object, and admitting it as a
            # low-confidence track is how 100 tracks per frame happen.
            if result.objectness < threshold:
                rejected += 1
                continue
            source_box = clamp_box_to_frame(view_to_source(proposal.box, region))
            if source_box is None:
                continue
            # Recognition quality gates the evidence weight: a crop whose best
            # and second-best prototypes are indistinguishable should move the
            # posterior less than one with a clear winner.
            margin_quality = min(1.0, max(0.15, 0.35 + 3.0 * result.margin))
            quality = float(margin_quality * (0.5 + 0.5 * result.objectness))
            observations.append(
                Observation(
                    box=source_box,
                    posterior=result.posterior,
                    level=level,
                    quality=quality,
                    proposal_score=proposal.score,
                    source=proposal.source,
                )
            )
        return observations, {
            'crops': len(crops),
            'rejected_as_background': rejected,
            'recognize_ms': round(elapsed_ms, 2),
            'mean_margin': round(
                float(np.mean([r.margin for r in results])) if results else 0.0, 4
            ),
            'mean_objectness': round(
                float(np.mean([r.objectness for r in results])) if results else 0.0, 4
            ),
            'top_classes': [result.best_class for result in results],
            'top_posteriors': [round(float(np.max(result.posterior)), 6) for result in results],
            'objectness': [round(float(result.objectness), 6) for result in results],
        }

    # -- telemetry ---------------------------------------------------------- #

    def _capture(
        self,
        request,
        view,
        motion,
        decision,
        proposals,
        observations,
        recognition_notes,
        annotations,
        output_notes,
        timing,
        notes,
    ) -> None:
        if not self.telemetry.active:
            return
        try:
            payload = {
                'sequence_id': request.sequence_id,
                'frame': request.frame,
                'frame_index': request.frame_index,
                'request_id': request.request_id,
                'captured_at': time.time(),
                'view': {
                    'resolution_level': view.resolution_level,
                    'center_x': view.center_x,
                    'center_y': view.center_y,
                    'source_region_xyxy': list(view.source_region_xyxy),
                },
                'camera_constraints': {
                    'maximum_center_delta': request.camera_constraints.maximum_center_delta,
                    'allowed_resolution_levels': list(
                        request.camera_constraints.allowed_resolution_levels
                    ),
                },
                'camera_command_feedback': (
                    None
                    if request.camera_command_feedback is None
                    else {
                        'frame': request.camera_command_feedback.frame,
                        'reason': request.camera_command_feedback.reason,
                    }
                ),
                'gmc': motion.summary(),
                'camera_decision': decision.summary(),
                'proposals': {
                    'count': len(proposals),
                    **self.proposals.last_stats,
                    'sources': sorted({p.source for p in proposals}),
                    'boxes': [
                        {'box': [round(v, 1) for v in p.box],
                         'score': round(p.score, 4),
                         'source': p.source}
                        for p in proposals[:40]
                    ],
                },
                'recognition': recognition_notes,
                'observations': [
                    {
                        'box': [round(v, 1) for v in o.box],
                        'level': o.level,
                        'quality': round(o.quality, 4),
                        'top': int(np.argmax(o.posterior)),
                        'top_posterior': round(float(np.max(o.posterior)), 4),
                        'source': o.source,
                    }
                    for o in observations[:40]
                ],
                'tracks': self.bank.summaries(),
                'output': output_notes,
                'emitted': [
                    {
                        'object_id': annotation.object_id,
                        'confidence': round(float(annotation.confidence), 6),
                        'bbox': [round(float(value), 8) for value in annotation.bbox],
                    }
                    for annotation in annotations
                ],
                'state': notes,
                'timing_ms': timing.marks,
            }
            self.telemetry.record(payload, view.image)
        except Exception:
            # Telemetry assembly must never be able to break a response.
            logger.debug('Telemetry capture skipped', exc_info=True)


_pipeline: Optional[DroneFlybyPipeline] = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> DroneFlybyPipeline:
    """The single process-wide pipeline."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = DroneFlybyPipeline()
    return _pipeline
