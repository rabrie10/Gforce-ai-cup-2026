"""The global track bank: what the system believes about the whole source frame.

The protocol scores every frame against every object in it, including the ones
the camera is not looking at, so this is the component that makes an active
camera policy affordable at all. Without it, every frame spent at L1 would
throw away 75% of the recall.

Two properties are deliberate:

* Identity is an accumulating, **decaying** evidence vector, not a latched
  label. Baseline 1 mislabelled tank as spacecraft in 7 of 8 localizations; a
  bank that latched the first guess would turn that into a permanent wrong
  answer reinforced by every later frame. The decay means better-resolved later
  evidence can overturn a weak seed.
* Geometry confidence is tracked separately from identity confidence and decays
  through propagation, so a well-identified but long-unobserved track is
  reported as what it is: a confident guess about *what*, an increasingly poor
  guess about *where*.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH, OBJECT_CLASSES
from v2.config import CONFIG, TrackConfig
from v2.geometry import (
    LEVEL_DOWNSCALE,
    Box,
    apply_affine_to_box,
    box_center,
    box_diagonal,
    box_short_side,
    clamp_box_to_frame,
    contains,
    iou,
)


NUM_CLASSES = len(OBJECT_CLASSES)
_EPSILON = 1e-6


def _level_for_region(region: Sequence[float]) -> int:
    """Recover the resolution level a source region corresponds to."""
    width = float(region[2]) - float(region[0])
    if width >= 0.75 * IMAGE_WIDTH:
        return 0
    if width >= 0.375 * IMAGE_WIDTH:
        return 1
    return 2


@dataclass
class Observation:
    """One direct measurement of an object, already in source coordinates."""

    box: Box
    posterior: np.ndarray
    level: int
    quality: float = 1.0
    proposal_score: float = 0.0
    source: str = 'proposal'


@dataclass
class Track:
    """Everything the pipeline knows about one candidate object."""

    track_id: int
    box: Box
    evidence: np.ndarray
    created_frame: int
    last_observed_frame: int
    last_observed_level: int
    hits: int = 1
    age: int = 0
    misses_in_view: int = 0
    geometry_confidence: float = 1.0
    propagation_model: str = 'observed'
    observed_levels: set = field(default_factory=set)
    history: deque = field(default_factory=lambda: deque(maxlen=4))
    confirmed: bool = False

    # -- identity ----------------------------------------------------------- #

    def posterior(self, ceiling: float) -> np.ndarray:
        """Softmax of accumulated evidence, with a ceiling on any one class."""
        shifted = self.evidence - self.evidence.max()
        weights = np.exp(shifted)
        total = weights.sum()
        if total <= 0:
            return np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        posterior = weights / total
        best = int(np.argmax(posterior))
        if posterior[best] > ceiling:
            surplus = posterior[best] - ceiling
            posterior[best] = ceiling
            others = np.ones(NUM_CLASSES, dtype=bool)
            others[best] = False
            remaining = posterior[others].sum()
            if remaining > _EPSILON:
                posterior[others] += surplus * posterior[others] / remaining
            else:
                posterior[others] += surplus / (NUM_CLASSES - 1)
        return posterior

    def identity(self, ceiling: float) -> Tuple[str, float, str, float]:
        """Best class and its posterior, plus the runner-up."""
        posterior = self.posterior(ceiling)
        order = np.argsort(posterior)[::-1]
        return (
            OBJECT_CLASSES[int(order[0])],
            float(posterior[order[0]]),
            OBJECT_CLASSES[int(order[1])],
            float(posterior[order[1]]),
        )

    def velocity(self) -> Optional[Tuple[float, float]]:
        """Per-frame source-pixel velocity from the two latest observations."""
        if len(self.history) < 2:
            return None
        (frame_a, center_a), (frame_b, center_b) = self.history[-2], self.history[-1]
        span = frame_b - frame_a
        if span <= 0:
            return None
        return ((center_b[0] - center_a[0]) / span, (center_b[1] - center_a[1]) / span)

    def summary(self, ceiling: float) -> dict:
        object_id, confidence, runner_up, runner_up_confidence = self.identity(ceiling)
        return {
            'track_id': self.track_id,
            'box': [round(float(value), 1) for value in self.box],
            'object_id': object_id,
            'identity_confidence': round(confidence, 4),
            'runner_up': runner_up,
            'runner_up_confidence': round(runner_up_confidence, 4),
            'hits': self.hits,
            'age': self.age,
            'confirmed': self.confirmed,
            'last_observed_frame': self.last_observed_frame,
            'last_observed_level': self.last_observed_level,
            'geometry_confidence': round(self.geometry_confidence, 4),
            'propagation_model': self.propagation_model,
            'short_side': round(box_short_side(self.box), 1),
        }


class TrackBank:
    """The frame-global state, propagated, matched, fused and culled."""

    def __init__(self, config: Optional[TrackConfig] = None) -> None:
        self.config = config or CONFIG.tracks
        self.tracks: List[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    # -- propagation -------------------------------------------------------- #

    def predict(self, motion_matrix: Optional[np.ndarray], motion_quality: float) -> None:
        """Advance every track one frame.

        The shared transform is preferred when it was accepted. D-007's caveat
        is honoured: no frozen per-target residual is carried, because matched
        fixed-size affine beat affine-plus-residual in the oracle. Constant
        velocity is the per-track fallback for when the global estimate was
        rejected, and holding still is the last resort - which the scene audit
        shows fails every single one-step check, so a held track is marked as
        such and loses geometry confidence fast.
        """
        for track in self.tracks:
            track.age += 1
            if motion_matrix is not None:
                predicted = apply_affine_to_box(motion_matrix, track.box)
                track.propagation_model = 'gmc'
                track.geometry_confidence *= 0.55 + 0.45 * max(0.0, min(1.0, motion_quality))
            else:
                velocity = track.velocity()
                if velocity is not None:
                    predicted = (
                        track.box[0] + velocity[0],
                        track.box[1] + velocity[1],
                        track.box[2] + velocity[0],
                        track.box[3] + velocity[1],
                    )
                    track.propagation_model = 'constant-velocity'
                    track.geometry_confidence *= 0.80
                else:
                    predicted = track.box
                    track.propagation_model = 'hold'
                    track.geometry_confidence *= 0.45
            track.box = predicted

    # -- matching ----------------------------------------------------------- #

    def _match(self, observations: Sequence[Observation]) -> List[Tuple[int, int, float]]:
        """Greedy highest-affinity matching, deterministic by construction."""
        pairs: List[Tuple[float, int, int]] = []
        for observation_index, observation in enumerate(observations):
            for track_index, track in enumerate(self.tracks):
                affinity = self._affinity(track, observation)
                if affinity > 0:
                    pairs.append((affinity, observation_index, track_index))
        pairs.sort(key=lambda row: (-row[0], row[1], row[2]))

        used_observations, used_tracks = set(), set()
        matches: List[Tuple[int, int, float]] = []
        for affinity, observation_index, track_index in pairs:
            if observation_index in used_observations or track_index in used_tracks:
                continue
            used_observations.add(observation_index)
            used_tracks.add(track_index)
            matches.append((observation_index, track_index, affinity))
        return matches

    def _affinity(self, track: Track, observation: Observation) -> float:
        """IoU when the boxes are big enough for it to mean something.

        At L0 a small object is a handful of pixels, so a modest localization
        error drives IoU to zero even when both boxes clearly describe the same
        thing. The centre-distance gate covers that regime; the two are combined
        rather than alternated so a good IoU always outranks a bare gate hit.
        """
        overlap = iou(track.box, observation.box)
        if overlap >= self.config.match_iou:
            return 1.0 + overlap

        track_center = box_center(track.box)
        observation_center = box_center(observation.box)
        distance = math.hypot(
            track_center[0] - observation_center[0],
            track_center[1] - observation_center[1],
        )
        reference = max(box_diagonal(track.box), box_diagonal(observation.box), 8.0)
        gate = self.config.match_center_gate * reference
        if distance > gate:
            return 0.0
        return max(0.0, 1.0 - distance / gate)

    # -- update ------------------------------------------------------------- #

    def update(
        self,
        observations: Sequence[Observation],
        frame: int,
        view_region: Optional[Sequence[float]] = None,
    ) -> dict:
        """Fuse this frame's direct measurements into the bank."""
        matches = self._match(observations)
        matched_tracks = {track_index for _, track_index, _ in matches}
        matched_observations = {observation_index for observation_index, _, _ in matches}

        for observation_index, track_index, _ in matches:
            self._absorb(self.tracks[track_index], observations[observation_index], frame)

        created = 0
        for index, observation in enumerate(observations):
            if index in matched_observations:
                continue
            self.tracks.append(self._spawn(observation, frame))
            created += 1

        # A track sitting inside the current view that produced no observation
        # is evidence against itself. A track outside the view is not: we
        # simply could not see it.
        #
        # And a track that *is* in view but is too small to be detectable at
        # this level is not evidence against itself either. The Baseline 1
        # audit found zero of the sixty sub-8px L0 appearances were localized,
        # so at L0 the absence of a small object means nothing about whether it
        # is there. Counting it as a miss would delete exactly the tiny targets
        # the whole active-resolution design exists to recover.
        if view_region is not None:
            level = _level_for_region(view_region)
            floor = self.config.detectable_short_side * LEVEL_DOWNSCALE[level]
            for index, track in enumerate(self.tracks):
                if index in matched_tracks or index >= len(self.tracks) - created:
                    continue
                if not contains(view_region, track.box):
                    continue
                if box_short_side(track.box) < floor:
                    continue
                track.misses_in_view += 1

        removed = self._cull()
        return {
            'matched': len(matches),
            'created': created,
            'removed': removed,
            'tracks': len(self.tracks),
        }

    def _absorb(self, track: Track, observation: Observation, frame: int) -> None:
        weight = self._observation_weight(observation)
        track.evidence *= self.config.evidence_decay
        track.evidence += weight * np.log(np.clip(observation.posterior, 1e-4, 1.0))

        # The direct measurement leads. Blending with the propagated box only
        # smooths quantization, and only meaningfully at L0 where one received
        # pixel is four source pixels; at L1 and L2 the measurement is trusted
        # almost completely, because objects move 60-80 source px per frame and
        # a lagging box loses IoU 0.50 by itself.
        blend = (0.75, 0.90, 0.95)[observation.level]
        blended = tuple(
            blend * new + (1.0 - blend) * old
            for new, old in zip(observation.box, track.box)
        )
        clamped = clamp_box_to_frame(blended)
        track.box = clamped if clamped is not None else observation.box

        track.hits += 1
        track.age = 0
        track.misses_in_view = 0
        track.last_observed_frame = frame
        track.last_observed_level = observation.level
        track.observed_levels.add(observation.level)
        track.geometry_confidence = min(1.0, 0.75 + 0.25 * observation.quality)
        track.propagation_model = 'observed'
        track.history.append((frame, box_center(track.box)))
        if track.hits >= self.config.confirm_hits:
            track.confirmed = True

    def _observation_weight(self, observation: Observation) -> float:
        level_weight = self.config.level_weight[observation.level]
        return float(level_weight * max(0.15, min(1.0, observation.quality)))

    def _spawn(self, observation: Observation, frame: int) -> Track:
        weight = self._observation_weight(observation)
        track = Track(
            track_id=self._next_id,
            box=observation.box,
            evidence=weight * np.log(np.clip(observation.posterior, 1e-4, 1.0)),
            created_frame=frame,
            last_observed_frame=frame,
            last_observed_level=observation.level,
        )
        track.observed_levels.add(observation.level)
        track.history.append((frame, box_center(observation.box)))
        track.confirmed = self.config.confirm_hits <= 1
        self._next_id += 1
        return track

    def _cull(self) -> int:
        """Drop tracks that have left the frame, gone stale or been disproved."""
        survivors: List[Track] = []
        for track in self.tracks:
            clamped = clamp_box_to_frame(track.box)
            if clamped is None:
                continue
            center = box_center(track.box)
            if not (0 <= center[0] <= IMAGE_WIDTH and 0 <= center[1] <= IMAGE_HEIGHT):
                continue
            limit = (
                self.config.max_age if track.confirmed else self.config.max_tentative_age
            )
            if track.age > limit:
                continue
            if track.misses_in_view >= 3:
                continue
            track.box = clamped
            survivors.append(track)

        # Hard ceiling. The reject option should keep the bank small, but it is
        # calibrated on this landscape and the hosted sequence is a different
        # one. If objectness degrades there, an unbounded bank would turn into
        # unbounded per-frame latency - and a slow frame is a skipped frame
        # scored as no detections. Confirmed, well-supported, recently seen
        # tracks survive a squeeze; speculative ones do not.
        limit = self.config.max_tracks
        if limit > 0 and len(survivors) > limit:
            survivors.sort(
                key=lambda track: (not track.confirmed, track.age, -track.hits)
            )
            survivors = survivors[:limit]

        removed = len(self.tracks) - len(survivors)
        self.tracks = survivors
        return removed

    # -- reporting ---------------------------------------------------------- #

    def reportable(self) -> List[Track]:
        """Tracks the output stage may consider, confirmed ones first."""
        return sorted(
            self.tracks,
            key=lambda track: (not track.confirmed, track.age, -track.hits),
        )

    def summaries(self) -> List[dict]:
        ceiling = self.config.posterior_ceiling
        return [track.summary(ceiling) for track in self.reportable()]
