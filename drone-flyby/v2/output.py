"""Confidence, deduplication and the response contract.

Two things decide the score here, and neither is the detector.

**Ranking.** COCO AP is computed over predictions sorted by confidence, so the
number attached to a box is not a label of belief, it is a position in a queue.
A confidence that mixes identity, geometry quality and staleness puts
well-observed, recently-refreshed tracks above propagated guesses of the same
class, which is precisely what AP rewards.

**Class coverage.** The metric is macro-averaged across the classes present in
the ground truth, and Baseline 1 emitted only 7 of 16 class IDs - capping its
possible score at 0.44 before any other error. A class nobody ever names scores
zero with certainty. So each track names its top ``emit_top_k`` classes, at
proportionally damped confidence. This is safe because AP is per class: a box
labelled "tank" can cost tank's precision and can never touch condor's. A
correctly ranked second or third guess turns a guaranteed zero into a real AP.

Deduplication has to respect that on purpose: the evaluator does no NMS, so
overlapping same-class boxes must be merged, but a track's own secondary
guesses are deliberate, not duplicates, and are exempt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from dtos import (
    OBJECT_CLASSES,
    DroneFlybyPredictionDto,
)
from v2.config import CONFIG, OutputConfig
from v2.geometry import Box, clamp_box_to_frame, iou, source_to_global
from v2.tracks import Track, TrackBank


@dataclass
class Candidate:
    """One annotation before deduplication, with the provenance dedup needs."""

    object_id: str
    box: Box
    confidence: float
    track_id: int
    rank: int          # 0 = the track's first choice, 1+ = a damped extra guess


class OutputBuilder:
    """Turns the track bank into a legal, well-ranked response."""

    def __init__(self, config: Optional[OutputConfig] = None) -> None:
        self.config = config or CONFIG.output

    # -- confidence --------------------------------------------------------- #

    def confidence_for(self, track: Track, posterior: float) -> float:
        """Combine identity, geometry, staleness and observation history.

        Age is never used on its own. A track observed once at L0 and then held
        for four frames and a track observed five times at L1 and propagated by
        an accepted affine for one frame have very different reliability at the
        same age, and the ranking has to see that difference.
        """
        geometry = max(0.05, min(1.0, track.geometry_confidence))
        staleness = math.exp(-track.age / max(1e-3, self.config.staleness_half_life))
        support = 0.55 + 0.45 * min(1.0, track.hits / 3.0)
        # A track only ever seen at L0 carries less trustworthy identity than
        # one confirmed at L1 or L2; the level bonus says so explicitly.
        best_level = max(track.observed_levels) if track.observed_levels else 0
        level_bonus = (0.85, 1.0, 1.0)[best_level]
        value = posterior * geometry * staleness * support * level_bonus
        return float(max(0.0, min(0.999, value)))

    # -- assembly ----------------------------------------------------------- #

    def candidates(self, bank: TrackBank) -> List[Candidate]:
        ceiling = CONFIG.tracks.posterior_ceiling
        top_k = max(1, self.config.emit_top_k)
        rows: List[Candidate] = []
        for track in bank.reportable():
            posterior = track.posterior(ceiling)
            order = np.argsort(posterior)[::-1][:top_k]
            for rank, class_index in enumerate(order):
                value = float(posterior[class_index])
                if rank > 0 and value < self.config.secondary_min_posterior:
                    break
                confidence = self.confidence_for(track, value) * (
                    self.config.secondary_damping ** rank
                )
                if confidence < self.config.min_confidence:
                    continue
                rows.append(
                    Candidate(
                        OBJECT_CLASSES[int(class_index)],
                        track.box,
                        confidence,
                        track.track_id,
                        rank,
                    )
                )
        return rows

    def deduplicate(self, candidates: Sequence[Candidate]) -> List[Candidate]:
        """Greedy suppression, highest confidence first."""
        ordered = sorted(
            candidates, key=lambda row: (-row.confidence, row.object_id, row.track_id)
        )
        kept: List[Candidate] = []
        for candidate in ordered:
            suppressed = False
            for existing in kept:
                overlap = iou(existing.box, candidate.box)
                if existing.object_id == candidate.object_id:
                    if overlap >= self.config.dedup_iou:
                        suppressed = True
                        break
                    continue
                # Different classes on the same physical object. Two separate
                # tracks claiming one object is a duplicate; one track offering
                # its own second or third guess is deliberate.
                if (
                    existing.track_id != candidate.track_id
                    and overlap >= self.config.cross_class_iou
                ):
                    suppressed = True
                    break
            if not suppressed:
                kept.append(candidate)
            if len(kept) >= self.config.max_annotations:
                break
        return kept

    def build(
        self, bank: TrackBank, original_width: int, original_height: int
    ) -> Tuple[List[DroneFlybyPredictionDto], dict]:
        """Produce the annotations list plus a small telemetry summary."""
        rows = self.deduplicate(self.candidates(bank))
        annotations: List[DroneFlybyPredictionDto] = []
        dropped = 0
        for candidate in rows:
            clamped = clamp_box_to_frame(candidate.box)
            if clamped is None:
                dropped += 1
                continue
            normalized = source_to_global(clamped, original_width, original_height)
            # Final belt-and-braces guard: the DTO rejects a degenerate box and
            # a rejected box costs the whole frame, including every other
            # detection in it.
            x1, y1, x2, y2 = normalized
            if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
                dropped += 1
                continue
            if candidate.object_id not in OBJECT_CLASSES:
                dropped += 1
                continue
            annotations.append(
                DroneFlybyPredictionDto(
                    object_id=candidate.object_id,
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    confidence=float(max(0.0, min(1.0, candidate.confidence))),
                )
            )
        summary = {
            'candidates': len(rows),
            'emitted': len(annotations),
            'dropped_invalid': dropped,
            'distinct_classes': len({row.object_id for row in rows}),
            'secondary_guesses': sum(1 for row in rows if row.rank > 0),
        }
        return annotations, summary
