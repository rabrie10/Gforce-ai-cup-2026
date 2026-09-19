"""Deterministic active-resolution policy.

No RL, no POMDP. The camera is pointed by a utility score over candidate
centres, and every term in that score traces to a measurement:

* **unknown / ambiguous identity** - the audit found only 7/16 class IDs were
  ever emitted, and mAP is macro-averaged over classes, so a class nobody has
  named is worth far more than a class already named well.
* **tiny apparent size** - 60/259 GT appearances are under 8 px at L0 and none
  of them was localized; at L1 only 10/259 are. Moving a small object to a
  better level is the single largest measured lever.
* **staleness** - constant-velocity propagation holds IoU>=0.50 for about three
  to four frames and collapses by h=6, so a track unobserved for longer is a
  liability wherever it sits.
* **boundary risk** - objects enter at the top and leave at the bottom, so a
  track near the bottom edge is about to stop being scorable and one near the
  top has just arrived and has no identity yet.
* **coverage** - one L1 crop holds a median of five complete objects against
  three at L2, which is why L1 and not L2 is the working level.

Transitions obey the request's own ``camera_constraints``; an unreachable
target is walked toward rather than abandoned, because a refused command wastes
a whole frame of camera time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from dtos import (
    FULL_FRAME_CENTER,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SOURCE_REGION_SIZES,
    CameraConstraintsDto,
    RequestedViewDto,
)
from v2.config import CONFIG, SchedulerConfig
from v2.geometry import (
    LEVEL_DOWNSCALE,
    box_center,
    box_short_side,
    clamp_center,
    contains,
    iou,
    source_region,
)
from v2.tracks import Track, TrackBank


@dataclass
class CameraDecision:
    """What the scheduler asked for, and why."""

    requested: Optional[RequestedViewDto]
    reason: str
    score: float = 0.0
    candidates: int = 0
    target_track: Optional[int] = None

    def summary(self) -> dict:
        return {
            'reason': self.reason,
            'score': round(self.score, 4),
            'candidates': self.candidates,
            'target_track': self.target_track,
            'requested': None
            if self.requested is None
            else {
                'resolution_level': self.requested.resolution_level,
                'center_x': self.requested.center_x,
                'center_y': self.requested.center_y,
            },
        }


def move_toward(
    current: Tuple[int, int], target: Tuple[int, int], limit: float
) -> Tuple[int, int]:
    """Step from ``current`` toward ``target`` without exceeding ``limit``.

    An unreachable centre is not a reason to stay put. Walking the allowed
    fraction of the way there spends the frame making progress instead of
    having the command refused.
    """
    dx = target[0] - current[0]
    dy = target[1] - current[1]
    distance = math.hypot(dx, dy)
    if distance <= limit or distance < 1e-6:
        return int(round(target[0])), int(round(target[1]))
    # Half a pixel of slack keeps float rounding from tipping over the limit.
    scale = max(0.0, (limit - 0.5)) / distance
    return int(round(current[0] + dx * scale)), int(round(current[1] + dy * scale))


class CameraScheduler:
    """Utility-driven view selection with a periodic global refresh."""

    def __init__(self, config: Optional[SchedulerConfig] = None) -> None:
        self.config = config or CONFIG.scheduler
        self.frames_since_l0 = 0
        self.l2_dwell = 0
        self.sweep_index = 0
        # Last frame index at which each tile of a coarse grid was inside a view.
        self._visited: dict = {}
        self._frame_counter = 0

    def reset(self) -> None:
        self.frames_since_l0 = 0
        self.l2_dwell = 0
        self.sweep_index = 0
        self._visited.clear()
        self._frame_counter = 0

    # -- bookkeeping -------------------------------------------------------- #

    def observe(self, level: int, center_x: int, center_y: int) -> None:
        """Record where the camera actually was for this frame."""
        self._frame_counter += 1
        if level == 0:
            self.frames_since_l0 = 0
        else:
            self.frames_since_l0 += 1
        self.l2_dwell = self.l2_dwell + 1 if level == 2 else 0

        region = source_region(level, center_x, center_y)
        for tile_x, tile_y in self._tiles_in(region):
            self._visited[(tile_x, tile_y)] = self._frame_counter

    @staticmethod
    def _tiles_in(region) -> List[Tuple[int, int]]:
        """Coarse 8x6 grid used only to remember what has been looked at."""
        tile_w, tile_h = IMAGE_WIDTH / 8.0, IMAGE_HEIGHT / 6.0
        x0 = max(0, int(region[0] // tile_w))
        x1 = min(7, int((region[2] - 1) // tile_w))
        y0 = max(0, int(region[1] // tile_h))
        y1 = min(5, int((region[3] - 1) // tile_h))
        return [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]

    def _unvisited_fraction(self, region) -> float:
        tiles = self._tiles_in(region)
        if not tiles:
            return 0.0
        horizon = max(1, self.config.l0_refresh_period)
        stale = 0
        for tile in tiles:
            last = self._visited.get(tile)
            if last is None or (self._frame_counter - last) >= horizon:
                stale += 1
        return stale / len(tiles)

    # -- utility ------------------------------------------------------------ #

    def _track_gain(self, track: Track, level: int, ceiling: float) -> float:
        _, confidence, _, runner_up = track.identity(ceiling)
        unknown = max(0.0, 1.0 - confidence)
        ambiguity = max(0.0, 1.0 - (confidence - runner_up))
        staleness = min(1.0, track.age / max(1.0, float(CONFIG.tracks.max_age)))

        apparent = box_short_side(track.box) / LEVEL_DOWNSCALE[level]
        tiny = 1.0 if apparent < 10.0 else (0.5 if apparent < 16.0 else 0.0)

        centre = box_center(track.box)
        # Bottom edge: about to leave the frame and stop being scorable.
        # Top edge: just arrived, so identity is almost certainly still missing.
        depth = centre[1] / IMAGE_HEIGHT
        boundary = 1.0 if depth > 0.82 else (0.7 if depth < self.config.entry_band else 0.0)

        gain = (
            self.config.weight_unknown * unknown
            + self.config.weight_ambiguous * ambiguity
            + self.config.weight_stale * staleness
            + self.config.weight_tiny * tiny
            + self.config.weight_boundary_risk * boundary
        )
        # A track with no direct observation yet is the most valuable of all.
        if not track.confirmed:
            gain *= 1.35
        return gain

    def _candidate_centres(self, bank: TrackBank, level: int) -> List[Tuple[int, int]]:
        centres = []
        for track in bank.reportable()[:24]:
            centre = box_center(track.box)
            centres.append(clamp_center(level, centre[0], centre[1]))

        width, height = SOURCE_REGION_SIZES[level]
        # A sweep grid with half-tile overlap, so discovery keeps working even
        # when the bank is empty or entirely confident.
        step_x = max(1, width // 2)
        step_y = max(1, height // 2)
        for y in range(height // 2, IMAGE_HEIGHT - height // 2 + 1, step_y):
            for x in range(width // 2, IMAGE_WIDTH - width // 2 + 1, step_x):
                centres.append(clamp_center(level, x, y))

        # Entry band: new objects arrive at the top of the frame on a forward
        # survey flight, so it is always worth having as a candidate.
        entry_y = max(height // 2, int(self.config.entry_band * IMAGE_HEIGHT * 0.5))
        for x in range(width // 2, IMAGE_WIDTH - width // 2 + 1, step_x):
            centres.append(clamp_center(level, x, entry_y))

        seen, unique = set(), []
        for centre in centres:
            if centre not in seen:
                seen.add(centre)
                unique.append(centre)
        return unique

    def _score_centre(
        self, centre: Tuple[int, int], level: int, bank: TrackBank, ceiling: float
    ) -> Tuple[float, Optional[int]]:
        region = source_region(level, centre[0], centre[1])
        total = 0.0
        covered = 0
        best_track, best_gain = None, 0.0
        for track in bank.tracks:
            overlap = iou(track.box, region)
            if overlap <= 0 and not contains(region, track.box):
                continue
            share = 1.0 if contains(region, track.box) else 0.35
            gain = self._track_gain(track, level, ceiling) * share
            if gain <= 0:
                continue
            total += gain
            covered += 1
            if gain > best_gain:
                best_gain, best_track = gain, track.track_id
        # Several targets in one crop is strictly better than one, but the
        # benefit is sub-linear: a crowded crop still only gets one exposure.
        if covered > 1:
            total += self.config.weight_coverage * math.log(covered)
        total += self.config.weight_unvisited * self._unvisited_fraction(region)
        return total, best_track

    # -- decision ----------------------------------------------------------- #

    def decide(
        self,
        current_level: int,
        current_center: Tuple[int, int],
        constraints: CameraConstraintsDto,
        bank: TrackBank,
    ) -> CameraDecision:
        if not self.config.enabled:
            return CameraDecision(requested=None, reason='scheduler-disabled')

        allowed = set(constraints.allowed_resolution_levels)
        if not allowed:
            return CameraDecision(requested=None, reason="no-allowed-level")
        ceiling = CONFIG.tracks.posterior_ceiling
        working = self.config.working_level

        # 1. Periodic global refresh. Restores coverage, re-anchors motion and
        #    re-seeds discovery over the whole frame.
        if (
            working != 0
            and self.frames_since_l0 >= self.config.l0_refresh_period
            and 0 in allowed
            and constraints.bounds_for_level(0) is not None
        ):
            return CameraDecision(
                requested=RequestedViewDto(
                    resolution_level=0,
                    center_x=FULL_FRAME_CENTER[0],
                    center_y=FULL_FRAME_CENTER[1],
                ),
                reason='periodic-l0-refresh',
            )

        # 2. L2 is a dwell, not a home. Leave as soon as the budget is spent.
        leaving_l2 = current_level == 2 and (
            self.l2_dwell >= self.config.max_l2_dwell
            or 2 not in allowed
            # Under an L0 policy the camera would never choose L2, so finding
            # itself there means something went wrong upstream. Dwelling on a
            # destination we did not want is 6.25% coverage for no reason.
            or working == 0
        )
        if leaving_l2 and 1 in allowed:
            target = self._best_view(1, current_center, bank, ceiling, constraints)
            if target is not None:
                centre, score, track_id = target
                requested = self._plan_move(1, centre, current_center, constraints)
                if requested is not None:
                    return CameraDecision(
                        requested=requested,
                        reason='l2-dwell-expired',
                        score=score,
                        target_track=track_id,
                    )

        # 3. Selective L2 disambiguation, only reachable from L1 and only for a
        #    target that is genuinely both uncertain and too small to resolve.
        if current_level == 1 and 2 in allowed:
            target = self._l2_target(bank, ceiling)
            if target is not None:
                track, centre = target
                requested = self._plan_move(2, centre, current_center, constraints)
                # Only worth it if the command actually reaches the target; a
                # half-way L2 crop is 6.25% of the frame pointed at nothing.
                if requested is not None and math.hypot(
                    requested.center_x - centre[0], requested.center_y - centre[1]
                ) < 1.0:
                    return CameraDecision(
                        requested=requested,
                        reason='l2-disambiguation',
                        target_track=track.track_id,
                    )

        # 4. Routine identity refresh. Staying at L2 is allowed while the dwell
        #    budget lasts; otherwise the working level is the home position.
        if current_level == 2 and not leaving_l2 and 2 in allowed:
            level = 2
        elif working in allowed:
            level = working
        else:
            # The working level is not reachable in one step - L0 is never
            # reachable from L2. Step toward it rather than settling for
            # wherever we happen to be, so the camera walks home in two frames
            # instead of loitering until some other rule fires.
            level = min(allowed, key=lambda value: (abs(value - working), value))

        best = self._best_view(level, current_center, bank, ceiling, constraints)
        if best is None:
            return CameraDecision(requested=None, reason='no-legal-candidate')
        centre, score, track_id = best
        transition_penalty = 0.0 if level == current_level else self.config.transition_cost

        requested = self._plan_move(level, centre, current_center, constraints)
        if requested is None:
            return CameraDecision(requested=None, reason='no-legal-move')
        if (
            level == current_level
            and (requested.center_x, requested.center_y) == tuple(current_center)
        ):
            return CameraDecision(
                requested=None,
                reason='hold-current-view',
                score=score - transition_penalty,
                target_track=track_id,
            )
        return CameraDecision(
            requested=requested,
            reason='utility-refresh',
            score=score - transition_penalty,
            target_track=track_id,
        )

    def _best_view(
        self,
        level: int,
        current_center: Tuple[int, int],
        bank: TrackBank,
        ceiling: float,
        constraints: CameraConstraintsDto,
    ) -> Optional[Tuple[Tuple[int, int], float, Optional[int]]]:
        if constraints.bounds_for_level(level) is None:
            return None
        if level == 0:
            return FULL_FRAME_CENTER, 0.0, None

        candidates = self._candidate_centres(bank, level)
        if not candidates:
            return None
        limit = max(1.0, float(constraints.maximum_center_delta))

        best: Optional[Tuple[Tuple[int, int], float, Optional[int]]] = None
        for centre in candidates:
            score, track_id = self._score_centre(centre, level, bank, ceiling)
            # Movement is not free: a distant centre is reached late and costs
            # the frames in between, so near candidates of equal value win.
            distance = math.hypot(centre[0] - current_center[0], centre[1] - current_center[1])
            score -= 0.25 * min(1.0, distance / limit)
            if best is None or score > best[1]:
                best = (centre, score, track_id)
        return best

    def _l2_target(self, bank: TrackBank, ceiling: float) -> Optional[Tuple[Track, Tuple[int, int]]]:
        best: Optional[Tuple[float, Track]] = None
        for track in bank.tracks:
            _, confidence, _, _ = track.identity(ceiling)
            if confidence >= self.config.l2_identity_threshold:
                continue
            if box_short_side(track.box) > self.config.l2_max_short_side:
                continue
            if track.age > 1:
                # Only chase a target we have just seen; an L2 crop aimed at a
                # stale guess is 6.25% of the frame pointed at nothing.
                continue
            priority = (1.0 - confidence) * (1.0 if track.confirmed else 0.8)
            if best is None or priority > best[0]:
                best = (priority, track)
        if best is None:
            return None
        centre = box_center(best[1].box)
        return best[1], clamp_center(2, centre[0], centre[1])

    @staticmethod
    def _clamp_to_bounds(level: int, centre: Sequence[float], constraints) -> Optional[Tuple[int, int]]:
        bounds = constraints.bounds_for_level(level)
        if bounds is None:
            return None
        return (
            int(min(max(int(round(centre[0])), bounds.minimum_center_x), bounds.maximum_center_x)),
            int(min(max(int(round(centre[1])), bounds.minimum_center_y), bounds.maximum_center_y)),
        )

    def _plan_move(
        self,
        level: int,
        desired: Sequence[float],
        current: Tuple[int, int],
        constraints: CameraConstraintsDto,
    ) -> Optional[RequestedViewDto]:
        """Turn a wish into a command the evaluator will actually accept.

        Order matters. Clamping *after* stepping can push the command back out
        past the movement limit, because the legal-centre box for a level is
        not the box the camera is currently in. So: clamp the destination
        first, step toward the clamped point, clamp again for rounding, and if
        the result still breaks the limit fall back to the projection of the
        current centre - which is reachable by construction, since the worst
        case is an L2 corner at 550.76 px against the 551 px L2 limit.
        """
        if level == 0:
            if constraints.bounds_for_level(0) is None:
                return None
            return RequestedViewDto(
                resolution_level=0,
                center_x=FULL_FRAME_CENTER[0],
                center_y=FULL_FRAME_CENTER[1],
            )

        target = self._clamp_to_bounds(level, desired, constraints)
        if target is None:
            return None
        limit = float(constraints.maximum_center_delta)
        stepped = self._clamp_to_bounds(level, move_toward(current, target, limit), constraints)
        if stepped is None:
            return None
        if math.hypot(stepped[0] - current[0], stepped[1] - current[1]) > limit:
            stepped = self._clamp_to_bounds(level, current, constraints)
            if stepped is None or math.hypot(
                stepped[0] - current[0], stepped[1] - current[1]
            ) > limit:
                return None
        return RequestedViewDto(
            resolution_level=int(level), center_x=int(stepped[0]), center_y=int(stepped[1])
        )
