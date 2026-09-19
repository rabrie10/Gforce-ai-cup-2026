"""Acceptance tests for Architecture V2.

These are the local gates the handoff requires before a hosted attempt: DTO and
geometry correctness, current-view to global mapping, camera transition
legality, track lifecycle, identity fusion, GMC quality gating and fallback,
deduplication, telemetry isolation and deterministic replay.
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtos import (  # noqa: E402
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
from utils import (  # noqa: E402
    describe_camera_rejection,
    encode_image,
    source_region_for_view,
    validate_response,
)
from v2.config import (  # noqa: E402
    CONFIG,
    GmcConfig,
    OutputConfig,
    SchedulerConfig,
    TrackConfig,
)
from v2.geometry import (  # noqa: E402
    apply_affine_to_box,
    compose_affine,
    decompose_affine,
    iou,
    source_to_view,
    view_to_source,
)
from v2.gmc import GlobalMotionEstimator  # noqa: E402
from v2.output import Candidate, OutputBuilder  # noqa: E402
from v2.scheduler import CameraScheduler, move_toward  # noqa: E402
from v2.telemetry import TelemetryRecorder  # noqa: E402
from v2.tracks import NUM_CLASSES, Observation, TrackBank  # noqa: E402


def constraints_for(level: int) -> CameraConstraintsDto:
    allowed = ALLOWED_RESOLUTION_LEVELS[level]
    bounds = []
    for target in allowed:
        width, height = SOURCE_REGION_SIZES[target]
        bounds.append(
            CameraLevelBoundsDto(
                resolution_level=target,
                width=960,
                height=540,
                minimum_center_x=width // 2,
                maximum_center_x=IMAGE_WIDTH - width // 2,
                minimum_center_y=height // 2,
                maximum_center_y=IMAGE_HEIGHT - height // 2,
            )
        )
    return CameraConstraintsDto(
        maximum_center_delta=MAXIMUM_CENTER_DELTA_PIXELS[level],
        allowed_resolution_levels=list(allowed),
        center_bounds=bounds,
        full_view_reset_exempt_from_delta=True,
    )


def request_at(
    level: int,
    center_x: int,
    center_y: int,
    image: np.ndarray = None,
    frame: int = 0,
    frame_index: int = 0,
    sequence_id: str = 'test-sequence',
) -> DroneFlybyPredictRequestDto:
    if image is None:
        image = np.zeros((540, 960, 3), dtype=np.uint8)
    view = DroneFlybyViewDto(
        resolution_level=level,
        center_x=center_x,
        center_y=center_y,
        view_id=f'view-{frame}',
        image=encode_image(image),
        image_media_type='image/png',
        width=960,
        height=540,
        source_region_xyxy=list(source_region_for_view(level, center_x, center_y)),
    )
    return DroneFlybyPredictRequestDto(
        sequence_id=sequence_id,
        frame=frame,
        frame_index=frame_index,
        request_id=f'request-{frame}',
        frame_interval_ms=333,
        response_timeout_ms=3333,
        original_width=IMAGE_WIDTH,
        original_height=IMAGE_HEIGHT,
        view=view,
        camera_constraints=constraints_for(level),
    )


def one_hot(object_id: str, strength: float = 0.95) -> np.ndarray:
    posterior = np.full(NUM_CLASSES, (1.0 - strength) / (NUM_CLASSES - 1), dtype=np.float32)
    posterior[OBJECT_CLASSES.index(object_id)] = strength
    return posterior


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

class GeometryTests(unittest.TestCase):
    def test_view_to_source_round_trips_at_every_level(self):
        for level, center in ((0, FULL_FRAME_CENTER), (1, (1440, 810)), (2, (600, 400))):
            region = source_region_for_view(level, *center)
            local = (100.0, 50.0, 300.0, 200.0)
            source = view_to_source(local, region)
            back = source_to_view(source, region)
            for produced, expected in zip(back, local):
                self.assertAlmostEqual(produced, expected, places=6)

    def test_l0_view_box_maps_to_the_whole_frame(self):
        region = source_region_for_view(0, *FULL_FRAME_CENTER)
        source = view_to_source((0.0, 0.0, 960.0, 540.0), region)
        self.assertEqual(source, (0.0, 0.0, float(IMAGE_WIDTH), float(IMAGE_HEIGHT)))

    def test_l2_view_box_maps_into_its_own_sub_region_only(self):
        region = source_region_for_view(2, 1440, 810)
        source = view_to_source((0.0, 0.0, 960.0, 540.0), region)
        self.assertEqual(source, tuple(float(value) for value in region))

    def test_affine_composition_matches_sequential_application(self):
        first = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -7.0]])
        second = np.array([[1.01, 0.02, -3.0], [-0.02, 1.01, 21.0]])
        box = (100.0, 200.0, 180.0, 260.0)
        sequential = apply_affine_to_box(second, apply_affine_to_box(first, box))
        composed = apply_affine_to_box(compose_affine(second, first), box)
        for produced, expected in zip(composed, sequential):
            self.assertAlmostEqual(produced, expected, places=4)

    def test_decompose_reports_a_pure_translation_as_unit_scale(self):
        parts = decompose_affine(np.array([[1.0, 0.0, 40.0], [0.0, 1.0, -15.0]]))
        self.assertAlmostEqual(parts['scale_x'], 1.0, places=9)
        self.assertAlmostEqual(parts['scale_y'], 1.0, places=9)
        self.assertAlmostEqual(parts['shear'], 0.0, places=9)
        self.assertEqual(parts['translation'], (40.0, -15.0))


# --------------------------------------------------------------------------- #
# Camera legality
# --------------------------------------------------------------------------- #

class CameraTests(unittest.TestCase):
    def test_the_shipped_default_holds_l0_and_issues_no_command(self):
        """The default configuration is the one that measured best (0.425).

        It must produce no camera command at all: an unnecessary request is a
        chance for a rejection, and a rejection costs a frame of camera time.
        """
        scheduler = CameraScheduler(SchedulerConfig())
        bank = TrackBank(TrackConfig())
        scheduler.observe(0, *FULL_FRAME_CENTER)
        decision = scheduler.decide(0, FULL_FRAME_CENTER, constraints_for(0), bank)
        self.assertIsNone(decision.requested)
        self.assertEqual(decision.reason, 'hold-current-view')

    def test_move_toward_never_exceeds_the_limit(self):
        current = (1920, 1080)
        for target in ((960, 540), (2880, 1620), (2880, 540), (960, 1620)):
            stepped = move_toward(current, target, 1102.0)
            self.assertLessEqual(
                math.hypot(stepped[0] - current[0], stepped[1] - current[1]), 1102.0
            )

    def test_scheduler_only_ever_requests_legal_transitions(self):
        """Walk a long trajectory and reject the whole run on one illegal move.

        Forced to working_level=1 on purpose. Active resolution is off by
        default on measured evidence, but the machinery has to stay provably
        legal so that turning it back on is a configuration change and not a
        re-verification exercise.
        """
        scheduler = CameraScheduler(SchedulerConfig(working_level=1))
        bank = TrackBank(TrackConfig())
        rng = np.random.default_rng(20260919)
        for index in range(12):
            bank.tracks.append(
                bank._spawn(
                    Observation(
                        box=(
                            float(rng.integers(0, 3600)),
                            float(rng.integers(0, 2000)),
                            0.0,
                            0.0,
                        ),
                        posterior=np.full(NUM_CLASSES, 1.0 / NUM_CLASSES, dtype=np.float32),
                        level=1,
                    ),
                    frame=0,
                )
            )
            track = bank.tracks[-1]
            track.box = (track.box[0], track.box[1], track.box[0] + 60, track.box[1] + 45)

        level, center = 0, FULL_FRAME_CENTER
        moves = 0
        for frame in range(90):
            scheduler.observe(level, center[0], center[1])
            decision = scheduler.decide(level, center, constraints_for(level), bank)
            if decision.requested is None:
                continue
            requested = decision.requested
            rejection = describe_camera_rejection(
                level,
                center,
                requested.resolution_level,
                (requested.center_x, requested.center_y),
            )
            self.assertIsNone(
                rejection,
                f'frame {frame}: illegal move from L{level}{center} -> '
                f'L{requested.resolution_level}'
                f'({requested.center_x},{requested.center_y}): {rejection}',
            )
            self.assertIsInstance(requested.center_x, int)
            self.assertIsInstance(requested.center_y, int)
            level = requested.resolution_level
            center = (requested.center_x, requested.center_y)
            moves += 1
            for track in bank.tracks:
                track.age += 1
        self.assertGreater(moves, 10, 'scheduler never moved the camera')

    def test_scheduler_never_requests_a_direct_l0_to_l2_jump(self):
        scheduler = CameraScheduler(SchedulerConfig(working_level=1))
        bank = TrackBank(TrackConfig())
        decision = scheduler.decide(0, FULL_FRAME_CENTER, constraints_for(0), bank)
        if decision.requested is not None:
            self.assertIn(decision.requested.resolution_level, (0, 1))

    def test_camera_walks_home_when_the_working_level_is_unreachable(self):
        """L0 is not reachable from L2, so the policy must step via L1.

        Settling for "wherever we are" instead would strand the camera at L2,
        looking at 6.25% of the frame, until some unrelated rule fired.
        """
        scheduler = CameraScheduler(SchedulerConfig(working_level=0, max_l2_dwell=99))
        bank = TrackBank(TrackConfig())
        decision = scheduler.decide(2, (1440, 810), constraints_for(2), bank)
        self.assertIsNotNone(decision.requested)
        self.assertEqual(decision.requested.resolution_level, 1)
        self.assertIsNone(
            describe_camera_rejection(
                2, (1440, 810), 1,
                (decision.requested.center_x, decision.requested.center_y),
            )
        )

    def test_periodic_l0_refresh_fires_and_uses_the_full_frame_centre(self):
        scheduler = CameraScheduler(
            SchedulerConfig(working_level=1, l0_refresh_period=3)
        )
        bank = TrackBank(TrackConfig())
        for _ in range(3):
            scheduler.observe(1, 1440, 810)
        decision = scheduler.decide(1, (1440, 810), constraints_for(1), bank)
        self.assertEqual(decision.reason, 'periodic-l0-refresh')
        self.assertEqual(decision.requested.resolution_level, 0)
        self.assertEqual(
            (decision.requested.center_x, decision.requested.center_y), FULL_FRAME_CENTER
        )


# --------------------------------------------------------------------------- #
# Track state
# --------------------------------------------------------------------------- #

class TrackBankTests(unittest.TestCase):
    def setUp(self):
        self.bank = TrackBank(TrackConfig(confirm_hits=2, max_age=4, max_tentative_age=2))

    def observe(self, box, object_id, frame, level=1, strength=0.95, region=None):
        return self.bank.update(
            [Observation(box=box, posterior=one_hot(object_id, strength), level=level)],
            frame=frame,
            view_region=region,
        )

    def test_a_track_is_created_confirmed_and_then_culled_when_stale(self):
        self.observe((100.0, 100.0, 160.0, 150.0), 'tank', 0)
        self.assertEqual(len(self.bank.tracks), 1)
        self.assertFalse(self.bank.tracks[0].confirmed)

        self.bank.predict(None, 0.0)
        self.observe((102.0, 104.0, 162.0, 154.0), 'tank', 1)
        self.assertTrue(self.bank.tracks[0].confirmed)

        for _ in range(6):
            self.bank.predict(None, 0.0)
            self.bank.update([], frame=99)
        self.assertEqual(len(self.bank.tracks), 0)

    def test_an_unconfirmed_track_is_culled_sooner_than_a_confirmed_one(self):
        self.observe((100.0, 100.0, 160.0, 150.0), 'tank', 0)
        for _ in range(3):
            self.bank.predict(None, 0.0)
            self.bank.update([], frame=9)
        self.assertEqual(len(self.bank.tracks), 0)

    def test_off_crop_tracks_survive_because_they_were_not_observable(self):
        """The whole point of the bank: absence outside the view is not evidence."""
        self.observe((100.0, 100.0, 160.0, 150.0), 'tank', 0)
        self.observe((100.0, 100.0, 160.0, 150.0), 'tank', 1)
        far_away = (2000.0, 1500.0, 2960.0, 2040.0)
        for frame in range(2, 5):
            self.bank.predict(None, 0.0)
            self.bank.update([], frame=frame, view_region=far_away)
        self.assertEqual(len(self.bank.tracks), 1)
        self.assertEqual(self.bank.tracks[0].misses_in_view, 0)

    def test_a_track_repeatedly_unseen_inside_the_view_is_removed(self):
        self.observe((100.0, 100.0, 160.0, 150.0), 'tank', 0)
        self.observe((100.0, 100.0, 160.0, 150.0), 'tank', 1)
        covering = (0.0, 0.0, 1920.0, 1080.0)
        for frame in range(2, 6):
            self.bank.predict(None, 0.0)
            self.bank.update([], frame=frame, view_region=covering)
        self.assertEqual(len(self.bank.tracks), 0)

    def test_a_weak_seed_is_overturned_by_repeated_better_evidence(self):
        """Baseline 1 named tank as spacecraft 7 times out of 8.

        A bank that latched its first guess would make that permanent, so the
        evidence decay has to let later, better-resolved observations win.
        """
        box = (100.0, 100.0, 160.0, 150.0)
        self.observe(box, 'spacecraft', 0, level=0, strength=0.55)
        self.assertEqual(self.bank.tracks[0].identity(0.93)[0], 'spacecraft')
        for frame in range(1, 5):
            self.bank.predict(None, 0.0)
            self.observe(box, 'tank', frame, level=2, strength=0.9)
        self.assertEqual(self.bank.tracks[0].identity(0.93)[0], 'tank')

    def test_posterior_ceiling_keeps_a_second_opinion_possible(self):
        box = (100.0, 100.0, 160.0, 150.0)
        for frame in range(12):
            if frame:
                self.bank.predict(None, 0.0)
            self.observe(box, 'hangar', frame, level=2, strength=0.999)
        _, confidence, _, runner_up = self.bank.tracks[0].identity(0.93)
        self.assertLessEqual(confidence, 0.93 + 1e-6)
        self.assertGreater(runner_up, 0.0)

    def test_matching_survives_small_boxes_where_iou_is_uninformative(self):
        self.observe((1000.0, 1000.0, 1006.0, 1006.0), 'ta-ta', 0)
        self.bank.predict(None, 0.0)
        # Three pixels of drift destroys IoU on a six-pixel box but is clearly
        # the same object; the centre gate has to catch it.
        self.observe((1003.0, 1003.0, 1009.0, 1009.0), 'ta-ta', 1)
        self.assertEqual(len(self.bank.tracks), 1)
        self.assertEqual(self.bank.tracks[0].hits, 2)

    def test_gmc_propagation_moves_every_track_including_off_crop_ones(self):
        self.observe((100.0, 100.0, 160.0, 150.0), 'tank', 0)
        self.observe((3000.0, 1800.0, 3060.0, 1850.0), 'condor', 0)
        before = [track.box for track in self.bank.tracks]
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 70.0]])
        self.bank.predict(matrix, 0.9)
        self.assertEqual(len(self.bank.tracks), 2)
        for track, original in zip(self.bank.tracks, before):
            self.assertAlmostEqual(track.box[1], original[1] + 70.0, places=3)
            self.assertAlmostEqual(track.box[0], original[0], places=3)
            self.assertEqual(track.propagation_model, 'gmc')

    def test_constant_velocity_is_used_when_the_global_estimate_is_rejected(self):
        self.observe((100.0, 100.0, 160.0, 150.0), 'tank', 0)
        self.bank.predict(None, 0.0)
        self.observe((100.0, 170.0, 160.0, 220.0), 'tank', 1)
        track = self.bank.tracks[0]
        velocity = track.velocity()
        self.assertIsNotNone(velocity)
        self.assertGreater(velocity[1], 0.0)
        before = track.box
        self.bank.predict(None, 0.0)
        self.assertEqual(track.propagation_model, 'constant-velocity')
        self.assertAlmostEqual(track.box[1], before[1] + velocity[1], places=6)
        self.assertAlmostEqual(track.box[0], before[0] + velocity[0], places=6)


# --------------------------------------------------------------------------- #
# GMC
# --------------------------------------------------------------------------- #

class GmcTests(unittest.TestCase):
    def test_implausible_transforms_are_rejected_outright(self):
        estimator = GlobalMotionEstimator(GmcConfig())
        accepted, _ = estimator._accept(np.array([[1.6, 0.0, 0.0], [0.0, 1.6, 0.0]]), 'affine')
        self.assertFalse(accepted)
        accepted, _ = estimator._accept(np.array([[1.0, 0.9, 0.0], [0.0, 1.0, 0.0]]), 'affine')
        self.assertFalse(accepted)
        accepted, _ = estimator._accept(
            np.array([[1.0, 0.0, 1e6], [0.0, 1.0, 0.0]]), 'affine'
        )
        self.assertFalse(accepted)
        accepted, _ = estimator._accept(
            np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0]]), 'affine'
        )
        self.assertFalse(accepted)

    def test_a_plausible_small_motion_is_accepted(self):
        estimator = GlobalMotionEstimator(GmcConfig())
        accepted, reason = estimator._accept(
            np.array([[1.002, 0.001, 3.0], [-0.001, 1.002, 9.0]]), 'affine'
        )
        self.assertTrue(accepted, reason)

    def test_first_frame_reports_no_estimate_rather_than_the_identity(self):
        estimator = GlobalMotionEstimator(GmcConfig())
        image = np.zeros((540, 960, 3), dtype=np.uint8)
        estimate = estimator.estimate(image, (0, 0, 3840, 2160))
        self.assertFalse(estimate.ok)
        self.assertEqual(estimate.reason, 'no-previous-frame')

    def test_a_featureless_pair_is_rejected_not_guessed(self):
        estimator = GlobalMotionEstimator(GmcConfig())
        blank = np.zeros((540, 960, 3), dtype=np.uint8)
        estimator.estimate(blank, (0, 0, 3840, 2160))
        estimate = estimator.estimate(blank, (0, 0, 3840, 2160))
        self.assertFalse(estimate.ok)

    def test_a_known_translation_is_recovered_in_source_pixels(self):
        """The estimate must come back in source pixels, not received ones.

        At L0 one received pixel is four source pixels, so a 48-pixel shift of
        the transmitted image is a 192-pixel shift of the world. Getting this
        conversion wrong would bias every propagated box by a factor of four
        and would be invisible in any test that checked only "it moved".
        """
        estimator = GlobalMotionEstimator(GmcConfig())
        rng = np.random.default_rng(20260919)
        # Spatially smoothed noise, not white noise. The canvas resamples the
        # received image, and per-pixel noise aliases differently in the two
        # frames, so white noise would be testing optical flow's behaviour on
        # an image that cannot occur rather than the coordinate conversion this
        # test exists for. Terrain has structure; so does this.
        import cv2 as _cv2
        texture = _cv2.GaussianBlur(
            rng.integers(0, 255, (700, 1200, 3), dtype=np.uint8), (0, 0), 2.0
        )
        texture = _cv2.normalize(texture, None, 0, 255, _cv2.NORM_MINMAX).astype(np.uint8)
        shift_received = 48
        first = texture[80:620, 120:1080]
        second = texture[80 + shift_received:620 + shift_received, 120:1080]
        estimator.estimate(first, (0, 0, 3840, 2160))
        estimate = estimator.estimate(second, (0, 0, 3840, 2160))
        self.assertTrue(estimate.ok, estimate.reason)
        # Content moved UP the received image, so the world moved by -192 px.
        self.assertAlmostEqual(float(estimate.matrix[1, 2]), -4.0 * shift_received, delta=16.0)
        self.assertLess(abs(float(estimate.matrix[0, 2])), 16.0)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

class OutputTests(unittest.TestCase):
    def setUp(self):
        self.builder = OutputBuilder(OutputConfig())

    def test_same_class_overlapping_boxes_are_merged(self):
        rows = [
            Candidate('tank', (100.0, 100.0, 200.0, 200.0), 0.9, 1, 0),
            Candidate('tank', (105.0, 105.0, 205.0, 205.0), 0.4, 2, 0),
        ]
        kept = self.builder.deduplicate(rows)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].confidence, 0.9)

    def test_two_tracks_claiming_one_object_as_different_classes_are_merged(self):
        rows = [
            Candidate('tank', (100.0, 100.0, 200.0, 200.0), 0.9, 1, 0),
            Candidate('spacecraft', (100.0, 100.0, 200.0, 200.0), 0.3, 2, 0),
        ]
        self.assertEqual(len(self.builder.deduplicate(rows)), 1)

    def test_a_single_track_may_still_offer_its_secondary_guesses(self):
        """Nine classes scored zero AP in Baseline 1 because they were never
        named. A same-track second guess costs a low-ranked false positive and
        can turn a guaranteed zero into a nonzero AP."""
        rows = [
            Candidate('tank', (100.0, 100.0, 200.0, 200.0), 0.6, 1, 0),
            Candidate('spacecraft', (100.0, 100.0, 200.0, 200.0), 0.3, 1, 1),
        ]
        kept = self.builder.deduplicate(rows)
        self.assertEqual(len(kept), 2)
        self.assertEqual([row.object_id for row in kept], ['tank', 'spacecraft'])

    def test_every_emitted_box_is_a_legal_normalized_box(self):
        bank = TrackBank(TrackConfig(confirm_hits=1))
        for index, object_id in enumerate(OBJECT_CLASSES):
            bank.update(
                [
                    Observation(
                        box=(
                            float(10 + index * 200),
                            float(10 + index * 100),
                            float(90 + index * 200),
                            float(80 + index * 100),
                        ),
                        posterior=one_hot(object_id),
                        level=1,
                    )
                ],
                frame=index,
            )
        annotations, summary = self.builder.build(bank, IMAGE_WIDTH, IMAGE_HEIGHT)
        self.assertGreater(len(annotations), 0)
        self.assertEqual(summary['dropped_invalid'], 0)
        for annotation in annotations:
            x1, y1, x2, y2 = annotation.bbox
            self.assertTrue(0 <= x1 < x2 <= 1)
            self.assertTrue(0 <= y1 < y2 <= 1)
            self.assertTrue(0 <= annotation.confidence <= 1)
            self.assertIn(annotation.object_id, OBJECT_CLASSES)

    def test_a_box_that_has_left_the_frame_is_dropped_not_clamped_to_zero_area(self):
        bank = TrackBank(TrackConfig(confirm_hits=1))
        bank.update(
            [Observation(box=(3839.5, 2159.5, 3900.0, 2200.0), posterior=one_hot('tank'),
                         level=1)],
            frame=0,
        )
        annotations, _ = self.builder.build(bank, IMAGE_WIDTH, IMAGE_HEIGHT)
        for annotation in annotations:
            x1, y1, x2, y2 = annotation.bbox
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)

    def test_confidence_falls_with_staleness_and_poor_geometry(self):
        bank = TrackBank(TrackConfig(confirm_hits=1))
        bank.update(
            [Observation(box=(100.0, 100.0, 200.0, 200.0), posterior=one_hot('tank'), level=1)],
            frame=0,
        )
        track = bank.tracks[0]
        fresh = self.builder.confidence_for(track, 0.9)
        track.age = 5
        track.geometry_confidence = 0.4
        self.assertLess(self.builder.confidence_for(track, 0.9), fresh)


# --------------------------------------------------------------------------- #
# Telemetry
# --------------------------------------------------------------------------- #

class TelemetryTests(unittest.TestCase):
    def test_a_disabled_recorder_accepts_records_and_does_nothing(self):
        recorder = TelemetryRecorder()
        recorder.start()
        recorder.record({'frame': 1})
        self.assertFalse(recorder.active)
        self.assertEqual(recorder.snapshot()['written'], 0)

    def test_an_enabled_recorder_writes_without_touching_the_repo(self):
        from v2.config import TelemetryConfig

        with tempfile.TemporaryDirectory() as directory:
            recorder = TelemetryRecorder(
                TelemetryConfig(
                    enabled=True, directory=directory, store_images=False, queue_size=8
                )
            )
            recorder.start()
            self.assertTrue(recorder.active)
            for frame in range(4):
                recorder.record({'frame': frame, 'frame_index': frame})
            recorder.stop()
            written = list(Path(directory).rglob('*.json'))
            self.assertEqual(len(written), 4)
            self.assertEqual(recorder.snapshot()['errors'], 0)

    def test_a_recorder_on_an_unwritable_path_degrades_quietly(self):
        from v2.config import TelemetryConfig

        with tempfile.TemporaryDirectory() as directory:
            blocker = Path(directory) / 'blocked'
            blocker.write_text('not a directory', encoding='utf-8')
            recorder = TelemetryRecorder(
                TelemetryConfig(enabled=True, directory=str(blocker), queue_size=8)
            )
            recorder.start()
            recorder.record({'frame': 1})
            self.assertFalse(recorder.active)
            self.assertIsNotNone(recorder.snapshot())

    def test_a_failing_recorder_cannot_break_a_prediction(self):
        from v2.pipeline import DroneFlybyPipeline

        pipeline = DroneFlybyPipeline()
        pipeline.load()
        exploding = mock.Mock()
        exploding.active = True
        exploding.record.side_effect = RuntimeError('writer exploded')
        pipeline.telemetry = exploding
        response = pipeline.predict(request_at(0, *FULL_FRAME_CENTER))
        validate_response(response)
        self.assertEqual(response.frame, 0)
        exploding.record.assert_called_once()


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

class PipelineTests(unittest.TestCase):
    @staticmethod
    def textured(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        image = rng.integers(40, 90, (540, 960, 3), dtype=np.uint8)
        # A few high-contrast blobs so discovery has something to find.
        for index in range(6):
            x, y = 80 + index * 130, 90 + (index % 3) * 140
            image[y:y + 14, x:x + 18] = 235
        return image

    def test_a_short_sequence_runs_end_to_end_and_stays_protocol_valid(self):
        from v2.pipeline import DroneFlybyPipeline

        pipeline = DroneFlybyPipeline()
        pipeline.warmup()
        level, center = 0, FULL_FRAME_CENTER
        for index in range(10):
            request = request_at(
                level, center[0], center[1],
                image=self.textured(index), frame=index, frame_index=index,
            )
            response = pipeline.predict(request)
            validate_response(response)
            self.assertEqual(response.request_id, request.request_id)
            self.assertEqual(response.frame, request.frame)
            if response.requested_view is not None:
                requested = response.requested_view
                self.assertIsNone(
                    describe_camera_rejection(
                        level, center,
                        requested.resolution_level,
                        (requested.center_x, requested.center_y),
                    )
                )
                level = requested.resolution_level
                center = (requested.center_x, requested.center_y)

    def test_replay_is_deterministic(self):
        from v2.pipeline import DroneFlybyPipeline

        def run() -> list:
            pipeline = DroneFlybyPipeline()
            pipeline.warmup()
            rows = []
            for index in range(6):
                response = pipeline.predict(
                    request_at(0, *FULL_FRAME_CENTER, image=self.textured(index),
                               frame=index, frame_index=index)
                )
                rows.append(
                    [
                        (a.object_id, tuple(round(float(v), 8) for v in a.bbox),
                         round(float(a.confidence), 8))
                        for a in response.annotations
                    ]
                )
            return rows

        self.assertEqual(run(), run())

    @staticmethod
    def seed(pipeline, boxes) -> None:
        """Put known tracks in the bank.

        The pipeline tests below are about global state and the response
        contract, not about whether a detector fires on synthetic pixels.
        Seeding directly keeps them deterministic and keeps a discovery
        regression from being reported as a state regression.
        """
        for frame in range(CONFIG.tracks.confirm_hits):
            pipeline.bank.update(
                [
                    Observation(box=box, posterior=one_hot(name), level=0)
                    for name, box in boxes
                ],
                frame=frame,
            )

    def test_a_new_sequence_id_resets_all_state(self):
        from v2.pipeline import DroneFlybyPipeline

        pipeline = DroneFlybyPipeline()
        pipeline.warmup()
        pipeline.predict(
            request_at(0, *FULL_FRAME_CENTER, image=self.textured(0), frame=0,
                       frame_index=0)
        )
        self.seed(pipeline, [('tank', (100.0, 100.0, 200.0, 180.0))])
        self.assertGreater(len(pipeline.bank.tracks), 0)

        pipeline.predict(
            request_at(0, *FULL_FRAME_CENTER, image=self.textured(99), frame=0,
                       frame_index=0, sequence_id='a-different-sequence')
        )
        self.assertEqual(pipeline._last_frame_index, 0)
        self.assertEqual(pipeline.scheduler._frame_counter, 1)
        self.assertEqual(pipeline._sequence_id, 'a-different-sequence')

    def test_a_corrupt_image_still_produces_a_valid_response(self):
        from v2.pipeline import DroneFlybyPipeline

        pipeline = DroneFlybyPipeline()
        pipeline.warmup()
        request = request_at(0, *FULL_FRAME_CENTER)
        object.__setattr__(request.view, 'image', 'not-base64-at-all')
        response = pipeline.predict(request)
        validate_response(response)
        self.assertEqual(response.frame, request.frame)

    def test_off_crop_tracks_are_still_emitted_after_the_camera_moves_away(self):
        """The protocol scores every object in the frame, not just the visible
        ones, so state established at L0 has to keep being reported at L1."""
        from v2.pipeline import DroneFlybyPipeline

        pipeline = DroneFlybyPipeline()
        pipeline.warmup()
        pipeline.predict(
            request_at(0, *FULL_FRAME_CENTER, image=self.textured(0), frame=0, frame_index=0)
        )
        # One track deep inside the L1 crop we are about to take, one far
        # outside it. Only the second one proves the point.
        self.seed(pipeline, [
            ('tank', (700.0, 400.0, 800.0, 470.0)),
            ('condor', (3200.0, 1800.0, 3320.0, 1900.0)),
        ])
        before = len(pipeline.bank.tracks)
        self.assertGreater(before, 0)

        response = pipeline.predict(
            request_at(1, 960, 540, image=self.textured(2), frame=2, frame_index=2)
        )
        validate_response(response)
        region = source_region_for_view(1, 960, 540)
        outside = [
            track for track in pipeline.bank.tracks
            if not (region[0] <= track.box[0] and track.box[2] <= region[2]
                    and region[1] <= track.box[1] and track.box[3] <= region[3])
        ]
        self.assertGreater(len(outside), 0, 'no track outside the crop to report')
        self.assertGreater(len(response.annotations), 0)

    def test_skipped_frames_advance_the_state_once_per_missed_frame(self):
        from v2.pipeline import DroneFlybyPipeline

        pipeline = DroneFlybyPipeline()
        pipeline.warmup()
        pipeline.predict(
            request_at(0, *FULL_FRAME_CENTER, image=self.textured(0), frame=0, frame_index=0)
        )
        with mock.patch.object(pipeline.bank, 'predict') as predict:
            pipeline.predict(
                request_at(0, *FULL_FRAME_CENTER, image=self.textured(1), frame=4,
                           frame_index=4)
            )
        self.assertEqual(predict.call_count, 4)


if __name__ == '__main__':
    unittest.main()
