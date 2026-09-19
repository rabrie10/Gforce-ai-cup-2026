"""Every tunable knob of Architecture V2, in one place, overridable by env.

The pipeline reads this once at import. Nothing below is read from the request
path, so a component can be swapped or disabled without touching the loop:
set the environment variable, restart, done.

Naming follows the repo convention of ``DRONE_*`` environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'on'}


# --------------------------------------------------------------------------- #
# Discovery / proposals
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ProposalConfig:
    """Class-agnostic object discovery.

    ``backend`` picks the proposal path. ``saliency`` is the learning-free
    local-contrast blob detector; ``yolo`` reuses a detector's boxes with the
    class head ignored; ``hybrid`` unions both. Recall under a bounded budget
    is what these are chosen on, never local 16-way AP.
    """

    backend: str = field(default_factory=lambda: _env_str('DRONE_PROPOSAL_BACKEND', 'yolo'))
    # Upper bound on proposals handed to the recognizer per frame.
    budget: int = field(default_factory=lambda: _env_int('DRONE_PROPOSAL_BUDGET', 32))
    # Saliency: multi-scale local contrast. Sizes are in received-image pixels.
    saliency_scales: tuple = (3, 7, 15, 31)
    saliency_min_side: float = field(
        default_factory=lambda: _env_float('DRONE_PROPOSAL_MIN_SIDE', 3.0)
    )
    saliency_max_side: float = field(
        default_factory=lambda: _env_float('DRONE_PROPOSAL_MAX_SIDE', 260.0)
    )
    saliency_percentile: float = field(
        default_factory=lambda: _env_float('DRONE_PROPOSAL_PERCENTILE', 99.2)
    )
    # Detector-backed proposals.
    #
    # The ONNX export is the default because it was measured to be worth it,
    # not because it is fashionable: 185.2 ms -> 113.1 ms per frame on four
    # threads, with byte-identical proposal sets on every frame tested.
    # Proposals were 55% of a 346 ms frame against a 333 ms interval, and a
    # frame that arrives late is scored as no detections at all. The .pt file
    # stays in the image and is used automatically if the export is missing.
    yolo_weights: str = field(
        default_factory=lambda: _env_str(
            'DRONE_PROPOSAL_YOLO', str(ROOT / 'models' / 'drone_yolo11n_l0.onnx')
        )
    )
    yolo_fallback_weights: str = field(
        default_factory=lambda: _env_str(
            'DRONE_PROPOSAL_YOLO_FALLBACK', str(ROOT / 'models' / 'drone_yolo11n_l0.pt')
        )
    )
    yolo_confidence: float = field(
        default_factory=lambda: _env_float('DRONE_PROPOSAL_YOLO_CONF', 0.01)
    )
    yolo_iou: float = field(default_factory=lambda: _env_float('DRONE_PROPOSAL_YOLO_IOU', 0.55))
    # Input size at L0. Higher levels are fed at 960 >> level so objects stay
    # at the apparent size the detector was trained on (probe C).
    yolo_imgsz: int = field(default_factory=lambda: _env_int('DRONE_PROPOSAL_YOLO_IMGSZ', 960))
    # ONNX Runtime intra-op threads. Kept equal to the vCPU count; spinning is
    # disabled in the session so these do not steal from torch between calls.
    onnx_threads: int = field(default_factory=lambda: _env_int('DRONE_ONNX_THREADS', 4))
    # Proposals closer than this IoU are merged before recognition.
    merge_iou: float = 0.45


# --------------------------------------------------------------------------- #
# Recognition
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RecognizerConfig:
    """Transfer-oriented crop identity.

    A frozen pretrained backbone embeds object-centred crops; identity comes
    from similarity to a cached multi-view reference gallery, not from a head
    fine-tuned on one physical instance per class.
    """

    enabled: bool = field(default_factory=lambda: _env_bool('DRONE_RECOGNIZER', True))
    backbone: str = field(default_factory=lambda: _env_str('DRONE_RECOGNIZER_BACKBONE', 'resnet18'))
    crop_size: int = field(default_factory=lambda: _env_int('DRONE_RECOGNIZER_CROP', 64))
    # Context multiplier around the proposal box before cropping.
    context: float = field(default_factory=lambda: _env_float('DRONE_RECOGNIZER_CONTEXT', 1.25))
    gallery_path: str = field(
        default_factory=lambda: _env_str('DRONE_GALLERY', str(ROOT / 'models' / 'reference_gallery.npz'))
    )
    # Softmax temperature over cosine similarity when forming the posterior.
    temperature: float = field(default_factory=lambda: _env_float('DRONE_RECOGNIZER_TEMP', 0.07))
    # Exemplars averaged per class when scoring (top-k cosine).
    top_k: int = field(default_factory=lambda: _env_int('DRONE_RECOGNIZER_TOPK', 5))
    # Minimum objectness before a crop is allowed to become an observation.
    # Objectness compares the best class prototype against the best background
    # prototype, so 0.5 means "looks more like an object than like terrain".
    objectness_threshold: float = field(
        default_factory=lambda: _env_float('DRONE_RECOGNIZER_OBJECTNESS', 0.5)
    )
    objectness_temperature: float = field(
        default_factory=lambda: _env_float('DRONE_RECOGNIZER_OBJ_TEMP', 0.03)
    )
    # Crops scored in one forward pass.
    batch: int = field(default_factory=lambda: _env_int('DRONE_RECOGNIZER_BATCH', 32))
    torch_threads: int = field(default_factory=lambda: _env_int('DRONE_TORCH_THREADS', 4))


# --------------------------------------------------------------------------- #
# Global motion compensation
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GmcConfig:
    """Image-derived shared motion, affine first (D-007), similarity fallback."""

    enabled: bool = field(default_factory=lambda: _env_bool('DRONE_GMC', True))
    model: str = field(default_factory=lambda: _env_str('DRONE_GMC_MODEL', 'affine'))
    # Width of the shared source-referenced canvas the estimate is made on.
    # This sets the precision floor: at 960 one canvas pixel is four source
    # pixels, and the RANSAC threshold below is in canvas pixels. Coarser
    # canvases are cheaper but quantize the transform, which matters because a
    # box only has to drift by a fraction of its own size to fall under IoU
    # 0.50. Probe D measures both.
    work_width: int = field(default_factory=lambda: _env_int('DRONE_GMC_WIDTH', 640))
    max_corners: int = field(default_factory=lambda: _env_int('DRONE_GMC_CORNERS', 300))
    # Optical-flow pyramid depth. Must span the largest displacement the canvas
    # can show, including the multi-frame jump a skipped frame produces.
    pyramid_levels: int = field(default_factory=lambda: _env_int('DRONE_GMC_PYRAMID', 5))
    quality_level: float = 0.01
    min_distance: int = 8
    # Quality gates. A transform failing any of these is discarded, never
    # applied at reduced weight: one bad global estimate must not corrupt
    # every track.
    min_inliers: int = field(default_factory=lambda: _env_int('DRONE_GMC_MIN_INLIERS', 20))
    min_inlier_ratio: float = field(
        default_factory=lambda: _env_float('DRONE_GMC_MIN_INLIER_RATIO', 0.45)
    )
    # Per-frame plausibility: a 3 fps survey flight cannot scale or shear much.
    max_scale_deviation: float = field(
        default_factory=lambda: _env_float('DRONE_GMC_MAX_SCALE_DEV', 0.25)
    )
    max_shear: float = field(default_factory=lambda: _env_float('DRONE_GMC_MAX_SHEAR', 0.15))
    max_translation_fraction: float = field(
        default_factory=lambda: _env_float('DRONE_GMC_MAX_TRANSLATION', 0.5)
    )
    # Reprojection threshold for the RANSAC estimator, in work-image pixels.
    ransac_threshold: float = 3.0


# --------------------------------------------------------------------------- #
# Track bank
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TrackConfig:
    """Global object state, in source pixels, maintained across camera moves."""

    # A proposal matches a track at or above this IoU (or within the gate below).
    match_iou: float = field(default_factory=lambda: _env_float('DRONE_TRACK_MATCH_IOU', 0.30))
    # Center-distance gate as a multiple of the track's diagonal, used when the
    # boxes are too small for IoU to be informative.
    match_center_gate: float = field(
        default_factory=lambda: _env_float('DRONE_TRACK_CENTER_GATE', 1.2)
    )
    # Frames a track survives without any direct observation.
    max_age: int = field(default_factory=lambda: _env_int('DRONE_TRACK_MAX_AGE', 8))
    # Frames a track that was never confirmed survives.
    max_tentative_age: int = field(default_factory=lambda: _env_int('DRONE_TRACK_MAX_TENTATIVE', 2))
    # Direct observations before a track is emitted at full confidence.
    confirm_hits: int = field(default_factory=lambda: _env_int('DRONE_TRACK_CONFIRM', 2))
    # Hard ceiling on bank size, so a landscape where the reject option
    # degrades cannot turn into unbounded per-frame latency.
    max_tracks: int = field(default_factory=lambda: _env_int('DRONE_TRACK_MAX', 48))
    # Apparent short side, in received pixels, below which a miss carries no
    # information. The Baseline 1 audit localized 0 of 60 sub-8px appearances.
    detectable_short_side: float = field(
        default_factory=lambda: _env_float('DRONE_TRACK_DETECTABLE', 8.0)
    )
    # Identity posterior update: evidence weight of one observation, scaled by
    # the observation's resolution level and the recognizer's own margin.
    evidence_decay: float = field(default_factory=lambda: _env_float('DRONE_TRACK_EVIDENCE_DECAY', 0.82))
    # Cap on any single class's posterior mass, so a weak seed can still be
    # overturned by later, better-resolved evidence.
    posterior_ceiling: float = field(
        default_factory=lambda: _env_float('DRONE_TRACK_POSTERIOR_CEILING', 0.93)
    )
    # Observation weight by the level it was made at. L2 detail is worth more
    # than L0 detail for identity; this is the only place that is encoded.
    level_weight: tuple = (0.35, 1.0, 1.35)


# --------------------------------------------------------------------------- #
# Camera scheduling
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SchedulerConfig:
    """Deterministic active-resolution policy. No RL, no POMDP."""

    enabled: bool = field(default_factory=lambda: _env_bool('DRONE_SCHEDULER', True))
    # Level the policy treats as its working level for routine identity refresh.
    #
    # Measured, not assumed. End to end on Helsinki, offline:
    #     working_level=0 -> 0.425 mAP@0.50
    #     working_level=1 -> 0.341 mAP@0.50
    #
    # Active resolution does not currently pay for itself, and probe C says
    # exactly why. The proposal detector needs >=16 px of apparent size *in its
    # own input* (recall 0.985 there, 0.41 at 8-16 px, 0.00 below 8 px), and it
    # was trained only on L0 renders - so at L1 it has to be fed the view
    # downscaled by two to stay in its trained scale range, which hands back
    # precisely the detail the zoom just bought. L1 therefore costs 75% of the
    # per-frame coverage and returns only the recognizer's small detail gain.
    #
    # The machinery is built, tested and one variable away. It becomes the
    # right default as soon as discovery can exploit resolution, which means a
    # proposal detector trained across L0/L1/L2 renders with scale
    # augmentation. That is the highest-value open item in this architecture.
    working_level: int = field(default_factory=lambda: _env_int('DRONE_SCHED_LEVEL', 0))
    # Frames between forced L0 global refreshes. Chosen against the measured
    # constant-velocity propagation horizon (IoU>=0.5 holds to about h=4-6).
    l0_refresh_period: int = field(default_factory=lambda: _env_int('DRONE_SCHED_L0_PERIOD', 6))
    # Consecutive frames the camera may stay at L2 before returning to L1.
    max_l2_dwell: int = field(default_factory=lambda: _env_int('DRONE_SCHED_L2_DWELL', 2))
    # A track is worth an L2 visit only when it is this uncertain and this small.
    l2_identity_threshold: float = field(
        default_factory=lambda: _env_float('DRONE_SCHED_L2_IDENTITY', 0.45)
    )
    l2_max_short_side: float = field(
        default_factory=lambda: _env_float('DRONE_SCHED_L2_MAX_SIDE', 60.0)
    )
    # Scoring weights for candidate crop centres.
    weight_unknown: float = 3.0
    weight_stale: float = 1.0
    weight_tiny: float = 1.5
    weight_ambiguous: float = 2.0
    weight_coverage: float = 0.8
    weight_boundary_risk: float = 1.2
    # Discount applied to a candidate that needs a level transition.
    transition_cost: float = field(default_factory=lambda: _env_float('DRONE_SCHED_TRANSITION_COST', 0.6))
    # Fraction of frame height treated as the entry band new objects arrive in.
    entry_band: float = field(default_factory=lambda: _env_float('DRONE_SCHED_ENTRY_BAND', 0.28))
    # Weight of the never-yet-observed-region term, which keeps the deterministic
    # sweep alive when the track bank has nothing urgent to say.
    weight_unvisited: float = 1.0


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OutputConfig:
    """Confidence, deduplication and the response contract."""

    # Tracks below this emitted confidence are dropped entirely.
    min_confidence: float = field(default_factory=lambda: _env_float('DRONE_OUT_MIN_CONF', 0.02))
    # At most this many annotations, well under the protocol's 500.
    max_annotations: int = field(default_factory=lambda: _env_int('DRONE_OUT_MAX', 180))
    # Same-class boxes above this IoU are merged; the evaluator does no NMS.
    dedup_iou: float = field(default_factory=lambda: _env_float('DRONE_OUT_DEDUP_IOU', 0.55))
    # Cross-class duplicate suppression: a lower-confidence box of a different
    # class overlapping this much is a duplicate of the same physical object.
    cross_class_iou: float = field(default_factory=lambda: _env_float('DRONE_OUT_XCLASS_IOU', 0.80))
    # How many classes each track may name, best posterior first.
    #
    # COCO AP is computed per class independently, so a box labelled "tank"
    # can only ever cost tank's precision - it cannot hurt condor. Baseline 1
    # emitted 7 of 16 class IDs and was therefore capped at 0.44 before any
    # other error. Naming a second and third candidate at proportionally lower
    # confidence turns a guaranteed zero into a ranked guess, and the ranking
    # is what AP actually integrates over.
    emit_top_k: int = field(default_factory=lambda: _env_int('DRONE_OUT_TOPK', 3))
    # A secondary guess is only worth emitting if the posterior really is split.
    secondary_min_posterior: float = field(
        default_factory=lambda: _env_float('DRONE_OUT_SECONDARY_MIN', 0.06)
    )
    # Each successive guess is damped so it cannot outrank a first choice of
    # the same class made by another track.
    secondary_damping: float = field(
        default_factory=lambda: _env_float('DRONE_OUT_SECONDARY_DAMP', 0.85)
    )
    # Confidence floors per staleness step, applied multiplicatively alongside
    # the geometry and identity terms rather than as a blind age decay.
    staleness_half_life: float = field(
        default_factory=lambda: _env_float('DRONE_OUT_STALENESS_HALFLIFE', 5.0)
    )


# --------------------------------------------------------------------------- #
# Telemetry
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TelemetryConfig:
    """Non-blocking validation capture. Off unless explicitly switched on."""

    enabled: bool = field(default_factory=lambda: _env_bool('DRONE_TELEMETRY', False))
    directory: str = field(
        default_factory=lambda: _env_str('DRONE_TELEMETRY_DIR', str(ROOT / 'telemetry'))
    )
    # Store the received image alongside the record.
    #
    # Measured, paired, three runs each on the dev host: pipeline total is
    # 402.4 / 403.4 / 401.9 ms with capture off and 417.0 / 438.4 / 423.5 ms
    # with it on. So capture costs about +24 ms, roughly 6% - real, bounded,
    # and well short of the 333 ms frame interval. The cost is the writer
    # thread competing for CPU and the GIL with the two inference runtimes,
    # not the enqueue call, which is off the request path.
    #
    # ``image_stride`` exists because a first attempt at attributing that cost
    # blamed PNG writing; dropping from 25 images to 5 changed nothing
    # measurable, so images are not the driver. The stride is kept anyway
    # because it cuts disk from 28 MB to 5.7 MB per 25 frames for a
    # representative visual sample, and metadata - which is what failure
    # attribution actually runs on - is still captured for every frame.
    store_images: bool = field(default_factory=lambda: _env_bool('DRONE_TELEMETRY_IMAGES', True))
    image_stride: int = field(default_factory=lambda: _env_int('DRONE_TELEMETRY_IMAGE_STRIDE', 5))
    # Bounded queue: when full, records are dropped rather than blocking.
    queue_size: int = field(default_factory=lambda: _env_int('DRONE_TELEMETRY_QUEUE', 256))


@dataclass(frozen=True)
class Config:
    proposals: ProposalConfig = field(default_factory=ProposalConfig)
    recognizer: RecognizerConfig = field(default_factory=RecognizerConfig)
    gmc: GmcConfig = field(default_factory=GmcConfig)
    tracks: TrackConfig = field(default_factory=TrackConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)


CONFIG = Config()
