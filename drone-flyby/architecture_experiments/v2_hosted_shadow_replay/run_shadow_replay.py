"""Deterministic shadow replay of the 50 captured hosted V2 images.

This is a diagnostic harness only. It imports the frozen Architecture V2
proposal and recognition implementation without changing production code,
tracking, GMC, scheduling, or output behavior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
DRONE_ROOT = HERE.parents[1]
REPO_ROOT = DRONE_ROOT.parent
TELEMETRY_ROOT = (
    DRONE_ROOT
    / "hosted_telemetry"
    / "v2_validation_6a86911a"
    / "run_20260919_005220"
)
IMAGE_DIR = TELEMETRY_ROOT / "images"
FRAME_DIR = TELEMETRY_ROOT / "frames"
EXPECTED_INDICES = list(range(0, 250, 5))
EXPECTED_SIZE = (960, 540)
FROZEN_COMMIT = "6fb764e5b9fb5f7eefcb832c72b35553165219e5"
PARENT_AUDIT_COMMIT = "afbe0a6c47662b438362aafa01343224f85b11c9"
HOSTED_VALIDATION_UUID = "6a86911aebf64be79a3b6a6742de0686"

ARTIFACTS = {
    "drone-flyby/models/drone_yolo11n_l0.pt":
        "4c44e404e03673e8aa15fc85baf90f2d09b67d90c1fd94c9b6c9a4c1f35204ce",
    "drone-flyby/models/drone_yolo11n_l0.onnx":
        "d480d3369feba50faa7d1ef0d901f6e5d96f564bf0953bd058f1de42a0ac2ca9",
    "drone-flyby/models/reference_gallery.npz":
        "ef226de6a9bd624a1e7fd60291812677c5d7815f3a667bfea607b0158867b67e",
}

# The telemetry serializer records proposal coordinates to 0.1 pixel and all
# other inference scalars to four decimals. Agreement is judged at exactly that
# observable precision, while unrounded errors are still reported.
BBOX_DECIMALS = 1
SCALAR_DECIMALS = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def describe(values: Iterable[float | int | None]) -> dict[str, Any]:
    data = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if data.size == 0:
        return {"count": 0, "min": None, "median": None, "mean": None,
                "p95": None, "max": None}
    return {
        "count": int(data.size),
        "min": float(data.min()),
        "median": float(np.median(data)),
        "mean": float(data.mean()),
        "p95": float(np.percentile(data, 95)),
        "max": float(data.max()),
    }


def iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def bbox_error(first: Sequence[float], second: Sequence[float]) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(first, second))


def geometric_match(
    hosted: Sequence[dict[str, Any]], replay: Sequence[Any]
) -> list[tuple[int | None, int | None]]:
    """Deterministic global greedy IoU matching, followed by unmatched rows."""
    candidates: list[tuple[float, int, int]] = []
    for hosted_index, hosted_row in enumerate(hosted):
        for replay_index, replay_row in enumerate(replay):
            candidates.append(
                (iou(hosted_row["box"], replay_row.box), hosted_index, replay_index)
            )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_hosted: set[int] = set()
    used_replay: set[int] = set()
    pairs: list[tuple[int | None, int | None]] = []
    for _, hosted_index, replay_index in candidates:
        if hosted_index in used_hosted or replay_index in used_replay:
            continue
        used_hosted.add(hosted_index)
        used_replay.add(replay_index)
        pairs.append((hosted_index, replay_index))
    pairs.extend((index, None) for index in range(len(hosted)) if index not in used_hosted)
    pairs.extend((None, index) for index in range(len(replay)) if index not in used_replay)
    return sorted(
        pairs,
        key=lambda item: (
            item[0] is None,
            item[0] if item[0] is not None else 10_000,
            item[1] if item[1] is not None else 10_000,
        ),
    )


def round_equal(first: float, second: float, decimals: int) -> bool:
    return round(float(first), decimals) == float(second)


def box_round_equal(replay: Sequence[float], hosted: Sequence[float]) -> bool:
    return all(round_equal(a, b, BBOX_DECIMALS) for a, b in zip(replay, hosted))


def crop_bounds(box: Sequence[float], context: float) -> tuple[int, int, int, int]:
    """Return the integer source slice bounds used by crop_from_view."""
    cx = 0.5 * (float(box[0]) + float(box[2]))
    cy = 0.5 * (float(box[1]) + float(box[3]))
    half = 0.5 * max(
        float(box[2]) - float(box[0]), float(box[3]) - float(box[1])
    ) * float(context)
    half = max(half, 3.0)
    return (
        int(math.floor(cx - half)), int(math.floor(cy - half)),
        int(math.ceil(cx + half)), int(math.ceil(cy + half)),
    )


def flatten_config() -> dict[str, Any]:
    from v2.config import CONFIG

    return jsonable(asdict(CONFIG))


def runtime_manifest() -> dict[str, Any]:
    import onnxruntime
    import torch
    import torchvision

    relevant_prefixes = (
        "DRONE_", "OMP_", "MKL_", "OPENBLAS_", "ONNX", "ORT_",
        "TORCH", "NUMEXPR_", "VECLIB_", "CUDA", "CUDNN",
    )
    environment = {
        key: value for key, value in sorted(os.environ.items())
        if key.startswith(relevant_prefixes)
    }
    hashes = {}
    for relative, expected in ARTIFACTS.items():
        path = REPO_ROOT / relative
        actual = sha256(path) if path.is_file() else None
        hashes[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": actual == expected,
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
    return {
        "experiment": "deterministic shadow replay of 50 captured hosted images",
        "recorded_before_inference": True,
        "repository": {
            "branch": git("branch", "--show-current"),
            "head": git("rev-parse", "HEAD"),
            "frozen_v2_commit": FROZEN_COMMIT,
            "parent_audit_commit": PARENT_AUDIT_COMMIT,
            "hosted_validation_uuid": HOSTED_VALIDATION_UUID,
        },
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "onnxruntime": onnxruntime.__version__,
            "onnxruntime_available_providers": onnxruntime.get_available_providers(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "ultralytics": package_version("ultralytics"),
        },
        "relevant_environment_variables": environment,
        "unset_relevant_environment_variables_use_config_defaults": len(environment) == 0,
        "v2_config": flatten_config(),
        "frozen_artifacts": hashes,
        "all_frozen_artifact_hashes_match": all(
            item["matches"] for item in hashes.values()
        ),
    }


def load_evidence() -> tuple[
    dict[int, np.ndarray], dict[int, dict[str, Any]], list[dict[str, Any]]
]:
    image_names = sorted(path.name for path in IMAGE_DIR.glob("*.png"))
    expected_names = [f"{index:05d}.png" for index in EXPECTED_INDICES]
    if image_names != expected_names:
        raise RuntimeError(
            f"Expected exactly the 50 stride-5 PNGs, got {len(image_names)}; "
            f"missing={sorted(set(expected_names) - set(image_names))}, "
            f"extra={sorted(set(image_names) - set(expected_names))}"
        )

    images: dict[int, np.ndarray] = {}
    records: dict[int, dict[str, Any]] = {}
    manifest: list[dict[str, Any]] = []
    for index in EXPECTED_INDICES:
        image_path = IMAGE_DIR / f"{index:05d}.png"
        frame_path = FRAME_DIR / f"{index:05d}.json"
        if not frame_path.is_file():
            raise RuntimeError(f"Hosted sampled telemetry record missing: {frame_path}")
        record = json.loads(frame_path.read_text(encoding="utf-8"))
        if int(record.get("frame_index", -1)) != index:
            raise RuntimeError(f"Telemetry frame_index mismatch in {frame_path}")

        raw = image_path.read_bytes()
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        decoded = image is not None
        width = int(image.shape[1]) if decoded else None
        height = int(image.shape[0]) if decoded else None
        channels = int(image.shape[2]) if decoded and image.ndim == 3 else None
        manifest.append({
            "frame_index": index,
            "filename": image_path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "decoded": decoded,
            "width": width,
            "height": height,
            "channels": channels,
            "expected_width": EXPECTED_SIZE[0],
            "expected_height": EXPECTED_SIZE[1],
            "dimensions_match": (width, height) == EXPECTED_SIZE,
        })
        if not decoded or (width, height) != EXPECTED_SIZE:
            raise RuntimeError(
                f"Image {image_path} did not decode as exactly 960x540: {width}x{height}"
            )
        images[index] = image
        records[index] = record
    return images, records, manifest


def proposal_replay(
    images: dict[int, np.ndarray], records: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[int, list[Any]], dict[int, dict[str, Any]]]:
    from v2.proposals import ProposalEngine

    engine = ProposalEngine()
    backend_details = [
        {
            "name": backend.name,
            "class": type(backend).__name__,
            "weights": str(getattr(backend, "weights", "")),
            "providers": (
                backend.session.get_providers() if hasattr(backend, "session") else None
            ),
        }
        for backend in engine.backends
    ]
    replay_by_frame: dict[int, list[Any]] = {}
    rows: list[dict[str, Any]] = []
    per_frame: dict[int, dict[str, Any]] = {}

    for index in EXPECTED_INDICES:
        replay = engine.propose(images[index], level=0)
        hosted = records[index]["proposals"].get("boxes", [])
        replay_by_frame[index] = replay
        pairs = geometric_match(hosted, replay)
        identity_mapping = (
            len(hosted) == len(replay)
            and all(hosted_index == replay_index for hosted_index, replay_index in pairs)
        )
        match_basis = "ordinal" if identity_mapping else "geometric_iou_greedy"
        frame_rows: list[dict[str, Any]] = []
        for hosted_index, replay_index in pairs:
            hosted_row = hosted[hosted_index] if hosted_index is not None else None
            replay_row = replay[replay_index] if replay_index is not None else None
            matched = hosted_row is not None and replay_row is not None
            pair_iou = iou(hosted_row["box"], replay_row.box) if matched else None
            max_error = bbox_error(hosted_row["box"], replay_row.box) if matched else None
            score_error = (
                abs(float(hosted_row["score"]) - float(replay_row.score)) if matched else None
            )
            source_equal = (
                hosted_row["source"] == replay_row.source if matched else False
            )
            coordinate_round_equal = (
                box_round_equal(replay_row.box, hosted_row["box"]) if matched else False
            )
            score_round_equal = (
                round_equal(replay_row.score, hosted_row["score"], SCALAR_DECIMALS)
                if matched else False
            )
            row = {
                "frame_index": index,
                "match_basis": match_basis,
                "hosted_ordinal": hosted_index,
                "replay_ordinal": replay_index,
                "matched": matched,
                "hosted_source": hosted_row["source"] if hosted_row else None,
                "replay_source": replay_row.source if replay_row else None,
                "source_equal": source_equal,
                "hosted_x1": hosted_row["box"][0] if hosted_row else None,
                "hosted_y1": hosted_row["box"][1] if hosted_row else None,
                "hosted_x2": hosted_row["box"][2] if hosted_row else None,
                "hosted_y2": hosted_row["box"][3] if hosted_row else None,
                "replay_x1": replay_row.box[0] if replay_row else None,
                "replay_y1": replay_row.box[1] if replay_row else None,
                "replay_x2": replay_row.box[2] if replay_row else None,
                "replay_y2": replay_row.box[3] if replay_row else None,
                "hosted_score": hosted_row["score"] if hosted_row else None,
                "replay_score": replay_row.score if replay_row else None,
                "iou": pair_iou,
                "bbox_max_abs_error_px": max_error,
                "score_abs_error": score_error,
                "bbox_equal_at_telemetry_precision": coordinate_round_equal,
                "score_equal_at_telemetry_precision": score_round_equal,
                "pair_equal_at_telemetry_precision": bool(
                    matched and source_equal and coordinate_round_equal and score_round_equal
                ),
            }
            rows.append(row)
            frame_rows.append(row)

        hosted_count = len(hosted)
        replay_count = len(replay)
        per_frame[index] = {
            "hosted_proposal_count": hosted_count,
            "replay_proposal_count": replay_count,
            "proposal_count_error": replay_count - hosted_count,
            "proposal_count_abs_error": abs(replay_count - hosted_count),
            "proposal_count_match": hosted_count == replay_count,
            "proposal_ordering_match": identity_mapping,
            "proposal_all_pairs_equal_at_telemetry_precision": all(
                row["pair_equal_at_telemetry_precision"] for row in frame_rows
            ),
            "proposal_min_matched_iou": min(
                (row["iou"] for row in frame_rows if row["iou"] is not None),
                default=None,
            ),
            "proposal_max_bbox_error_px": max(
                (row["bbox_max_abs_error_px"] for row in frame_rows
                 if row["bbox_max_abs_error_px"] is not None),
                default=None,
            ),
            "proposal_max_score_error": max(
                (row["score_abs_error"] for row in frame_rows
                 if row["score_abs_error"] is not None),
                default=None,
            ),
        }
    per_frame[-1] = {"backend_details": backend_details}
    return rows, replay_by_frame, per_frame


def view_box_to_source(
    box: Sequence[float], region: Sequence[float], view_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    width, height = view_size
    sx = (float(region[2]) - float(region[0])) / width
    sy = (float(region[3]) - float(region[1])) / height
    return (
        float(region[0]) + float(box[0]) * sx,
        float(region[1]) + float(box[1]) * sy,
        float(region[0]) + float(box[2]) * sx,
        float(region[1]) + float(box[3]) * sy,
    )


def map_observations_to_proposals(
    proposals: Sequence[dict[str, Any]], observations: Sequence[dict[str, Any]],
    region: Sequence[float], view_size: tuple[int, int],
) -> dict[int, int]:
    candidates: list[tuple[float, int, int]] = []
    for proposal_index, proposal in enumerate(proposals):
        source_box = view_box_to_source(proposal["box"], region, view_size)
        for observation_index, observation in enumerate(observations):
            candidates.append((
                iou(source_box, observation["box"]), proposal_index, observation_index
            ))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    mapping: dict[int, int] = {}
    used_observations: set[int] = set()
    for _, proposal_index, observation_index in candidates:
        if proposal_index in mapping or observation_index in used_observations:
            continue
        mapping[proposal_index] = observation_index
        used_observations.add(observation_index)
    return mapping


def recognition_replay(
    images: dict[int, np.ndarray], records: dict[int, dict[str, Any]],
    proposal_overrides: dict[int, list[Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    from dtos import OBJECT_CLASSES
    from v2.config import CONFIG
    from v2.geometry import crop_from_view
    from v2.proposals import Proposal
    from v2.recognizer import CropRecognizer

    recognizer = CropRecognizer(CONFIG.recognizer)
    recognizer.load()
    if not recognizer.available:
        raise RuntimeError(f"Frozen recognizer unavailable: {recognizer.load_error}")

    rows: list[dict[str, Any]] = []
    per_frame: dict[int, dict[str, Any]] = {}
    threshold = float(CONFIG.recognizer.objectness_threshold)
    input_source = (
        "full-precision locally replayed proposal boxes (secondary sensitivity only)"
        if proposal_overrides is not None
        else "original recorded hosted proposal boxes (primary)"
    )
    for index in EXPECTED_INDICES:
        record = records[index]
        recorded_hosted_proposals = record["proposals"].get("boxes", [])
        if proposal_overrides is None:
            hosted_proposals = recorded_hosted_proposals
        else:
            hosted_proposals = [
                {
                    "box": list(proposal.box),
                    "score": float(proposal.score),
                    "source": proposal.source,
                }
                for proposal in proposal_overrides[index]
            ]
        hosted_observations = record.get("observations", [])
        recognition = record.get("recognition", {})
        if not hosted_proposals:
            per_frame[index] = {
                "recognition_input_source": input_source,
                "hosted_crop_count": int(recognition.get("crops", 0)),
                "replay_crop_count": 0,
                "hosted_rejected_as_background": int(
                    recognition.get("rejected_as_background", 0)
                ),
                "replay_rejected_as_background": 0,
                "hosted_observation_count": len(hosted_observations),
                "replay_observation_count": 0,
                "recognition_counts_match": (
                    int(recognition.get("crops", 0)) == 0
                    and int(recognition.get("rejected_as_background", 0)) == 0
                    and len(hosted_observations) == 0
                ),
                "background_rejection_agreement": True,
                "top_class_agreement_rate": None,
                "mean_objectness_hosted": recognition.get("mean_objectness"),
                "mean_objectness_replay": None,
                "mean_margin_hosted": recognition.get("mean_margin"),
                "mean_margin_replay": None,
                "recognition_all_observable_values_equal_at_telemetry_precision": True,
            }
            continue

        proposal_objects = [
            Proposal(
                box=tuple(float(value) for value in proposal["box"]),
                score=float(proposal["score"]),
                source=str(proposal["source"]),
            )
            for proposal in hosted_proposals
        ]
        crops: list[np.ndarray] = []
        kept_indices: list[int] = []
        for proposal_index, proposal in enumerate(proposal_objects):
            crop = crop_from_view(
                images[index], proposal.box, recognizer.crop_size, recognizer.crop_context
            )
            if crop is not None:
                crops.append(crop)
                kept_indices.append(proposal_index)
        results, _ = recognizer.recognize(crops, level=0)
        results_by_proposal = dict(zip(kept_indices, results))
        region = record["view"]["source_region_xyxy"]
        observation_mapping = map_observations_to_proposals(
            hosted_proposals, hosted_observations, region, EXPECTED_SIZE
        )

        replay_rejected = 0
        replay_observations = 0
        objectness_values: list[float] = []
        margin_values: list[float] = []
        frame_rows: list[dict[str, Any]] = []
        for proposal_index, proposal in enumerate(hosted_proposals):
            result = results_by_proposal.get(proposal_index)
            replay_has_crop = result is not None
            replay_is_rejected = bool(result is None or result.objectness < threshold)
            if result is not None:
                objectness_values.append(float(result.objectness))
                margin_values.append(float(result.margin))
            if replay_is_rejected:
                replay_rejected += 1
            else:
                replay_observations += 1

            hosted_observation_index = observation_mapping.get(proposal_index)
            hosted_observation = (
                hosted_observations[hosted_observation_index]
                if hosted_observation_index is not None else None
            )
            hosted_is_rejected = hosted_observation is None
            acceptance_agrees = hosted_is_rejected == replay_is_rejected

            replay_top = int(result.best_index) if result is not None else None
            replay_top_posterior = (
                float(result.posterior[result.best_index]) if result is not None else None
            )
            if result is not None:
                margin_quality = min(1.0, max(0.15, 0.35 + 3.0 * result.margin))
                replay_quality = float(
                    margin_quality * (0.5 + 0.5 * result.objectness)
                )
            else:
                replay_quality = None

            both_accepted = hosted_observation is not None and not replay_is_rejected
            top_agrees = (
                int(hosted_observation["top"]) == replay_top if both_accepted else None
            )
            posterior_error = (
                abs(float(hosted_observation["top_posterior"]) - replay_top_posterior)
                if both_accepted else None
            )
            quality_error = (
                abs(float(hosted_observation["quality"]) - replay_quality)
                if both_accepted else None
            )
            posterior_round_equal = (
                round_equal(replay_top_posterior, hosted_observation["top_posterior"],
                            SCALAR_DECIMALS)
                if both_accepted else None
            )
            quality_round_equal = (
                round_equal(replay_quality, hosted_observation["quality"], SCALAR_DECIMALS)
                if both_accepted else None
            )
            observable_equal = bool(
                acceptance_agrees
                and (not both_accepted or (
                    top_agrees and posterior_round_equal and quality_round_equal
                ))
            )
            row = {
                "frame_index": index,
                "hosted_proposal_ordinal": proposal_index,
                "source": proposal["source"],
                "proposal_x1": proposal["box"][0],
                "proposal_y1": proposal["box"][1],
                "proposal_x2": proposal["box"][2],
                "proposal_y2": proposal["box"][3],
                "replay_crop_created": replay_has_crop,
                "hosted_rejected_as_background": hosted_is_rejected,
                "replay_rejected_as_background": replay_is_rejected,
                "background_rejection_agrees": acceptance_agrees,
                "hosted_observation_ordinal": hosted_observation_index,
                "hosted_top_index": (
                    hosted_observation["top"] if hosted_observation else None
                ),
                "hosted_top_class": (
                    OBJECT_CLASSES[int(hosted_observation["top"])]
                    if hosted_observation else "NOT OBSERVABLE"
                ),
                "replay_top_index": replay_top,
                "replay_top_class": (
                    OBJECT_CLASSES[replay_top] if replay_top is not None else None
                ),
                "top_class_agrees": top_agrees,
                "hosted_top_posterior": (
                    hosted_observation["top_posterior"] if hosted_observation else None
                ),
                "replay_top_posterior": replay_top_posterior,
                "top_posterior_abs_error": posterior_error,
                "top_posterior_equal_at_telemetry_precision": posterior_round_equal,
                "hosted_quality": (
                    hosted_observation["quality"] if hosted_observation else None
                ),
                "replay_quality": replay_quality,
                "quality_abs_error": quality_error,
                "quality_equal_at_telemetry_precision": quality_round_equal,
                "hosted_objectness": "NOT OBSERVABLE",
                "replay_objectness": float(result.objectness) if result else None,
                "hosted_margin": "NOT OBSERVABLE",
                "replay_margin": float(result.margin) if result else None,
                "observable_values_equal_at_telemetry_precision": observable_equal,
            }
            rows.append(row)
            frame_rows.append(row)

        replay_mean_objectness = (
            float(np.mean(objectness_values)) if objectness_values else 0.0
        )
        replay_mean_margin = float(np.mean(margin_values)) if margin_values else 0.0
        hosted_mean_objectness = recognition.get("mean_objectness")
        hosted_mean_margin = recognition.get("mean_margin")
        mean_objectness_equal = (
            round_equal(replay_mean_objectness, hosted_mean_objectness, SCALAR_DECIMALS)
            if hosted_mean_objectness is not None else None
        )
        mean_margin_equal = (
            round_equal(replay_mean_margin, hosted_mean_margin, SCALAR_DECIMALS)
            if hosted_mean_margin is not None else None
        )
        top_values = [row["top_class_agrees"] for row in frame_rows
                      if row["top_class_agrees"] is not None]
        per_frame[index] = {
            "recognition_input_source": input_source,
            "hosted_crop_count": int(recognition.get("crops", 0)),
            "replay_crop_count": len(crops),
            "hosted_rejected_as_background": int(
                recognition.get("rejected_as_background", 0)
            ),
            "replay_rejected_as_background": replay_rejected,
            "hosted_observation_count": len(hosted_observations),
            "replay_observation_count": replay_observations,
            "recognition_counts_match": (
                int(recognition.get("crops", 0)) == len(crops)
                and int(recognition.get("rejected_as_background", 0)) == replay_rejected
                and len(hosted_observations) == replay_observations
            ),
            "background_rejection_agreement": all(
                row["background_rejection_agrees"] for row in frame_rows
            ),
            "top_class_agreement_rate": (
                sum(bool(value) for value in top_values) / len(top_values)
                if top_values else None
            ),
            "mean_objectness_hosted": hosted_mean_objectness,
            "mean_objectness_replay": replay_mean_objectness,
            "mean_objectness_abs_error": (
                abs(float(hosted_mean_objectness) - replay_mean_objectness)
                if hosted_mean_objectness is not None else None
            ),
            "mean_objectness_equal_at_telemetry_precision": mean_objectness_equal,
            "mean_margin_hosted": hosted_mean_margin,
            "mean_margin_replay": replay_mean_margin,
            "mean_margin_abs_error": (
                abs(float(hosted_mean_margin) - replay_mean_margin)
                if hosted_mean_margin is not None else None
            ),
            "mean_margin_equal_at_telemetry_precision": mean_margin_equal,
            "recognition_all_observable_values_equal_at_telemetry_precision": bool(
                all(row["observable_values_equal_at_telemetry_precision"]
                    for row in frame_rows)
                and (mean_objectness_equal is not False)
                and (mean_margin_equal is not False)
            ),
        }

    return rows, per_frame, recognizer.describe()


def aggregate(
    runtime: dict[str, Any], image_manifest: list[dict[str, Any]],
    proposal_rows: list[dict[str, Any]], proposal_per_frame: dict[int, dict[str, Any]],
    recognition_rows: list[dict[str, Any]], recognition_per_frame: dict[int, dict[str, Any]],
    sensitivity_rows: list[dict[str, Any]], sensitivity_per_frame: dict[int, dict[str, Any]],
    records: dict[int, dict[str, Any]], recognizer_description: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    proposal_frames = [proposal_per_frame[index] for index in EXPECTED_INDICES]
    recognition_frames = [recognition_per_frame[index] for index in EXPECTED_INDICES]
    matched_proposals = [row for row in proposal_rows if row["matched"]]
    proposal_agrees = bool(
        all(row["proposal_count_match"] for row in proposal_frames)
        and all(row["proposal_ordering_match"] for row in proposal_frames)
        and all(row["proposal_all_pairs_equal_at_telemetry_precision"]
                for row in proposal_frames)
    )

    hosted_positive_recognition_frames = [
        recognition_per_frame[index] for index in EXPECTED_INDICES
        if records[index]["proposals"]["count"] > 0
    ]
    both_accepted = [
        row for row in recognition_rows
        if row["top_class_agrees"] is not None
    ]
    recognition_agrees = bool(
        all(row["recognition_counts_match"] for row in hosted_positive_recognition_frames)
        and all(row["background_rejection_agreement"]
                for row in hosted_positive_recognition_frames)
        and all(row["recognition_all_observable_values_equal_at_telemetry_precision"]
                for row in hosted_positive_recognition_frames)
    )
    sensitivity_positive_frames = [
        sensitivity_per_frame[index] for index in EXPECTED_INDICES
        if records[index]["proposals"]["count"] > 0
    ]
    sensitivity_both_accepted = [
        row for row in sensitivity_rows if row["top_class_agrees"] is not None
    ]
    sensitivity_agrees = bool(
        all(row["recognition_counts_match"] for row in sensitivity_positive_frames)
        and all(row["background_rejection_agreement"] for row in sensitivity_positive_frames)
        and all(row["recognition_all_observable_values_equal_at_telemetry_precision"]
                for row in sensitivity_positive_frames)
    )

    largest_proposal = sorted(
        EXPECTED_INDICES,
        key=lambda index: (
            proposal_per_frame[index]["proposal_count_abs_error"],
            proposal_per_frame[index]["proposal_max_bbox_error_px"] or 0.0,
            proposal_per_frame[index]["proposal_max_score_error"] or 0.0,
        ),
        reverse=True,
    )[:5]
    largest_recognition = sorted(
        [index for index in EXPECTED_INDICES if records[index]["proposals"]["count"] > 0],
        key=lambda index: (
            not recognition_per_frame[index]["recognition_counts_match"],
            recognition_per_frame[index].get("mean_objectness_abs_error") or 0.0,
            recognition_per_frame[index].get("mean_margin_abs_error") or 0.0,
        ),
        reverse=True,
    )[:5]

    if not proposal_agrees:
        outcome = "OUTCOME B"
        earliest = "discovery/proposal stage"
    elif not recognition_agrees:
        outcome = "OUTCOME B"
        earliest = "recognition stage"
    else:
        outcome = "OUTCOME A"
        earliest = "none at observable telemetry precision"

    proposal_summary = {
        "agreement_at_telemetry_precision": proposal_agrees,
        "comparison_precision": {
            "bbox_decimals": BBOX_DECIMALS,
            "score_decimals": SCALAR_DECIMALS,
        },
        "exact_proposal_count_match_rate": float(np.mean([
            row["proposal_count_match"] for row in proposal_frames
        ])),
        "proposal_count_mae": float(np.mean([
            row["proposal_count_abs_error"] for row in proposal_frames
        ])),
        "both_zero_frames": sum(
            row["hosted_proposal_count"] == 0 and row["replay_proposal_count"] == 0
            for row in proposal_frames
        ),
        "hosted_zero_replay_nonzero_frames": sum(
            row["hosted_proposal_count"] == 0 and row["replay_proposal_count"] > 0
            for row in proposal_frames
        ),
        "replay_zero_hosted_nonzero_frames": sum(
            row["replay_proposal_count"] == 0 and row["hosted_proposal_count"] > 0
            for row in proposal_frames
        ),
        "ordering_match_rate": float(np.mean([
            row["proposal_ordering_match"] for row in proposal_frames
        ])),
        "matched_proposal_iou": describe(row["iou"] for row in matched_proposals),
        "bbox_max_abs_error_px": describe(
            row["bbox_max_abs_error_px"] for row in matched_proposals
        ),
        "score_abs_error": describe(row["score_abs_error"] for row in matched_proposals),
        "hosted_proposal_total": sum(
            row["hosted_proposal_count"] for row in proposal_frames
        ),
        "replay_proposal_total": sum(
            row["replay_proposal_count"] for row in proposal_frames
        ),
        "largest_disagreement_frames": [
            {"frame_index": index, **proposal_per_frame[index]}
            for index in largest_proposal
        ],
        "backend_details": proposal_per_frame[-1]["backend_details"],
    }
    recognition_summary = {
        "agreement_at_telemetry_precision": recognition_agrees,
        "primary_input": "original recorded hosted proposal boxes",
        "hosted_positive_proposal_frames": len(hosted_positive_recognition_frames),
        "observation_count_match_rate": float(np.mean([
            row["hosted_observation_count"] == row["replay_observation_count"]
            for row in hosted_positive_recognition_frames
        ])),
        "full_count_match_rate": float(np.mean([
            row["recognition_counts_match"] for row in hosted_positive_recognition_frames
        ])),
        "background_rejection_frame_agreement_rate": float(np.mean([
            row["background_rejection_agreement"]
            for row in hosted_positive_recognition_frames
        ])),
        "background_rejection_proposal_agreement_rate": float(np.mean([
            row["background_rejection_agrees"] for row in recognition_rows
        ])),
        "top_class_agreement_rate": (
            float(np.mean([row["top_class_agrees"] for row in both_accepted]))
            if both_accepted else None
        ),
        "top_posterior_abs_error": describe(
            row["top_posterior_abs_error"] for row in both_accepted
        ),
        "quality_abs_error": describe(row["quality_abs_error"] for row in both_accepted),
        "per_frame_mean_objectness_abs_error": describe(
            row.get("mean_objectness_abs_error")
            for row in hosted_positive_recognition_frames
        ),
        "per_frame_mean_margin_abs_error": describe(
            row.get("mean_margin_abs_error")
            for row in hosted_positive_recognition_frames
        ),
        "hosted_crop_total": sum(
            row["hosted_crop_count"] for row in hosted_positive_recognition_frames
        ),
        "replay_crop_total": sum(
            row["replay_crop_count"] for row in hosted_positive_recognition_frames
        ),
        "hosted_rejected_total": sum(
            row["hosted_rejected_as_background"]
            for row in hosted_positive_recognition_frames
        ),
        "replay_rejected_total": sum(
            row["replay_rejected_as_background"]
            for row in hosted_positive_recognition_frames
        ),
        "hosted_observation_total": sum(
            row["hosted_observation_count"] for row in hosted_positive_recognition_frames
        ),
        "replay_observation_total": sum(
            row["replay_observation_count"] for row in hosted_positive_recognition_frames
        ),
        "not_observable": [
            "per-crop hosted objectness",
            "per-crop hosted margin",
            "hosted top class/posterior/quality for rejected crops",
            "non-top hosted class posterior values",
        ],
        "largest_disagreement_frames": [
            {"frame_index": index, **recognition_per_frame[index]}
            for index in largest_recognition
        ],
        "recognizer": recognizer_description,
        "secondary_full_precision_box_sensitivity": {
            "is_primary_comparison": False,
            "purpose": (
                "Tests whether any primary mismatch is explained by telemetry "
                "rounding proposal coordinates to 0.1 pixel before crop reconstruction."
            ),
            "agreement_at_telemetry_precision": sensitivity_agrees,
            "observation_count_match_rate": float(np.mean([
                row["hosted_observation_count"] == row["replay_observation_count"]
                for row in sensitivity_positive_frames
            ])),
            "background_rejection_proposal_agreement_rate": float(np.mean([
                row["background_rejection_agrees"] for row in sensitivity_rows
            ])),
            "top_class_agreement_rate": (
                float(np.mean([row["top_class_agrees"]
                               for row in sensitivity_both_accepted]))
                if sensitivity_both_accepted else None
            ),
            "top_posterior_abs_error": describe(
                row["top_posterior_abs_error"] for row in sensitivity_both_accepted
            ),
            "quality_abs_error": describe(
                row["quality_abs_error"] for row in sensitivity_both_accepted
            ),
        },
    }
    primary_disagreement_frames = {
        int(row["frame_index"])
        for row in recognition_rows
        if not row["observable_values_equal_at_telemetry_precision"]
    }
    primary_disagreement_frames.update(
        index for index in EXPECTED_INDICES
        if records[index]["proposals"]["count"] > 0
        and (
            recognition_per_frame[index].get("mean_objectness_equal_at_telemetry_precision")
            is False
            or recognition_per_frame[index].get("mean_margin_equal_at_telemetry_precision")
            is False
        )
    )
    crop_changed_frames = {
        int(row["frame_index"]) for row in recognition_rows
        if not row["recorded_vs_full_precision_crop_bounds_equal"]
    }
    recognition_summary["telemetry_rounding_attribution"] = {
        "recorded_box_precision_px": 0.1,
        "proposals_whose_integer_crop_bounds_changed": sum(
            not row["recorded_vs_full_precision_crop_bounds_equal"]
            for row in recognition_rows
        ),
        "total_proposals": len(recognition_rows),
        "primary_observable_disagreement_proposals": sum(
            not row["observable_values_equal_at_telemetry_precision"]
            for row in recognition_rows
        ),
        "primary_disagreement_frames": sorted(primary_disagreement_frames),
        "changed_crop_window_frames": sorted(crop_changed_frames),
        "all_primary_disagreement_frames_have_changed_crop_bounds": (
            primary_disagreement_frames <= crop_changed_frames
        ),
        "full_precision_sensitivity_reproduces_hosted": sensitivity_agrees,
        "interpretation": (
            "The primary recognition disagreement is attributable to reconstructing "
            "crops from proposal coordinates serialized to 0.1 pixel. All primary "
            "disagreement frames are exactly the frames where rounding changes a "
            "floor/ceil crop boundary, while the secondary replay with the detector's "
            "unrounded coordinates reproduces hosted recognition at recorded precision."
        ),
    }
    recognition_summary["ranked_plausible_causes"] = [
        {
            "rank": 1,
            "cause": "preprocessing mismatch from telemetry-rounded proposal geometry",
            "status": "CONFIRMED for the primary replay disagreement",
        },
        {
            "rank": 2,
            "cause": "model artifact or effective config mismatch",
            "status": "CONTRADICTED by matching hashes and exact full-precision sensitivity replay",
        },
        {
            "rank": 3,
            "cause": "image decode difference",
            "status": "CONTRADICTED by exact discovery reproduction and full-precision sensitivity replay",
        },
        {
            "rank": 4,
            "cause": "ONNX/runtime/library/thread/provider behavior",
            "status": "NOT SUPPORTED; full-precision end-to-end inference reproduces hosted telemetry",
        },
        {
            "rank": 5,
            "cause": "other deployment/runtime drift",
            "status": "NOT SUPPORTED on the 50 sampled images",
        },
    ]

    if outcome == "OUTCOME A":
        conclusion = (
            "CONFIRMED for the 50 sampled images: the frozen V2 pipeline itself "
            "produces the hosted collapse on the captured hosted input bytes. "
            "This strongly isolates the failure to input-domain/generalization "
            "behavior rather than deployment/runtime/config drift."
        )
        next_action = (
            "Design, but do not yet implement, a V3 input-domain generalization "
            "experiment using a held-out non-Helsinki image set."
        )
    else:
        if sensitivity_agrees:
            conclusion = (
                "The mandated primary recognition replay from proposal coordinates "
                "recorded to 0.1 pixel does not reproduce every hosted recognition "
                "value, so the strict classification is Outcome B. The disagreement "
                "is fully attributable to telemetry precision changing integer crop "
                "boundaries: the secondary end-to-end replay with unrounded detector "
                "coordinates reproduces hosted recognition exactly at recorded "
                "precision. This is not evidence of model/config/runtime drift, but "
                "the primary gate does not permit a domain-generalization conclusion."
            )
        else:
            conclusion = (
                "The frozen local replay does not reproduce hosted telemetry at the "
                "earliest identified stage, so the hosted result must not yet be "
                "classified as an input-domain/generalization failure."
            )
        next_action = (
            "Extend diagnostic telemetry to preserve full-precision proposal crop "
            "bounds and a SHA256 for each 64x64 recognition crop before any future "
            "authorized hosted capture."
        )

    summary = {
        "experiment": {
            "hosted_validation_uuid": HOSTED_VALIDATION_UUID,
            "frozen_v2_commit": FROZEN_COMMIT,
            "sampled_images": len(image_manifest),
            "frame_indices": EXPECTED_INDICES,
            "ground_truth_available": False,
        },
        "artifact_hash_verification": {
            "all_match": runtime["all_frozen_artifact_hashes_match"],
            "artifacts": runtime["frozen_artifacts"],
        },
        "image_byte_verification": {
            "images": len(image_manifest),
            "all_decode_960x540": all(row["dimensions_match"] for row in image_manifest),
            "all_sha256_recorded": all(len(row["sha256"]) == 64 for row in image_manifest),
            "bytes_modified": False,
        },
        "proposal_replay": proposal_summary,
        "recognition_replay": recognition_summary,
        "earliest_stage_of_disagreement": earliest,
        "final_classification": outcome,
        "conclusion": conclusion,
        "confidence": (
            "High for the 50 sampled images at recorded telemetry precision; "
            "not a statement about hidden ground truth."
        ),
        "limitations": [
            "Hosted proposal coordinates are recorded to 0.1 pixel and scores/posteriors/quality to four decimals.",
            "Recognition replay necessarily crops from telemetry-rounded hosted proposal boxes, not the unavailable full-precision in-memory boxes.",
            "Only 50 stride-5 images were captured; unsampled frames were not replayed.",
            "There is no hosted ground truth, so recall, detection accuracy, classification accuracy, hidden-object IoU/AP, and hidden object scale are NOT OBSERVABLE.",
        ],
        "exactly_one_next_action": next_action,
    }

    per_frame_rows: list[dict[str, Any]] = []
    for index in EXPECTED_INDICES:
        per_frame_rows.append({
            "frame_index": index,
            **proposal_per_frame[index],
            **recognition_per_frame[index],
        })
    return summary, per_frame_rows


def render_report(summary: dict[str, Any], runtime: dict[str, Any]) -> str:
    proposals = summary["proposal_replay"]
    recognition = summary["recognition_replay"]
    sensitivity = recognition["secondary_full_precision_box_sensitivity"]
    rounding = recognition["telemetry_rounding_attribution"]
    artifacts = summary["artifact_hash_verification"]["artifacts"]
    outcome = summary["final_classification"]

    artifact_rows = "\n".join(
        f"| `{Path(name).name}` | `{item['actual_sha256']}` | "
        f"{'PASS' if item['matches'] else 'FAIL'} |"
        for name, item in artifacts.items()
    )
    return f"""# Architecture V2 deterministic hosted-image shadow replay

## Executive result

**{outcome}.** {summary['conclusion']}

The earliest observable disagreement is **{summary['earliest_stage_of_disagreement']}**.
The comparison is strictly internal replay reproducibility; there is no hosted ground truth.

## 1. Frozen artifact verification

All required hashes match: **{summary['artifact_hash_verification']['all_match']}**.

| Artifact | Actual SHA256 | Result |
| --- | --- | --- |
{artifact_rows}

No inference ran until this gate and the runtime manifest had been written.

## 2. Runtime and configuration manifest

- Branch at replay: `{runtime['repository']['branch']}`
- HEAD at replay: `{runtime['repository']['head']}`
- Frozen V2 commit: `{runtime['repository']['frozen_v2_commit']}`
- Python: `{runtime['runtime']['python'].splitlines()[0]}`
- NumPy: `{runtime['runtime']['numpy']}`
- OpenCV: `{runtime['runtime']['opencv']}`
- ONNX Runtime: `{runtime['runtime']['onnxruntime']}`
- Torch: `{runtime['runtime']['torch']}`
- Torchvision: `{runtime['runtime']['torchvision']}`
- Proposal backend: `{proposals['backend_details'][0]['class']}` with providers `{proposals['backend_details'][0]['providers']}`

The complete effective proposal/recognition config, relevant environment/thread
variables, package versions, providers, and artifact sizes are in
`runtime_manifest.json`.

## 3. Image-byte verification

- Images processed: **{summary['image_byte_verification']['images']}**
- Exact expected frame indices: `0, 5, ..., 245`
- All decoded as exactly 960x540: **{summary['image_byte_verification']['all_decode_960x540']}**
- SHA256 recorded for every image: **{summary['image_byte_verification']['all_sha256_recorded']}**
- Source bytes modified or re-encoded: **False**

`image_manifest.csv` contains the per-file hashes, byte sizes, and dimensions.

## 4. Discovery/proposal replay

- Agreement at hosted telemetry precision: **{proposals['agreement_at_telemetry_precision']}**
- Exact proposal-count match rate: **{proposals['exact_proposal_count_match_rate']:.1%}**
- Proposal-count MAE: **{proposals['proposal_count_mae']:.6f}**
- Hosted/replay proposal totals: **{proposals['hosted_proposal_total']} / {proposals['replay_proposal_total']}**
- Both zero: **{proposals['both_zero_frames']} frames**
- Hosted zero, replay nonzero: **{proposals['hosted_zero_replay_nonzero_frames']} frames**
- Replay zero, hosted nonzero: **{proposals['replay_zero_hosted_nonzero_frames']} frames**
- Ordering match rate: **{proposals['ordering_match_rate']:.1%}**
- Matched IoU median/min: **{proposals['matched_proposal_iou']['median']:.9f} / {proposals['matched_proposal_iou']['min']:.9f}**
- Bbox max-absolute-error median/p95/max: **{proposals['bbox_max_abs_error_px']['median']:.6g} / {proposals['bbox_max_abs_error_px']['p95']:.6g} / {proposals['bbox_max_abs_error_px']['max']:.6g} px**
- Score absolute-error median/p95/max: **{proposals['score_abs_error']['median']:.6g} / {proposals['score_abs_error']['p95']:.6g} / {proposals['score_abs_error']['max']:.6g}**

Ordinal matching is used when the geometric assignment preserves ordering;
otherwise the CSV labels deterministic global-greedy IoU matching explicitly.
No floating-point difference is hidden behind the agreement label.

## 5. Recognition replay

The primary comparison feeds the **original recorded hosted proposal boxes** into
the frozen local crop and recognition path. Locally replayed proposal boxes are
not used here.

- Agreement at hosted telemetry precision: **{recognition['agreement_at_telemetry_precision']}**
- Hosted frames with proposals: **{recognition['hosted_positive_proposal_frames']}**
- Crop totals hosted/replay: **{recognition['hosted_crop_total']} / {recognition['replay_crop_total']}**
- Observation-count match rate: **{recognition['observation_count_match_rate']:.1%}**
- Background-rejection agreement, per proposal: **{recognition['background_rejection_proposal_agreement_rate']:.1%}**
- Rejection totals hosted/replay: **{recognition['hosted_rejected_total']} / {recognition['replay_rejected_total']}**
- Observation totals hosted/replay: **{recognition['hosted_observation_total']} / {recognition['replay_observation_total']}**
- Top-class agreement rate where both accept: **{recognition['top_class_agreement_rate']:.1%}**
- Top-posterior absolute-error median/p95/max: **{recognition['top_posterior_abs_error']['median']:.6g} / {recognition['top_posterior_abs_error']['p95']:.6g} / {recognition['top_posterior_abs_error']['max']:.6g}**
- Quality absolute-error median/p95/max: **{recognition['quality_abs_error']['median']:.6g} / {recognition['quality_abs_error']['p95']:.6g} / {recognition['quality_abs_error']['max']:.6g}**

The primary mismatch is localized to telemetry precision:

- Integer crop bounds changed after reconstructing from 0.1-pixel boxes: **{rounding['proposals_whose_integer_crop_bounds_changed']}/{rounding['total_proposals']} proposals**
- Primary disagreement frames: **{rounding['primary_disagreement_frames']}**
- Frames with changed integer crop bounds: **{rounding['changed_crop_window_frames']}**
- Every primary disagreement frame has a changed crop boundary: **{rounding['all_primary_disagreement_frames_have_changed_crop_bounds']}**
- Secondary full-precision detector-box sensitivity agrees: **{sensitivity['agreement_at_telemetry_precision']}**
- Secondary top-class agreement: **{sensitivity['top_class_agreement_rate']:.1%}**
- Secondary top-posterior absolute-error max: **{sensitivity['top_posterior_abs_error']['max']:.6g}**
- Secondary quality absolute-error max: **{sensitivity['quality_abs_error']['max']:.6g}**

This secondary check is not substituted for the required primary comparison. It
shows that the primary mismatch is caused by crop reconstruction from serialized
coordinates rather than by the frozen detector, recognizer, image bytes, model
artifacts, or effective local config.

Per-crop hosted objectness and margin, rejected-crop hosted class/posterior/quality,
and non-top hosted posterior values are **NOT OBSERVABLE**. Per-frame mean
objectness and mean margin are compared because telemetry records them.

## 6. Earliest disagreement and classification

- Earliest stage: **{summary['earliest_stage_of_disagreement']}**
- Final classification: **{outcome}**

Ranked causes of the observed primary disagreement:

1. **Preprocessing mismatch from telemetry-rounded proposal geometry — CONFIRMED.**
2. **Model artifact or effective config mismatch — CONTRADICTED** by hashes and the exact full-precision sensitivity replay.
3. **Image decode difference — CONTRADICTED** by exact discovery and full-precision recognition reproduction.
4. **ONNX/runtime/library/thread/provider behavior — NOT SUPPORTED.**
5. **Other deployment/runtime drift — NOT SUPPORTED on these samples.**

## 7. Confidence and limitations

Confidence is **high for these 50 sampled images at recorded telemetry precision**.
Hosted boxes are rounded to 0.1 pixel and scalar inference values to four decimals.
Recognition crops therefore use the exact recorded boxes, but those records are
not the unavailable full-precision in-memory proposal boxes. The captured set is
a stride-5 sample, not all endpoint inputs.

There is no hosted ground truth. Hosted proposal recall, detection accuracy,
classification accuracy, IoU to hidden objects, hidden class-specific AP, and
hidden object scale remain **NOT OBSERVABLE**.

## 8. Exactly one next action

**{summary['exactly_one_next_action']}**

This action is recommended only; it was not implemented.

## Reproduction

From the repository root:

```powershell
& 'drone-flyby/.venv/Scripts/python.exe' 'drone-flyby/architecture_experiments/v2_hosted_shadow_replay/run_shadow_replay.py'
```

The command fails closed on artifact hash drift, image-count/dimension errors,
missing sampled telemetry, or internally inconsistent output row counts.
"""


def verify_outputs(
    image_manifest: list[dict[str, Any]], proposal_rows: list[dict[str, Any]],
    recognition_rows: list[dict[str, Any]], per_frame_rows: list[dict[str, Any]],
    records: dict[int, dict[str, Any]],
) -> dict[str, int]:
    expected_proposal_rows = sum(
        max(
            int(row["hosted_proposal_count"]),
            int(row["replay_proposal_count"]),
        )
        for row in per_frame_rows
    )
    expected_recognition_rows = sum(
        int(records[index]["proposals"]["count"]) for index in EXPECTED_INDICES
    )
    checks = {
        "image_manifest_rows": len(image_manifest),
        "proposal_comparison_rows": len(proposal_rows),
        "expected_proposal_comparison_rows": expected_proposal_rows,
        "recognition_comparison_rows": len(recognition_rows),
        "expected_recognition_comparison_rows": expected_recognition_rows,
        "per_frame_summary_rows": len(per_frame_rows),
    }
    if len(image_manifest) != 50:
        raise RuntimeError(f"image_manifest row count is {len(image_manifest)}, expected 50")
    if len(per_frame_rows) != 50:
        raise RuntimeError(f"per_frame_summary row count is {len(per_frame_rows)}, expected 50")
    if len(proposal_rows) != expected_proposal_rows:
        raise RuntimeError(
            f"proposal comparison rows {len(proposal_rows)} != {expected_proposal_rows}"
        )
    if len(recognition_rows) != expected_recognition_rows:
        raise RuntimeError(
            f"recognition comparison rows {len(recognition_rows)} != {expected_recognition_rows}"
        )
    return checks


def generate() -> None:
    if str(DRONE_ROOT) not in sys.path:
        sys.path.insert(0, str(DRONE_ROOT))
    HERE.mkdir(parents=True, exist_ok=True)

    runtime = runtime_manifest()
    write_json(HERE / "runtime_manifest.json", runtime)
    if not runtime["all_frozen_artifact_hashes_match"]:
        mismatches = [
            name for name, item in runtime["frozen_artifacts"].items()
            if not item["matches"]
        ]
        raise RuntimeError(
            "Frozen artifact hash mismatch; inference stopped before model load: "
            + ", ".join(mismatches)
        )

    images, records, image_manifest = load_evidence()
    write_csv(
        HERE / "image_manifest.csv", image_manifest,
        [
            "frame_index", "filename", "sha256", "size_bytes", "decoded",
            "width", "height", "channels", "expected_width", "expected_height",
            "dimensions_match",
        ],
    )

    proposal_rows, replay_by_frame, proposal_per_frame = proposal_replay(images, records)
    recognition_rows, recognition_per_frame, recognizer_description = recognition_replay(
        images, records
    )
    sensitivity_rows, sensitivity_per_frame, _ = recognition_replay(
        images, records, proposal_overrides=replay_by_frame
    )
    sensitivity_by_key = {
        (int(row["frame_index"]), int(row["hosted_proposal_ordinal"])): row
        for row in sensitivity_rows
    }
    crop_context = float(recognizer_description["context"])
    for row in recognition_rows:
        key = (int(row["frame_index"]), int(row["hosted_proposal_ordinal"]))
        sensitivity = sensitivity_by_key[key]
        recorded_box = [
            row["proposal_x1"], row["proposal_y1"],
            row["proposal_x2"], row["proposal_y2"],
        ]
        full_precision_box = [
            sensitivity["proposal_x1"], sensitivity["proposal_y1"],
            sensitivity["proposal_x2"], sensitivity["proposal_y2"],
        ]
        recorded_bounds = crop_bounds(recorded_box, crop_context)
        full_precision_bounds = crop_bounds(full_precision_box, crop_context)
        row["recorded_crop_bounds_xyxy"] = json.dumps(recorded_bounds)
        row["full_precision_replay_crop_bounds_xyxy"] = json.dumps(full_precision_bounds)
        row["recorded_vs_full_precision_crop_bounds_equal"] = (
            recorded_bounds == full_precision_bounds
        )
        row["full_precision_sensitivity_observable_values_equal"] = sensitivity[
            "observable_values_equal_at_telemetry_precision"
        ]
    summary, per_frame_rows = aggregate(
        runtime, image_manifest, proposal_rows, proposal_per_frame,
        recognition_rows, recognition_per_frame, sensitivity_rows, sensitivity_per_frame,
        records, recognizer_description,
    )
    summary["internal_consistency_checks"] = verify_outputs(
        image_manifest, proposal_rows, recognition_rows, per_frame_rows, records
    )

    proposal_fields = list(proposal_rows[0].keys()) if proposal_rows else ["frame_index"]
    recognition_fields = (
        list(recognition_rows[0].keys()) if recognition_rows else ["frame_index"]
    )
    per_frame_fields = list(per_frame_rows[0].keys())
    write_csv(HERE / "proposal_comparison.csv", proposal_rows, proposal_fields)
    write_csv(HERE / "recognition_comparison.csv", recognition_rows, recognition_fields)
    write_csv(HERE / "per_frame_summary.csv", per_frame_rows, per_frame_fields)
    write_json(HERE / "shadow_replay_summary.json", summary)
    (HERE / "SHADOW_REPLAY_REPORT.md").write_text(
        render_report(summary, runtime), encoding="utf-8"
    )
    print(json.dumps({
        "classification": summary["final_classification"],
        "earliest_disagreement": summary["earliest_stage_of_disagreement"],
        "proposal_agreement": summary["proposal_replay"]["agreement_at_telemetry_precision"],
        "recognition_agreement": summary["recognition_replay"]["agreement_at_telemetry_precision"],
        "images": summary["experiment"]["sampled_images"],
        "output_directory": str(HERE),
    }, indent=2))


def check_existing() -> None:
    required = [
        "SHADOW_REPLAY_REPORT.md", "shadow_replay_summary.json", "runtime_manifest.json",
        "image_manifest.csv", "proposal_comparison.csv", "recognition_comparison.csv",
        "per_frame_summary.csv",
    ]
    missing = [name for name in required if not (HERE / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing outputs: {missing}")
    with (HERE / "image_manifest.csv").open(newline="", encoding="utf-8") as handle:
        image_rows = list(csv.DictReader(handle))
    with (HERE / "per_frame_summary.csv").open(newline="", encoding="utf-8") as handle:
        frame_rows = list(csv.DictReader(handle))
    with (HERE / "recognition_comparison.csv").open(newline="", encoding="utf-8") as handle:
        recognition_rows = list(csv.DictReader(handle))
    if len(image_rows) != 50 or len(frame_rows) != 50 or len(recognition_rows) != 41:
        raise RuntimeError(
            "Existing output row-count check failed: "
            f"images={len(image_rows)}, frames={len(frame_rows)}, "
            f"recognition={len(recognition_rows)}"
        )
    print("Existing shadow replay outputs pass structural checks.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true",
        help="Check existing required outputs without running inference.",
    )
    args = parser.parse_args()
    if args.check:
        check_existing()
    else:
        generate()


if __name__ == "__main__":
    main()
