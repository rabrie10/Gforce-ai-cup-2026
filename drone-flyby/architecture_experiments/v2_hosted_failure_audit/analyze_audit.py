"""Reproducible GT-free comparison of hosted V2 and matched Helsinki V2.

This script reads telemetry and images but never changes them. It writes only
derived CSV, JSON, and PNG audit artifacts beneath this directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


AUDIT_DIR = Path(__file__).resolve().parent
DRONE_ROOT = AUDIT_DIR.parents[1]
HOSTED_DIR = (
    DRONE_ROOT
    / "hosted_telemetry"
    / "v2_validation_6a86911a"
    / "run_20260919_005220"
)
LOCAL_TELEMETRY_DIR = AUDIT_DIR / "helsinki_reference_telemetry"
LOCAL_IMAGE_DIR = DRONE_ROOT / "src" / "helsinki" / "images"
CSV_DIR = AUDIT_DIR / "csv"
FIGURE_DIR = AUDIT_DIR / "figures"
METRICS_PATH = AUDIT_DIR / "computed_metrics.json"

CLASS_NAMES = (
    "hangar",
    "helicopter",
    "jet_plane",
    "large_launcher",
    "large_tower",
    "medium_launcher",
    "medium_plane",
    "mine_roller",
    "small_launcher",
    "small_plane",
    "small_tower",
    "ta-ta",
    "tank",
    "condor",
    "jammer",
    "spacecraft",
)
VIEW_WIDTH = 960
VIEW_HEIGHT = 540


def load_records(frame_dir: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for path in sorted(frame_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        index = int(payload["frame_index"])
        if index in records:
            raise ValueError(f"duplicate frame_index={index} in {frame_dir}")
        records[index] = payload
    return records


def finite(values: Iterable[Any]) -> np.ndarray:
    cleaned = []
    for value in values:
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            cleaned.append(number)
    return np.asarray(cleaned, dtype=np.float64)


def describe(values: Iterable[Any]) -> dict[str, Any]:
    data = finite(values)
    if not len(data):
        return {"n": 0, "mean": None, "median": None, "p05": None,
                "p95": None, "min": None, "max": None, "std": None}
    return {
        "n": int(len(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p05": float(np.percentile(data, 5)),
        "p95": float(np.percentile(data, 95)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "std": float(np.std(data)),
    }


def ratio(hosted: Any, local: Any) -> float | None:
    if hosted is None or local in (None, 0):
        return None
    return float(hosted) / float(local)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def flatten_requested(requested: Any) -> tuple[Any, Any, Any]:
    if not isinstance(requested, dict):
        return None, None, None
    return (
        requested.get("resolution_level", requested.get("level")),
        requested.get("center_x"),
        requested.get("center_y"),
    )


def extract_rows(dataset: str, records: dict[int, dict[str, Any]],
                 expected_indices: Iterable[int]) -> tuple[list[dict[str, Any]],
                                                            list[dict[str, Any]],
                                                            list[dict[str, Any]],
                                                            list[dict[str, Any]],
                                                            list[dict[str, Any]]]:
    frame_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for index in expected_indices:
        record = records.get(index)
        if record is None:
            frame_rows.append({"dataset": dataset, "frame_index": index,
                               "record_present": False})
            continue

        proposals = record.get("proposals", {})
        recognition = record.get("recognition", {})
        observations = record.get("observations", [])
        tracks = record.get("tracks", [])
        output = record.get("output", {})
        bank = record.get("state", {}).get("bank", {})
        gmc = record.get("gmc", {})
        view = record.get("view", {})
        decision = record.get("camera_decision", {})
        feedback = record.get("camera_command_feedback")
        request_level, request_x, request_y = flatten_requested(decision.get("requested"))
        translation = gmc.get("translation") or [None, None]
        tx = translation[0] if len(translation) > 0 else None
        ty = translation[1] if len(translation) > 1 else None
        tmag = None if tx is None or ty is None else math.hypot(float(tx), float(ty))
        crops = recognition.get("crops")
        rejected = recognition.get("rejected_as_background")
        rejection_fraction = (
            float(rejected) / float(crops)
            if crops not in (None, 0) and rejected is not None else None
        )
        confirmed = sum(bool(row.get("confirmed")) for row in tracks)
        ages = [row.get("age") for row in tracks]
        posterior_gaps = [
            float(row.get("identity_confidence", 0.0))
            - float(row.get("runner_up_confidence", 0.0))
            for row in tracks
        ]
        frame_rows.append({
            "dataset": dataset,
            "frame_index": index,
            "record_present": True,
            "sequence_id": record.get("sequence_id"),
            "request_frame": record.get("frame"),
            "resolution_level": view.get("resolution_level"),
            "center_x": view.get("center_x"),
            "center_y": view.get("center_y"),
            "camera_decision_reason": decision.get("reason"),
            "camera_requested_level": request_level,
            "camera_requested_center_x": request_x,
            "camera_requested_center_y": request_y,
            "camera_feedback_present": feedback is not None,
            "camera_feedback_reason": feedback.get("reason") if isinstance(feedback, dict) else None,
            "proposal_count": proposals.get("count", len(proposals.get("boxes", []))),
            "crops": crops,
            "rejected_as_background": rejected,
            "rejection_fraction": rejection_fraction,
            "mean_objectness": recognition.get("mean_objectness"),
            "mean_margin": recognition.get("mean_margin"),
            "observation_count": len(observations),
            "observation_top_posterior_mean": (
                float(np.mean([row["top_posterior"] for row in observations]))
                if observations else None
            ),
            "bank_tracks": bank.get("tracks", len(tracks)),
            "bank_matched": bank.get("matched"),
            "bank_created": bank.get("created"),
            "bank_removed": bank.get("removed"),
            "confirmed_tracks": confirmed,
            "tentative_tracks": len(tracks) - confirmed,
            "track_age_mean": float(np.mean(ages)) if ages else None,
            "track_age_max": max(ages) if ages else None,
            "track_identity_confidence_mean": (
                float(np.mean([row.get("identity_confidence", 0.0) for row in tracks]))
                if tracks else None
            ),
            "track_posterior_gap_mean": float(np.mean(posterior_gaps)) if posterior_gaps else None,
            "output_candidates": output.get("candidates"),
            "output_emitted": output.get("emitted"),
            "output_dropped_invalid": output.get("dropped_invalid"),
            "output_distinct_classes": output.get("distinct_classes"),
            "output_secondary_guesses": output.get("secondary_guesses"),
            "gmc_model": gmc.get("model"),
            "gmc_ok": gmc.get("ok"),
            "gmc_reason": gmc.get("reason"),
            "gmc_quality": gmc.get("quality"),
            "gmc_inliers": gmc.get("inliers"),
            "gmc_candidates": gmc.get("candidates"),
            "gmc_inlier_ratio": gmc.get("inlier_ratio"),
            "gmc_translation_x": tx,
            "gmc_translation_y": ty,
            "gmc_translation_magnitude": tmag,
            "total_latency_ms": record.get("timing_ms", {}).get("total"),
        })

        for ordinal, row in enumerate(proposals.get("boxes", [])):
            x1, y1, x2, y2 = row["box"]
            width, height = x2 - x1, y2 - y1
            proposal_rows.append({
                "dataset": dataset,
                "frame_index": index,
                "proposal_ordinal": ordinal,
                "source": row.get("source"),
                "score": row.get("score"),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width_px": width,
                "height_px": height,
                "area_px2": width * height,
                "short_side_px": min(width, height),
                "center_x_norm": ((x1 + x2) / 2.0) / VIEW_WIDTH,
                "center_y_norm": ((y1 + y2) / 2.0) / VIEW_HEIGHT,
            })

        per_frame_track_classes = Counter(row.get("object_id") for row in tracks)
        per_frame_observation_classes = Counter(
            CLASS_NAMES[int(row["top"])] for row in observations
            if 0 <= int(row["top"]) < len(CLASS_NAMES)
        )
        for class_name in CLASS_NAMES:
            class_rows.append({
                "dataset": dataset,
                "frame_index": index,
                "class_name": class_name,
                "track_primary_count": per_frame_track_classes[class_name],
                "observation_top_count": per_frame_observation_classes[class_name],
            })

        for ordinal, row in enumerate(observations):
            x1, y1, x2, y2 = row["box"]
            top_index = int(row["top"])
            observation_rows.append({
                "dataset": dataset,
                "frame_index": index,
                "observation_ordinal": ordinal,
                "source": row.get("source"),
                "top_index": top_index,
                "top_class": CLASS_NAMES[top_index] if 0 <= top_index < len(CLASS_NAMES) else None,
                "top_posterior": row.get("top_posterior"),
                "quality": row.get("quality"),
                "level": row.get("level"),
                "x1_source": x1, "y1_source": y1, "x2_source": x2, "y2_source": y2,
                "width_px_source": x2 - x1,
                "height_px_source": y2 - y1,
                "short_side_px_source": min(x2 - x1, y2 - y1),
            })

        for row in tracks:
            x1, y1, x2, y2 = row["box"]
            track_rows.append({
                "dataset": dataset,
                "frame_index": index,
                "track_id": row.get("track_id"),
                "object_id": row.get("object_id"),
                "identity_confidence": row.get("identity_confidence"),
                "runner_up": row.get("runner_up"),
                "runner_up_confidence": row.get("runner_up_confidence"),
                "posterior_gap": (
                    float(row.get("identity_confidence", 0.0))
                    - float(row.get("runner_up_confidence", 0.0))
                ),
                "hits": row.get("hits"),
                "age": row.get("age"),
                "confirmed": row.get("confirmed"),
                "last_observed_frame": row.get("last_observed_frame"),
                "last_observed_level": row.get("last_observed_level"),
                "geometry_confidence": row.get("geometry_confidence"),
                "propagation_model": row.get("propagation_model"),
                "short_side_px_source": row.get("short_side"),
                "width_px_source": x2 - x1,
                "height_px_source": y2 - y1,
            })
    return frame_rows, proposal_rows, track_rows, observation_rows, class_rows


def image_metrics(dataset: str, path: Path, frame_index: int,
                  downsample_helsinki: bool) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to read {path}")
    original_height, original_width = image.shape[:2]
    if downsample_helsinki:
        image = cv2.resize(image, (VIEW_WIDTH, VIEW_HEIGHT), interpolation=cv2.INTER_AREA)
    if image.shape[:2] != (VIEW_HEIGHT, VIEW_WIDTH):
        raise ValueError(f"unexpected comparison image size {image.shape[:2]} for {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    edges = cv2.Canny(gray, 50, 150)
    components, _, stats, _ = cv2.connectedComponentsWithStats(edges, 8)
    component_areas = stats[1:, cv2.CC_STAT_AREA] if components > 1 else np.asarray([])
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probability = histogram[histogram > 0] / histogram.sum()
    entropy = float(-(probability * np.log2(probability)).sum())
    b, g, r = cv2.split(image.astype(np.float32))
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = math.sqrt(float(np.var(rg) + np.var(yb))) + 0.3 * math.sqrt(
        float(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )
    orb = cv2.ORB_create(nfeatures=10000)
    keypoints = orb.detect(gray, None)
    return {
        "dataset": dataset,
        "frame_index": frame_index,
        "source_file": path.name,
        "source_width": original_width,
        "source_height": original_height,
        "analysis_width": image.shape[1],
        "analysis_height": image.shape[0],
        "brightness_mean": float(np.mean(gray)),
        "contrast_std": float(np.std(gray)),
        "luma_p05": float(np.percentile(gray, 5)),
        "luma_p95": float(np.percentile(gray, 95)),
        "dynamic_range_p95_p05": float(np.percentile(gray, 95) - np.percentile(gray, 5)),
        "entropy_bits": entropy,
        "saturation_mean": float(np.mean(hsv[:, :, 1])),
        "colorfulness": colorfulness,
        "gradient_mean": float(np.mean(gradient)),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "edge_density": float(np.count_nonzero(edges) / edges.size),
        "edge_component_count": int(max(0, components - 1)),
        "edge_component_area_median": (
            float(np.median(component_areas)) if len(component_areas) else 0.0
        ),
        "orb_keypoints": len(keypoints),
        "dark_fraction": float(np.mean(gray < 40)),
        "bright_fraction": float(np.mean(gray > 215)),
    }


def summarize_dataset(frame_rows: list[dict[str, Any]],
                      proposal_rows: list[dict[str, Any]],
                      track_rows: list[dict[str, Any]],
                      observation_rows: list[dict[str, Any]],
                      image_rows: list[dict[str, Any]]) -> dict[str, Any]:
    present = [row for row in frame_rows if row.get("record_present")]
    gmc_ok = [row for row in present if row.get("gmc_ok")]
    metric_names = (
        "proposal_count", "crops", "rejected_as_background", "rejection_fraction",
        "mean_objectness", "mean_margin", "observation_count",
        "observation_top_posterior_mean", "bank_tracks", "bank_matched",
        "bank_created", "bank_removed", "confirmed_tracks", "tentative_tracks",
        "track_age_mean", "track_age_max", "track_identity_confidence_mean",
        "track_posterior_gap_mean", "output_candidates", "output_emitted",
        "output_distinct_classes", "output_secondary_guesses",
        "gmc_quality", "gmc_inliers", "gmc_inlier_ratio",
        "gmc_translation_magnitude", "total_latency_ms",
    )
    track_groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in track_rows:
        track_groups[row.get("track_id")].append(row)
    lifecycles = []
    for track_id, rows in track_groups.items():
        indices = [int(row["frame_index"]) for row in rows]
        lifecycles.append({
            "track_id": track_id,
            "span_frames": max(indices) - min(indices) + 1,
            "telemetry_appearances": len(indices),
            "max_hits": max(int(row.get("hits") or 0) for row in rows),
            "ever_confirmed": any(bool(row.get("confirmed")) for row in rows),
        })
    proposal_scores = finite(row.get("score") for row in proposal_rows)
    temporal_blocks = []
    if present:
        last_index = max(int(row["frame_index"]) for row in present)
        for start in range(0, last_index + 1, 25):
            rows = [row for row in present if start <= int(row["frame_index"]) < start + 25]
            if not rows:
                continue
            temporal_blocks.append({
                "frame_start": start,
                "frame_end": start + 24,
                "records": len(rows),
                "proposal_count_mean": describe(row.get("proposal_count") for row in rows)["mean"],
                "zero_proposal_frames": sum(row.get("proposal_count") == 0 for row in rows),
                "observation_count_mean": describe(row.get("observation_count") for row in rows)["mean"],
                "bank_tracks_mean": describe(row.get("bank_tracks") for row in rows)["mean"],
                "output_emitted_mean": describe(row.get("output_emitted") for row in rows)["mean"],
                "zero_output_frames": sum(row.get("output_emitted") == 0 for row in rows),
            })
    summary = {
        "expected_frames": len(frame_rows),
        "records_present": len(present),
        "missing_frame_indices": [row["frame_index"] for row in frame_rows
                                  if not row.get("record_present")],
        "zero_proposal_frames": sum(row.get("proposal_count") == 0 for row in present),
        "zero_output_frames": sum(row.get("output_emitted") == 0 for row in present),
        "gmc_accepted_frames": len(gmc_ok),
        "gmc_accepted_rate": len(gmc_ok) / len(present) if present else None,
        "gmc_reason_counts": dict(Counter(row.get("gmc_reason") for row in present)),
        "gmc_model_counts": dict(Counter(row.get("gmc_model") for row in present)),
        "resolution_level_counts": dict(Counter(str(row.get("resolution_level")) for row in present)),
        "camera_decision_reason_counts": dict(Counter(row.get("camera_decision_reason") for row in present)),
        "camera_requested_frames": sum(row.get("camera_requested_level") is not None for row in present),
        "camera_feedback_frames": sum(bool(row.get("camera_feedback_present")) for row in present),
        "whole_run_track_primary_classes": dict(Counter(row.get("object_id") for row in track_rows)),
        "whole_run_observation_top_classes": {},
        "metrics": {name: describe(row.get(name) for row in present) for name in metric_names},
        "proposal_metrics": {
            name: describe(row.get(name) for row in proposal_rows)
            for name in ("score", "width_px", "height_px", "area_px2", "short_side_px",
                         "center_x_norm", "center_y_norm")
        },
        "track_metrics": {
            name: describe(row.get(name) for row in track_rows)
            for name in ("identity_confidence", "runner_up_confidence", "posterior_gap",
                         "hits", "age", "geometry_confidence", "short_side_px_source")
        },
        "observation_metrics": {
            name: describe(row.get(name) for row in observation_rows)
            for name in ("top_posterior", "quality", "short_side_px_source")
        },
        "track_confirmed_fraction": (
            sum(bool(row.get("confirmed")) for row in track_rows) / len(track_rows)
            if track_rows else None
        ),
        "track_propagation_model_counts": dict(Counter(row.get("propagation_model") for row in track_rows)),
        "proposal_score_threshold_fractions": {
            "gte_0_1": float(np.mean(proposal_scores >= 0.1)) if len(proposal_scores) else None,
            "gte_0_5": float(np.mean(proposal_scores >= 0.5)) if len(proposal_scores) else None,
            "gte_0_9": float(np.mean(proposal_scores >= 0.9)) if len(proposal_scores) else None,
        },
        "track_lifecycle": {
            "unique_track_ids": len(lifecycles),
            "created_total": int(sum(int(row.get("bank_created") or 0) for row in present)),
            "matched_total": int(sum(int(row.get("bank_matched") or 0) for row in present)),
            "removed_total": int(sum(int(row.get("bank_removed") or 0) for row in present)),
            "span_frames": describe(row["span_frames"] for row in lifecycles),
            "ever_confirmed_fraction": (
                sum(row["ever_confirmed"] for row in lifecycles) / len(lifecycles)
                if lifecycles else None
            ),
            "max_hits_gte_5_fraction": (
                sum(row["max_hits"] >= 5 for row in lifecycles) / len(lifecycles)
                if lifecycles else None
            ),
        },
        "temporal_blocks_25_frames": temporal_blocks,
        "image_metrics": {
            name: describe(row.get(name) for row in image_rows)
            for name in (
                "brightness_mean", "contrast_std", "dynamic_range_p95_p05", "entropy_bits",
                "saturation_mean", "colorfulness", "gradient_mean", "laplacian_variance",
                "edge_density", "edge_component_count", "edge_component_area_median",
                "orb_keypoints", "dark_fraction", "bright_fraction",
            )
        },
    }
    return summary


def add_observation_class_counts(summary: dict[str, Any], class_rows: list[dict[str, Any]]) -> None:
    counts = defaultdict(int)
    for row in class_rows:
        counts[row["class_name"]] += int(row["observation_top_count"])
    summary["whole_run_observation_top_classes"] = dict(counts)


def plot_time_series(frame_rows: list[dict[str, Any]]) -> None:
    fields = [
        ("proposal_count", "Proposals"),
        ("observation_count", "Observations"),
        ("bank_tracks", "Track bank size"),
        ("output_emitted", "Emitted annotations"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    for axis, (field, label) in zip(axes, fields):
        for dataset, color in (("hosted", "#b23a48"), ("helsinki", "#246a73")):
            rows = [row for row in frame_rows if row["dataset"] == dataset and row.get("record_present")]
            axis.plot([row["frame_index"] for row in rows], [row.get(field) for row in rows],
                      label=dataset, color=color, linewidth=1.4)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Frame index (runs have different lengths)")
    fig.suptitle("Per-frame pipeline volume")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pipeline_volume_over_time.png", dpi=180)
    plt.close(fig)


def plot_distributions(proposal_rows: list[dict[str, Any]], track_rows: list[dict[str, Any]]) -> None:
    specs = [
        (proposal_rows, "score", "Proposal score", (0, 1)),
        (proposal_rows, "short_side_px", "Proposal short side (received px)", None),
        (proposal_rows, "area_px2", "Proposal area (received px²)", None),
        (track_rows, "identity_confidence", "Track primary posterior", (0, 1)),
        (track_rows, "posterior_gap", "Track posterior gap", (0, 1)),
        (track_rows, "age", "Track age (frames)", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, (rows, field, label, limits) in zip(axes.ravel(), specs):
        for dataset, color in (("hosted", "#b23a48"), ("helsinki", "#246a73")):
            values = finite(row.get(field) for row in rows if row["dataset"] == dataset)
            if not len(values):
                continue
            upper = np.percentile(values, 99) if limits is None else limits[1]
            lower = np.min(values) if limits is None else limits[0]
            bins = np.linspace(lower, upper if upper > lower else lower + 1, 31)
            axis.hist(np.clip(values, bins[0], bins[-1]), bins=bins, density=True,
                      histtype="step", linewidth=1.8, label=dataset, color=color)
        axis.set_title(label)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Discovery and track-state distributions")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "discovery_track_distributions.png", dpi=180)
    plt.close(fig)


def plot_spatial(proposal_rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for axis, dataset in zip(axes, ("hosted", "helsinki")):
        rows = [row for row in proposal_rows if row["dataset"] == dataset]
        hist = axis.hist2d(
            [row["center_x_norm"] for row in rows],
            [row["center_y_norm"] for row in rows],
            bins=(12, 7), range=((0, 1), (0, 1)), cmap="magma", density=True,
        )
        axis.set_title(dataset)
        axis.set_xlabel("Normalized x")
        axis.invert_yaxis()
        fig.colorbar(hist[3], ax=axis, fraction=0.046, label="Density")
    axes[0].set_ylabel("Normalized y")
    fig.suptitle("Proposal-center spatial distribution")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "proposal_spatial_distribution.png", dpi=180)
    plt.close(fig)


def plot_gmc_output(frame_rows: list[dict[str, Any]]) -> None:
    specs = [
        ("gmc_quality", "GMC quality"),
        ("gmc_inlier_ratio", "GMC inlier ratio"),
        ("gmc_translation_magnitude", "GMC translation magnitude (px)"),
        ("output_distinct_classes", "Output distinct classes/frame"),
        ("output_secondary_guesses", "Secondary guesses/frame"),
        ("rejection_fraction", "Recognition background rejection fraction"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, (field, label) in zip(axes.ravel(), specs):
        data = []
        labels = []
        for dataset in ("hosted", "helsinki"):
            values = finite(row.get(field) for row in frame_rows
                            if row["dataset"] == dataset and row.get("record_present"))
            data.append(values)
            labels.append(dataset)
        axis.boxplot(data, tick_labels=labels, showfliers=False)
        axis.set_title(label)
        axis.grid(alpha=0.2, axis="y")
    fig.suptitle("GMC, recognition, and output summaries")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "gmc_recognition_output.png", dpi=180)
    plt.close(fig)


def plot_images(image_rows: list[dict[str, Any]]) -> None:
    specs = [
        ("brightness_mean", "Brightness (mean luma)"),
        ("contrast_std", "Contrast (luma std)"),
        ("entropy_bits", "Luma entropy (bits)"),
        ("gradient_mean", "Mean gradient magnitude"),
        ("edge_density", "Canny edge density"),
        ("orb_keypoints", "ORB keypoints"),
        ("laplacian_variance", "Laplacian variance"),
        ("saturation_mean", "Mean saturation"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, (field, label) in zip(axes.ravel(), specs):
        data = [
            finite(row.get(field) for row in image_rows if row["dataset"] == dataset)
            for dataset in ("hosted", "helsinki")
        ]
        axis.boxplot(data, tick_labels=["hosted", "helsinki"], showfliers=True)
        axis.set_title(label)
        axis.grid(alpha=0.2, axis="y")
    fig.suptitle("GT-free image characteristics at the common 960×540 input size")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "image_characteristics.png", dpi=180)
    plt.close(fig)


def plot_contact_sheet() -> None:
    hosted_paths = sorted((HOSTED_DIR / "images").glob("*.png"))
    local_paths = sorted(LOCAL_IMAGE_DIR.glob("frame_*.png"))
    hosted_selected = [hosted_paths[index] for index in np.linspace(0, len(hosted_paths) - 1, 6, dtype=int)]
    local_selected = [local_paths[index] for index in np.linspace(0, len(local_paths) - 1, 6, dtype=int)]
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.2))
    for row_index, (label, paths) in enumerate((("hosted", hosted_selected), ("helsinki", local_selected))):
        for column_index, path in enumerate(paths):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if label == "helsinki":
                image = cv2.resize(image, (VIEW_WIDTH, VIEW_HEIGHT), interpolation=cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axis = axes[row_index, column_index]
            axis.imshow(image)
            axis.set_title(f"{label} {path.stem}", fontsize=9)
            axis.axis("off")
    fig.suptitle("Representative inputs at the common 960×540 transmitted size")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "input_contact_sheet.png", dpi=180)
    plt.close(fig)


def build_comparison(hosted: dict[str, Any], local: dict[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    paths = {
        "proposal_count_mean": ("metrics", "proposal_count", "mean"),
        "proposal_score_median": ("proposal_metrics", "score", "median"),
        "proposal_short_side_median": ("proposal_metrics", "short_side_px", "median"),
        "crops_mean": ("metrics", "crops", "mean"),
        "rejection_fraction_mean": ("metrics", "rejection_fraction", "mean"),
        "mean_objectness_mean": ("metrics", "mean_objectness", "mean"),
        "mean_margin_mean": ("metrics", "mean_margin", "mean"),
        "observations_mean": ("metrics", "observation_count", "mean"),
        "bank_tracks_mean": ("metrics", "bank_tracks", "mean"),
        "track_primary_posterior_median": ("track_metrics", "identity_confidence", "median"),
        "track_posterior_gap_median": ("track_metrics", "posterior_gap", "median"),
        "emitted_mean": ("metrics", "output_emitted", "mean"),
        "distinct_classes_mean": ("metrics", "output_distinct_classes", "mean"),
        "secondary_guesses_mean": ("metrics", "output_secondary_guesses", "mean"),
        "gmc_quality_mean": ("metrics", "gmc_quality", "mean"),
        "gmc_translation_median": ("metrics", "gmc_translation_magnitude", "median"),
        "brightness_mean": ("image_metrics", "brightness_mean", "mean"),
        "contrast_mean": ("image_metrics", "contrast_std", "mean"),
        "entropy_mean": ("image_metrics", "entropy_bits", "mean"),
        "gradient_mean": ("image_metrics", "gradient_mean", "mean"),
        "edge_density_mean": ("image_metrics", "edge_density", "mean"),
        "orb_keypoints_mean": ("image_metrics", "orb_keypoints", "mean"),
    }
    for name, path in paths.items():
        h: Any = hosted
        l: Any = local
        for key in path:
            h = h[key]
            l = l[key]
        comparisons[name] = {"hosted": h, "helsinki": l, "hosted_over_helsinki": ratio(h, l)}
    return comparisons


def generate() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    hosted_records = load_records(HOSTED_DIR / "frames")
    local_records = load_records(LOCAL_TELEMETRY_DIR / "frames")
    hosted = extract_rows("hosted", hosted_records, range(249))
    local = extract_rows("helsinki", local_records, range(25))
    frame_rows = hosted[0] + local[0]
    proposal_rows = hosted[1] + local[1]
    track_rows = hosted[2] + local[2]
    observation_rows = hosted[3] + local[3]
    class_rows = hosted[4] + local[4]

    hosted_images = [
        image_metrics("hosted", path, int(path.stem), False)
        for path in sorted((HOSTED_DIR / "images").glob("*.png"))
    ]
    local_images = [
        image_metrics("helsinki", path, int(path.stem.split("_")[-1]), True)
        for path in sorted(LOCAL_IMAGE_DIR.glob("frame_*.png"))
    ]
    image_rows = hosted_images + local_images

    write_csv(CSV_DIR / "frame_metrics.csv", frame_rows, list(frame_rows[0].keys()))
    write_csv(CSV_DIR / "proposal_metrics.csv", proposal_rows, list(proposal_rows[0].keys()))
    write_csv(CSV_DIR / "track_metrics.csv", track_rows, list(track_rows[0].keys()))
    write_csv(CSV_DIR / "observation_metrics.csv", observation_rows, list(observation_rows[0].keys()))
    write_csv(CSV_DIR / "image_metrics.csv", image_rows, list(image_rows[0].keys()))
    write_csv(CSV_DIR / "class_counts_by_frame.csv", class_rows, list(class_rows[0].keys()))

    hosted_summary = summarize_dataset(
        [row for row in frame_rows if row["dataset"] == "hosted"],
        [row for row in proposal_rows if row["dataset"] == "hosted"],
        [row for row in track_rows if row["dataset"] == "hosted"],
        [row for row in observation_rows if row["dataset"] == "hosted"],
        hosted_images,
    )
    local_summary = summarize_dataset(
        [row for row in frame_rows if row["dataset"] == "helsinki"],
        [row for row in proposal_rows if row["dataset"] == "helsinki"],
        [row for row in track_rows if row["dataset"] == "helsinki"],
        [row for row in observation_rows if row["dataset"] == "helsinki"],
        local_images,
    )
    add_observation_class_counts(hosted_summary, [row for row in class_rows if row["dataset"] == "hosted"])
    add_observation_class_counts(local_summary, [row for row in class_rows if row["dataset"] == "helsinki"])
    payload = {
        "schema_version": 1,
        "scope": "GT-free hosted vs matched Helsinki V2 forensic metrics",
        "sources": {
            "hosted": str(HOSTED_DIR.relative_to(DRONE_ROOT)).replace("\\", "/"),
            "helsinki_telemetry": str(LOCAL_TELEMETRY_DIR.relative_to(DRONE_ROOT)).replace("\\", "/"),
            "helsinki_images": str(LOCAL_IMAGE_DIR.relative_to(DRONE_ROOT)).replace("\\", "/"),
        },
        "hosted": hosted_summary,
        "helsinki": local_summary,
        "comparison": build_comparison(hosted_summary, local_summary),
        "observability": {
            "hosted_ground_truth": "NOT OBSERVABLE",
            "hosted_proposal_recall": "NOT OBSERVABLE",
            "hosted_classification_accuracy": "NOT OBSERVABLE",
            "hosted_iou": "NOT OBSERVABLE",
            "hidden_class_specific_ap": "NOT OBSERVABLE",
            "exact_emitted_class_frequency": "NOT OBSERVABLE: emitted annotation identities are not recorded",
            "exact_emitted_confidence_distribution": "NOT OBSERVABLE: emitted annotation confidences are not recorded",
            "third_class_posterior": "NOT OBSERVABLE",
            "gmc_affine_matrix_magnitude": "NOT OBSERVABLE: telemetry stores translation only",
        },
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    plot_time_series(frame_rows)
    plot_distributions(proposal_rows, track_rows)
    plot_spatial(proposal_rows)
    plot_gmc_output(frame_rows)
    plot_images(image_rows)
    plot_contact_sheet()


def check() -> None:
    expected = {
        "frame_metrics.csv": 1 + 249 + 25,
        "proposal_metrics.csv": None,
        "track_metrics.csv": None,
        "observation_metrics.csv": None,
        "image_metrics.csv": 1 + 50 + 25,
        "class_counts_by_frame.csv": 1 + (248 + 25) * len(CLASS_NAMES),
    }
    for name, expected_lines in expected.items():
        path = CSV_DIR / name
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"missing or empty {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        if expected_lines is not None and len(rows) != expected_lines:
            raise AssertionError(f"{name}: expected {expected_lines} lines, got {len(rows)}")
        widths = {len(row) for row in rows}
        if len(widths) != 1:
            raise AssertionError(f"{name}: inconsistent column counts {widths}")
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert metrics["hosted"]["records_present"] == 248
    assert metrics["hosted"]["missing_frame_indices"] == [1]
    assert metrics["helsinki"]["records_present"] == 25
    assert metrics["hosted"]["resolution_level_counts"] == {"0": 248}
    assert metrics["helsinki"]["resolution_level_counts"] == {"0": 25}
    for name in (
        "pipeline_volume_over_time.png",
        "discovery_track_distributions.png",
        "proposal_spatial_distribution.png",
        "gmc_recognition_output.png",
        "image_characteristics.png",
        "input_contact_sheet.png",
    ):
        path = FIGURE_DIR / name
        if not path.exists() or path.stat().st_size < 10_000:
            raise AssertionError(f"missing or implausibly small figure {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only:
        generate()
    check()
    print("audit artifacts verified")


if __name__ == "__main__":
    main()
