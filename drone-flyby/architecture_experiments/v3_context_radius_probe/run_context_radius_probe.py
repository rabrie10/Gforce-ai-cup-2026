"""Frozen-V2 local-context radius ablation.

This diagnostic keeps a selected target and a centered rectangle of its native
Helsinki surroundings unchanged, replaces every pixel outside that rectangle,
renders through the 3840x2160 -> 960x540 L0 path, and applies the frozen V2
ultra-low diagnostic decoder. It does not train or modify production code.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort


SEED = 20260919
BRANCH = "drone-v3-context-radius-probe"
SOURCE_SIZE = (3840, 2160)
L0_SIZE = (960, 540)
CONTEXT_LEVELS = (1.0, 1.25, 1.5, 2.0, 4.0, 8.0)
HELSINKI_REPLACEMENTS = (0, 9, 14, 24)
HOSTED_REPLACEMENTS = (0, 95, 145, 245)
EXPECTED_HASHES = {
    "drone_yolo11n_l0.pt": "4c44e404e03673e8aa15fc85baf90f2d09b67d90c1fd94c9b6c9a4c1f35204ce",
    "drone_yolo11n_l0.onnx": "d480d3369feba50faa7d1ef0d901f6e5d96f564bf0953bd058f1de42a0ac2ca9",
    "reference_gallery.npz": "ef226de6a9bd624a1e7fd60291812677c5d7815f3a667bfea607b0158867b67e",
}
HOSTED_SCENE_TYPES = {
    0: "open_grass_road",
    95: "river_industrial",
    145: "harbor_industrial",
    245: "dense_urban_blocks",
}

# Pre-registered interpretation gate. A control is valid only when native rank
# is <=20 at IoU>=.30. CASE W: <4 valid controls. CASE L: >=3 valid target
# classes recover in both families, combined recovery at 8x exceeds 1x by at
# least 25 percentage points, and median score recovery improves. Otherwise N.
MIN_VALID_CONTROLS = 4
MIN_CLASSES_RECOVERED_BOTH = 3
MIN_RECOVERY_RATE_GAIN = 0.25


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_prior(prior_script: Path):
    spec = importlib.util.spec_from_file_location("background_transfer_probe", prior_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load prior probe: {prior_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def centered_context_box(bbox: Sequence[int], multiplier: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    requested_w = (x2 - x1) * multiplier
    requested_h = (y2 - y1) * multiplier
    left = max(0, math.floor(cx - requested_w / 2.0))
    top = max(0, math.floor(cy - requested_h / 2.0))
    right = min(SOURCE_SIZE[0], math.ceil(cx + requested_w / 2.0))
    bottom = min(SOURCE_SIZE[1], math.ceil(cy + requested_h / 2.0))
    if not (left <= x1 < x2 <= right and top <= y1 < y2 <= bottom):
        raise RuntimeError("Context box does not preserve the full target bbox")
    return left, top, right, bottom


def choose_replacements(targets) -> dict[str, dict]:
    mapping = {}
    for index, target in enumerate(targets):
        eligible = [frame for frame in HELSINKI_REPLACEMENTS if frame != target.frame_id]
        helsinki_id = eligible[index % len(eligible)]
        hosted_id = HOSTED_REPLACEMENTS[index % len(HOSTED_REPLACEMENTS)]
        mapping[target.object_id] = {
            "helsinki_frame_id": helsinki_id,
            "hosted_frame_id": hosted_id,
            "hosted_scene_type": HOSTED_SCENE_TYPES[hosted_id],
        }
    return mapping


def render_native(source: np.ndarray) -> np.ndarray:
    return cv2.resize(source, L0_SIZE, interpolation=cv2.INTER_AREA)


def render_context_composite(
    source: np.ndarray, replacement: np.ndarray, context_box: Sequence[int]
) -> np.ndarray:
    left, top, right, bottom = context_box
    composite = replacement.copy()
    composite[top:bottom, left:right] = source[top:bottom, left:right]
    if not np.array_equal(
        composite[top:bottom, left:right], source[top:bottom, left:right]
    ):
        raise RuntimeError("Preserved context pixels changed")
    return cv2.resize(composite, L0_SIZE, interpolation=cv2.INTER_AREA)


def evaluate(decoder, image: np.ndarray, target_box: Sequence[float], prior) -> dict:
    candidates = decoder.decode(image, target_box)
    rank030, score030 = prior.candidate_summary(candidates, 0.30)
    rank050, score050 = prior.candidate_summary(candidates, 0.50)
    max_iou = max((candidate.iou for candidate in candidates), default=0.0)
    return {
        "rank_at_iou_030": rank030,
        "rank_at_iou_050": rank050,
        "best_matching_score": score030 if score030 is not None else 0.0,
        "best_matching_score_iou_050": score050 if score050 is not None else 0.0,
        "max_iou": max_iou,
        "above_production_threshold": bool(score030 is not None and score030 >= prior.PRODUCTION_THRESHOLD),
        "proposal_count": len(candidates),
        "recall_at_1_iou_030": bool(rank030 is not None and rank030 <= 1),
        "recall_at_5_iou_030": bool(rank030 is not None and rank030 <= 5),
        "recall_at_20_iou_030": bool(rank030 is not None and rank030 <= 20),
        "recall_at_1_iou_050": bool(rank050 is not None and rank050 <= 1),
        "recall_at_5_iou_050": bool(rank050 is not None and rank050 <= 5),
        "recall_at_20_iou_050": bool(rank050 is not None and rank050 <= 20),
    }


def csv_value(value):
    return "" if value is None else value


def native_row(target, metrics: dict) -> dict:
    x1, y1, x2, y2 = target.bbox
    valid = bool(metrics["rank_at_iou_030"] is not None and metrics["rank_at_iou_030"] <= 20)
    return {
        "source_class": target.object_id,
        "source_frame": target.frame_id,
        "source_bbox_x1": x1,
        "source_bbox_y1": y1,
        "source_bbox_x2": x2,
        "source_bbox_y2": y2,
        "rank_at_iou_030": csv_value(metrics["rank_at_iou_030"]),
        "rank_at_iou_050": csv_value(metrics["rank_at_iou_050"]),
        "best_matching_score": metrics["best_matching_score"],
        "best_matching_score_iou_050": metrics["best_matching_score_iou_050"],
        "max_iou": metrics["max_iou"],
        "above_production_threshold": metrics["above_production_threshold"],
        "proposal_count": metrics["proposal_count"],
        "recall_at_1_iou_030": metrics["recall_at_1_iou_030"],
        "recall_at_5_iou_030": metrics["recall_at_5_iou_030"],
        "recall_at_20_iou_030": metrics["recall_at_20_iou_030"],
        "recall_at_1_iou_050": metrics["recall_at_1_iou_050"],
        "recall_at_5_iou_050": metrics["recall_at_5_iou_050"],
        "recall_at_20_iou_050": metrics["recall_at_20_iou_050"],
        "valid_native_control": valid,
        "weak_control_reason": "" if valid else "native target not top-20 at IoU>=0.30",
    }


def context_row(target, family: str, replacement_id: str, multiplier: float,
                context_box: Sequence[int], metrics: dict, native: dict) -> dict:
    left, top, right, bottom = context_box
    native_rank_value = native["rank_at_iou_030"]
    native_rank = None if native_rank_value == "" else int(native_rank_value)
    rank = metrics["rank_at_iou_030"]
    missing_rank = metrics["proposal_count"] + 1
    rank_for_normalization = rank if rank is not None else missing_rank
    score_ratio = (
        metrics["best_matching_score"] / native["best_matching_score"]
        if native["best_matching_score"] > 0 else None
    )
    iou_ratio = metrics["max_iou"] / native["max_iou"] if native["max_iou"] > 0 else None
    valid = bool(native["valid_native_control"])
    recovered = bool(
        valid
        and rank is not None
        and rank <= 20
        and score_ratio is not None
        and score_ratio >= 0.5
    )
    x1, y1, x2, y2 = target.bbox
    return {
        "source_class": target.object_id,
        "source_frame": target.frame_id,
        "replacement_family": family,
        "replacement_background_id": replacement_id,
        "context_multiplier": multiplier,
        "requested_context_width": (x2 - x1) * multiplier,
        "requested_context_height": (y2 - y1) * multiplier,
        "context_x1": left,
        "context_y1": top,
        "context_x2": right,
        "context_y2": bottom,
        "effective_context_width": right - left,
        "effective_context_height": bottom - top,
        "context_clipped": bool(
            right - left < math.ceil((x2 - x1) * multiplier)
            or bottom - top < math.ceil((y2 - y1) * multiplier)
        ),
        "target_x1_l0": x1 / 4.0,
        "target_y1_l0": y1 / 4.0,
        "target_x2_l0": x2 / 4.0,
        "target_y2_l0": y2 / 4.0,
        "rank_at_iou_030": csv_value(rank),
        "rank_at_iou_050": csv_value(metrics["rank_at_iou_050"]),
        "best_matching_score": metrics["best_matching_score"],
        "best_matching_score_iou_050": metrics["best_matching_score_iou_050"],
        "max_iou": metrics["max_iou"],
        "above_production_threshold": metrics["above_production_threshold"],
        "proposal_count": metrics["proposal_count"],
        "recall_at_1_iou_030": metrics["recall_at_1_iou_030"],
        "recall_at_5_iou_030": metrics["recall_at_5_iou_030"],
        "recall_at_20_iou_030": metrics["recall_at_20_iou_030"],
        "recall_at_1_iou_050": metrics["recall_at_1_iou_050"],
        "recall_at_5_iou_050": metrics["recall_at_5_iou_050"],
        "recall_at_20_iou_050": metrics["recall_at_20_iou_050"],
        "native_rank_at_iou_030": csv_value(native_rank),
        "native_best_matching_score": native["best_matching_score"],
        "native_max_iou": native["max_iou"],
        "valid_native_control": valid,
        "rank_delta_from_native": (
            rank_for_normalization - native_rank if native_rank is not None else ""
        ),
        "rank_recovery_ratio_native_over_context": (
            native_rank / rank_for_normalization if native_rank is not None else ""
        ),
        "score_recovery_ratio": csv_value(score_ratio),
        "iou_recovery_ratio": csv_value(iou_ratio),
        "top20_status_restored": bool(valid and rank is not None and rank <= 20),
        "activation_recovered": recovered,
    }


def mean_bool(rows: Sequence[dict], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else 0.0


def median_numeric(rows: Sequence[dict], key: str):
    values = [float(row[key]) for row in rows if row[key] != "" and row[key] is not None]
    return float(np.median(values)) if values else None


def aggregate_curve(rows: Sequence[dict]) -> dict:
    result = {}
    for family in ("helsinki", "hosted_validation", "all"):
        result[family] = {}
        family_rows = rows if family == "all" else [r for r in rows if r["replacement_family"] == family]
        for level in CONTEXT_LEVELS:
            subset = [r for r in family_rows if r["context_multiplier"] == level]
            valid = [r for r in subset if r["valid_native_control"]]
            result[family][f"{level:g}x"] = {
                "n": len(subset),
                "valid_n": len(valid),
                "recall_at_1_iou_030": mean_bool(subset, "recall_at_1_iou_030"),
                "recall_at_5_iou_030": mean_bool(subset, "recall_at_5_iou_030"),
                "recall_at_20_iou_030": mean_bool(subset, "recall_at_20_iou_030"),
                "recall_at_1_iou_050": mean_bool(subset, "recall_at_1_iou_050"),
                "recall_at_5_iou_050": mean_bool(subset, "recall_at_5_iou_050"),
                "recall_at_20_iou_050": mean_bool(subset, "recall_at_20_iou_050"),
                "production_threshold_crossing_rate": mean_bool(subset, "above_production_threshold"),
                "activation_recovery_rate": mean_bool(valid, "activation_recovered"),
                "median_rank_delta_from_native": median_numeric(valid, "rank_delta_from_native"),
                "median_score_recovery_ratio": median_numeric(valid, "score_recovery_ratio"),
                "median_iou_recovery_ratio": median_numeric(valid, "iou_recovery_ratio"),
                "median_proposal_count": median_numeric(subset, "proposal_count"),
            }
    return result


def per_target_recovery(rows: Sequence[dict], native_rows: Sequence[dict]) -> dict:
    native_by_class = {row["source_class"]: row for row in native_rows}
    result = {}
    for object_id, native in native_by_class.items():
        result[object_id] = {
            "valid_native_control": native["valid_native_control"],
            "native_rank_at_iou_030": native["rank_at_iou_030"],
            "native_best_matching_score": native["best_matching_score"],
            "native_max_iou": native["max_iou"],
            "families": {},
        }
        for family in ("helsinki", "hosted_validation"):
            subset = sorted(
                [r for r in rows if r["source_class"] == object_id and r["replacement_family"] == family],
                key=lambda r: r["context_multiplier"],
            )
            recovered = [r["context_multiplier"] for r in subset if r["activation_recovered"]]
            result[object_id]["families"][family] = {
                "minimum_recovery_radius": min(recovered) if recovered else None,
                "recovered_by_8x": bool(recovered),
                "levels": {
                    f"{r['context_multiplier']:g}x": {
                        "rank_at_iou_030": r["rank_at_iou_030"],
                        "score_recovery_ratio": r["score_recovery_ratio"],
                        "iou_recovery_ratio": r["iou_recovery_ratio"],
                        "top20_status_restored": r["top20_status_restored"],
                        "activation_recovered": r["activation_recovered"],
                    }
                    for r in subset
                },
            }
    return result


def classify(curve: dict, target_recovery: dict, valid_count: int) -> dict:
    recovered_both = [
        object_id for object_id, target in target_recovery.items()
        if target["valid_native_control"]
        and all(target["families"][family]["recovered_by_8x"] for family in ("helsinki", "hosted_validation"))
    ]
    at_1 = curve["all"]["1x"]["activation_recovery_rate"]
    at_8 = curve["all"]["8x"]["activation_recovery_rate"]
    score_1 = curve["all"]["1x"]["median_score_recovery_ratio"]
    score_8 = curve["all"]["8x"]["median_score_recovery_ratio"]
    if valid_count < MIN_VALID_CONTROLS:
        case, label = "CASE W", "weak control"
        explanation = "Fewer than four selected targets were top-20 native controls, so the radius recovery comparison is not reliable."
    elif (
        len(recovered_both) >= MIN_CLASSES_RECOVERED_BOTH
        and at_8 - at_1 >= MIN_RECOVERY_RATE_GAIN
        and score_1 is not None and score_8 is not None and score_8 > score_1
    ):
        case, label = "CASE L", "local context memorization supported"
        explanation = "Activation recovery increased materially with preserved context and occurred across multiple classes in both replacement families."
    else:
        case, label = "CASE N", "no strong local-context recovery"
        explanation = "Increasing the preserved local rectangle did not satisfy the pre-registered multi-class recovery gate."
    return {
        "case": case,
        "label": label,
        "explanation": explanation,
        "valid_native_controls": valid_count,
        "classes_recovered_in_both_families": recovered_both,
        "recovery_rate_1x": at_1,
        "recovery_rate_8x": at_8,
        "recovery_rate_gain_1x_to_8x": at_8 - at_1,
        "median_score_recovery_ratio_1x": score_1,
        "median_score_recovery_ratio_8x": score_8,
        "recommended_action": "Begin V3 discovery implementation.",
    }


def make_contact_sheet(qc: dict, output_path: Path) -> None:
    display_levels = ("native", 1.0, 1.5, 2.0, 4.0, 8.0)
    rows = []
    for object_id in list(qc)[:3]:
        target = qc[object_id]
        x1, y1, x2, y2 = target["bbox_l0"]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        panels = []
        for level in display_levels:
            image = target["images"][level].copy()
            crop_w, crop_h = 240, 160
            left = max(0, min(L0_SIZE[0] - crop_w, cx - crop_w // 2))
            top = max(0, min(L0_SIZE[1] - crop_h, cy - crop_h // 2))
            crop = image[top:top + crop_h, left:left + crop_w].copy()
            cv2.rectangle(crop, (int(x1) - left, int(y1) - top), (int(x2) - left, int(y2) - top), (0, 255, 255), 1)
            crop = cv2.resize(crop, (360, 240), interpolation=cv2.INTER_NEAREST)
            label = "native" if level == "native" else f"{level:g}x"
            cv2.rectangle(crop, (0, 0), (360, 27), (0, 0, 0), -1)
            cv2.putText(crop, f"{object_id} | hosted | {label}", (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(crop)
        rows.append(np.hstack(panels))
    sheet = np.vstack(rows)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Could not write {output_path}")


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def fmt_rank(value) -> str:
    return "none" if value == "" or value is None else str(value)


def build_report(summary: dict, manifest: dict, native_rows: Sequence[dict]) -> str:
    classification = summary["classification"]
    curve = summary["recovery_curve"]
    per_target = summary["per_target"]
    lines = [
        "# Frozen V2 local-context radius ablation",
        "",
        "## Executive result",
        "",
        f"**{classification['case']} — {classification['label'].upper()}.** {classification['explanation']}",
        "",
        "This is the final forensic diagnostic before V3. It measures frozen V2 activation recovery as progressively larger native Helsinki rectangles are preserved around the same target, while all outside pixels are replaced.",
        "",
        "## Experiment identity",
        "",
        f"- Branch: `{manifest['git']['branch']}`",
        f"- Experiment input commit: `{manifest['git']['input_head']}`",
        f"- Frozen V2 lineage: `{manifest['frozen_v2_commit']}`",
        f"- Valid native controls: **{classification['valid_native_controls']}/8**",
        f"- Random seed: `{manifest['random_seed']}`",
        f"- Context composites: **{manifest['counts']['context_composites']}**",
        f"- Runtime: OpenCV `{manifest['versions']['opencv']}`, NumPy `{manifest['versions']['numpy']}`, ONNX Runtime `{manifest['versions']['onnxruntime']}` on `{manifest['versions']['onnx_providers'][0]}`",
        "",
        "## Native control check",
        "",
        "| Class | Frame | Rank .30 | Rank .50 | Score .30 | Max IoU | >=0.01 | Proposals | Valid |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in native_rows:
        lines.append(
            f"| {row['source_class']} | {row['source_frame']} | {fmt_rank(row['rank_at_iou_030'])} | {fmt_rank(row['rank_at_iou_050'])} | {row['best_matching_score']:.6f} | {row['max_iou']:.3f} | {row['above_production_threshold']} | {row['proposal_count']} | {row['valid_native_control']} |"
        )
    lines += [
        "",
        "A valid recovery control was pre-defined as native-original rank <=20 at IoU >=0.30. Other targets remain reported but do not count in normalized recovery rates.",
        "",
        "## Recovery curve normalized to native",
        "",
        "| Family | Context | R@20 .30 | R@20 .50 | Activation recovered | Median rank delta | Median score ratio | Median IoU ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family in ("helsinki", "hosted_validation", "all"):
        for level in CONTEXT_LEVELS:
            row = curve[family][f"{level:g}x"]
            lines.append(
                f"| {family} | {level:g}x | {pct(row['recall_at_20_iou_030'])} | {pct(row['recall_at_20_iou_050'])} | {pct(row['activation_recovery_rate'])} | {row['median_rank_delta_from_native']} | {row['median_score_recovery_ratio']} | {row['median_iou_recovery_ratio']} |"
            )
    lines += [
        "",
        "Primary recovery criterion (fixed before inference): top-20 at IoU >=0.30 and matching score >=50% of that target's native-original score.",
        "",
        "## Result by target class and replacement family",
        "",
        "| Class | Native valid | Helsinki minimum recovery | Hosted minimum recovery | Recovered in both |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for object_id, target in per_target.items():
        hel = target["families"]["helsinki"]["minimum_recovery_radius"]
        hosted = target["families"]["hosted_validation"]["minimum_recovery_radius"]
        both = hel is not None and hosted is not None
        lines.append(f"| {object_id} | {target['valid_native_control']} | {hel if hel is not None else 'none'} | {hosted if hosted is not None else 'none'} | {both} |")
    lines += [
        "",
        "## Rendering and controls",
        "",
        "- Each target remains at its native source-frame coordinates and scale.",
        "- The preserved rectangle is centered on the annotation bbox. For multiplier `m`, requested width/height are `m*bbox_width` and `m*bbox_height`; edges use floor/ceil and clip to source bounds. Requested and effective dimensions are recorded per row.",
        "- Every pixel inside the effective rectangle is copied byte-for-byte from the native frame; everything outside comes from one deterministic replacement image.",
        "- Hosted 960x540 captures are expanded to 3840x2160 with nearest-neighbor replication. Every composite then uses `cv2.INTER_AREA` for the exact 3840x2160 -> 960x540 L0 render.",
        "- Frozen ONNX decoding uses class-score max pooling, confidence floor 1e-4, V2 IoU 0.55 NMS, and no production top-K budget.",
        "",
        "## Interpretation gate",
        "",
        f"The gate was fixed in code before inference: CASE W if fewer than {MIN_VALID_CONTROLS} native targets are top-20 at IoU >=0.30. Otherwise CASE L requires at least {MIN_CLASSES_RECOVERED_BOTH} target classes to recover in both background families, a combined 8x-minus-1x recovery-rate gain of at least {MIN_RECOVERY_RATE_GAIN:.0%}, and a higher median score-recovery ratio at 8x. All other valid-control outcomes are CASE N.",
        "",
        f"Observed combined activation recovery changed from **{pct(classification['recovery_rate_1x'])}** at 1x to **{pct(classification['recovery_rate_8x'])}** at 8x. Classes recovering in both families: **{', '.join(classification['classes_recovered_in_both_families']) or 'none'}**.",
        "",
        f"**Final classification: {classification['case']} — {classification['label']}.**",
        "",
        "## Limitations",
        "",
        "- Rectangular replacement boundaries are an intervention artifact; increasing radius moves that boundary away but does not reproduce an unmodified whole frame until the rectangle reaches frame bounds.",
        "- One physical Helsinki instance per class tests context dependence for those selected instances, not generalization to unseen instances.",
        "- Hosted captures have no ground truth and are used only as replacement scenery; unrelated proposals can affect rank.",
        "- Only one deterministic replacement image per family is used for each target to keep the final probe compact.",
        "",
        "## Exactly one recommended next action",
        "",
        "**Begin V3 discovery implementation.**",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "& 'drone-flyby/.venv/Scripts/python.exe' 'drone-flyby/architecture_experiments/v3_context_radius_probe/run_context_radius_probe.py'",
        "```",
        "",
        "The harness fails closed on branch mismatch, frozen-artifact hash drift, missing inputs, changed target selection, invalid preserved pixels, or unexpected row counts. Full-resolution composites are generated only with `--save-composites` and remain ignored.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-composites", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(__file__).resolve().parent
    drone_dir = experiment_dir.parents[1]
    repo = drone_dir.parent
    branch = git(repo, "branch", "--show-current")
    if branch != BRANCH:
        raise RuntimeError(f"Refusing to run on branch {branch!r}; expected {BRANCH!r}")

    prior_dir = experiment_dir.parent / "v3_background_transfer_probe"
    prior = load_prior(prior_dir / "run_background_transfer_probe.py")
    prior_manifest = json.loads((prior_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    model_dir = drone_dir / "models"
    artifact_hashes = {}
    for name, expected in EXPECTED_HASHES.items():
        actual = prior.sha256(model_dir / name)
        artifact_hashes[name] = {"expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected:
            raise RuntimeError(f"Frozen artifact hash mismatch for {name}: {actual}")

    images_dir = drone_dir / "src" / "helsinki" / "images"
    annotations_dir = drone_dir / "src" / "helsinki" / "annotations"
    hosted_dir = drone_dir / "hosted_telemetry" / "v2_validation_6a86911a" / "run_20260919_005220" / "images"
    targets = prior.select_target_sources(annotations_dir, images_dir)
    expected_targets = [
        (row["object_id"], row["frame_id"], tuple(row["bbox"]))
        for row in prior_manifest["target_sources"]
    ]
    actual_targets = [(row.object_id, row.frame_id, tuple(row.bbox)) for row in targets]
    if actual_targets != expected_targets:
        raise RuntimeError(f"Prior target selection changed: {actual_targets}")
    replacements = choose_replacements(targets)

    helsinki_backgrounds = {}
    for frame_id in sorted({value["helsinki_frame_id"] for value in replacements.values()}):
        path = images_dir / f"frame_{frame_id:06d}.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (SOURCE_SIZE[1], SOURCE_SIZE[0]):
            raise RuntimeError(f"Missing/wrong-sized Helsinki replacement: {path}")
        helsinki_backgrounds[frame_id] = image
    hosted_backgrounds = {}
    for frame_id in sorted({value["hosted_frame_id"] for value in replacements.values()}):
        path = hosted_dir / f"{frame_id:05d}.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (L0_SIZE[1], L0_SIZE[0]):
            raise RuntimeError(f"Missing/wrong-sized hosted replacement: {path}")
        hosted_backgrounds[frame_id] = cv2.resize(image, SOURCE_SIZE, interpolation=cv2.INTER_NEAREST)

    decoder = prior.DiagnosticDecoder(model_dir / "drone_yolo11n_l0.onnx")
    cache_dir = experiment_dir / "cache"
    if args.save_composites:
        cache_dir.mkdir(exist_ok=True)
    native_rows = []
    context_rows = []
    qc = {}
    native_metrics_by_class = {}

    for index, target in enumerate(targets):
        source = cv2.imread(target.image_path, cv2.IMREAD_COLOR)
        if source is None or source.shape[:2] != (SOURCE_SIZE[1], SOURCE_SIZE[0]):
            raise RuntimeError(f"Missing/wrong-sized source: {target.image_path}")
        target_box_l0 = tuple(value / 4.0 for value in target.bbox)
        native_image = render_native(source)
        native_metrics = evaluate(decoder, native_image, target_box_l0, prior)
        nrow = native_row(target, native_metrics)
        native_rows.append(nrow)
        native_metrics_by_class[target.object_id] = nrow
        replacement = replacements[target.object_id]
        qc[target.object_id] = {"bbox_l0": target_box_l0, "images": {"native": native_image}}
        print(f"native {index + 1}/8 {target.object_id}: rank={nrow['rank_at_iou_030']} score={nrow['best_matching_score']:.6f}", flush=True)

        for family, replacement_source, replacement_id in (
            ("helsinki", helsinki_backgrounds[replacement["helsinki_frame_id"]], f"frame_{replacement['helsinki_frame_id']:06d}"),
            ("hosted_validation", hosted_backgrounds[replacement["hosted_frame_id"]], f"{replacement['hosted_frame_id']:05d}"),
        ):
            for multiplier in CONTEXT_LEVELS:
                context_box = centered_context_box(target.bbox, multiplier)
                rendered = render_context_composite(source, replacement_source, context_box)
                metrics = evaluate(decoder, rendered, target_box_l0, prior)
                context_rows.append(context_row(
                    target, family, replacement_id, multiplier, context_box, metrics, nrow
                ))
                if family == "hosted_validation" and multiplier in (1.0, 1.5, 2.0, 4.0, 8.0):
                    qc[target.object_id]["images"][multiplier] = rendered
                if args.save_composites:
                    name = f"{target.object_id}_{family}_{multiplier:g}x.png"
                    cv2.imwrite(str(cache_dir / name), rendered)
            print(f"  {family} complete", flush=True)

    if len(native_rows) != 8 or len(context_rows) != 8 * 2 * len(CONTEXT_LEVELS):
        raise RuntimeError("Unexpected experiment row count")
    curve = aggregate_curve(context_rows)
    target_recovery = per_target_recovery(context_rows, native_rows)
    valid_count = sum(bool(row["valid_native_control"]) for row in native_rows)
    classification = classify(curve, target_recovery, valid_count)
    summary = {
        "classification": classification,
        "recovery_curve": curve,
        "per_target": target_recovery,
    }
    manifest = {
        "experiment": "frozen_v2_local_context_radius_ablation",
        "frozen_v2_commit": "6fb764e5b9fb5f7eefcb832c72b35553165219e5",
        "prior_background_transfer_commit": "8a79714031f678ff8b5297616d3d8a1bb783b0cb",
        "git": {"branch": branch, "input_head": git(repo, "rev-parse", "HEAD")},
        "random_seed": SEED,
        "artifact_hashes": artifact_hashes,
        "target_sources": prior_manifest["target_sources"],
        "replacement_assignments": replacements,
        "config": {
            "context_levels": list(CONTEXT_LEVELS),
            "context_math": "center=(bbox edge midpoint); requested size=multiplier*bbox size; floor left/top; ceil right/bottom; clip to 3840x2160",
            "preservation": "byte-exact rectangular copy from native source before L0 rendering",
            "source_size": list(SOURCE_SIZE),
            "detector_input_size": list(L0_SIZE),
            "hosted_expansion": "cv2.INTER_NEAREST 960x540 -> 3840x2160",
            "rendering": "cv2.INTER_AREA 3840x2160 -> 960x540",
            "diagnostic_decode_floor": prior.DIAGNOSTIC_FLOOR,
            "production_confidence_floor": prior.PRODUCTION_THRESHOLD,
            "nms_iou": prior.NMS_IOU,
            "production_budget_not_applied": True,
            "class_identity": "pooled away by maximum class response",
            "primary_recovery_criterion": "rank<=20 at IoU>=0.30 and score>=50% of native score",
            "interpretation_gate": {
                "case_w": f"valid native controls < {MIN_VALID_CONTROLS}",
                "case_l": f">={MIN_CLASSES_RECOVERED_BOTH} classes recover in both families, 8x-vs-1x recovery gain >={MIN_RECOVERY_RATE_GAIN}, and median score ratio increases",
                "case_n": "all other outcomes with sufficient valid controls",
            },
        },
        "counts": {
            "source_targets": len(targets),
            "native_controls": len(native_rows),
            "valid_native_controls": valid_count,
            "context_composites": len(context_rows),
            "replacement_families": 2,
        },
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "onnx_providers": decoder.session.get_providers(),
        },
    }
    write_csv(experiment_dir / "native_control_metrics.csv", native_rows)
    write_csv(experiment_dir / "context_metrics.csv", context_rows)
    write_json(experiment_dir / "context_radius_summary.json", summary)
    write_json(experiment_dir / "experiment_manifest.json", manifest)
    make_contact_sheet(qc, experiment_dir / "context_radius_qc.png")
    (experiment_dir / "CONTEXT_RADIUS_REPORT.md").write_text(
        build_report(summary, manifest, native_rows), encoding="utf-8", newline="\n"
    )
    print(json.dumps({"classification": classification, "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
