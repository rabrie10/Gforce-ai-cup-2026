"""Paired background-transfer replay for the frozen Architecture V2 detector.

This is a diagnostic experiment, not a production path.  It composites the
same tight rectangular Helsinki target patch onto paired Helsinki and hosted
scenery, renders both through the L0 3840x2160 -> 960x540 path, and decodes the
frozen ONNX detector at an ultra-low confidence floor without the production
proposal budget.

Raw hosted captures and generated composites remain outside Git.  The script
writes only derived CSV/JSON, a compact QC contact sheet, and this report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import onnxruntime as ort


SEED = 20260919
SOURCE_SIZE = (3840, 2160)
L0_SIZE = (960, 540)
DIAGNOSTIC_FLOOR = 1e-4
NMS_IOU = 0.55
PRODUCTION_THRESHOLD = 0.01
TARGET_SHORT_SIDES = {
    "lt8": 6,
    "8to16": 12,
    "gt16": 24,
}
HOSTED_FRAME_IDS = (0, 95, 145, 245)
HELSINKI_BACKGROUND_IDS = (0, 9, 14, 24)
SELECTED_CLASSES = (
    "condor",
    "helicopter",
    "jammer",
    "large_launcher",
    "small_launcher",
    "spacecraft",
    "ta-ta",
    "tank",
)
HOSTED_SCENE_TYPES = {
    0: "open_grass_road",
    45: "airport_woodland",
    95: "river_industrial",
    145: "harbor_industrial",
    195: "dense_urban_rail",
    245: "dense_urban_blocks",
}
EXPECTED_HASHES = {
    "drone_yolo11n_l0.pt": "4c44e404e03673e8aa15fc85baf90f2d09b67d90c1fd94c9b6c9a4c1f35204ce",
    "drone_yolo11n_l0.onnx": "d480d3369feba50faa7d1ef0d901f6e5d96f564bf0953bd058f1de42a0ac2ca9",
    "reference_gallery.npz": "ef226de6a9bd624a1e7fd60291812677c5d7815f3a667bfea607b0158867b67e",
}


@dataclass(frozen=True)
class TargetSource:
    object_id: str
    frame_id: int
    bbox: tuple[int, int, int, int]
    image_path: str


@dataclass(frozen=True)
class Candidate:
    rank: int
    bbox: tuple[float, float, float, float]
    score: float
    iou: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL
    ).strip()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def overlaps_any(box: Sequence[float], boxes: Iterable[Sequence[float]]) -> bool:
    for other in boxes:
        if min(box[2], other[2]) > max(box[0], other[0]) and min(box[3], other[3]) > max(box[1], other[1]):
            return True
    return False


class DiagnosticDecoder:
    """Frozen V2 ONNX inference with the production decode but no top-K budget."""

    def __init__(self, model_path: Path, threads: int = 4) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def letterbox(image: np.ndarray, target_long_side: int = 960, stride: int = 32):
        height, width = image.shape[:2]
        scale = min(target_long_side / width, target_long_side / height)
        new_width = max(stride, int(round(width * scale)))
        new_height = max(stride, int(round(height * scale)))
        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        padded_width = int(math.ceil(new_width / stride) * stride)
        padded_height = int(math.ceil(new_height / stride) * stride)
        canvas = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
        canvas[:new_height, :new_width] = resized
        return canvas, new_width / width, new_height / height

    def decode(self, image: np.ndarray, target_box: Sequence[float]) -> list[Candidate]:
        canvas, scale_x, scale_y = self.letterbox(image)
        tensor = canvas[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        prediction = self.session.run(
            None, {self.input_name: np.ascontiguousarray(tensor)}
        )[0]
        if prediction.ndim != 3:
            raise RuntimeError(f"Unexpected ONNX output shape: {prediction.shape}")
        rows = prediction[0]
        if rows.shape[0] < rows.shape[1]:
            rows = rows.T
        boxes_xywh = rows[:, :4]
        scores = rows[:, 4:].max(axis=1)
        keep = scores >= DIAGNOSTIC_FLOOR
        boxes_xywh, scores = boxes_xywh[keep], scores[keep]
        if not len(scores):
            return []

        half_w, half_h = boxes_xywh[:, 2] * 0.5, boxes_xywh[:, 3] * 0.5
        x1 = (boxes_xywh[:, 0] - half_w) / scale_x
        y1 = (boxes_xywh[:, 1] - half_h) / scale_y
        x2 = (boxes_xywh[:, 0] + half_w) / scale_x
        y2 = (boxes_xywh[:, 1] + half_h) / scale_y
        rects = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        indices = cv2.dnn.NMSBoxes(
            rects.tolist(),
            scores.astype(float).tolist(),
            DIAGNOSTIC_FLOOR,
            NMS_IOU,
        )
        if indices is None or len(indices) == 0:
            return []
        order = np.asarray(indices).reshape(-1)
        order = order[np.argsort(-scores[order], kind="stable")]
        height, width = image.shape[:2]
        candidates: list[Candidate] = []
        for index in order:
            box = (
                float(max(0.0, x1[index])),
                float(max(0.0, y1[index])),
                float(min(width, x2[index])),
                float(min(height, y2[index])),
            )
            if box[2] - box[0] < 1.0 or box[3] - box[1] < 1.0:
                continue
            candidates.append(
                Candidate(
                    rank=len(candidates) + 1,
                    bbox=box,
                    score=float(scores[index]),
                    iou=box_iou(box, target_box),
                )
            )
        return candidates


def select_target_sources(annotations_dir: Path, images_dir: Path) -> list[TargetSource]:
    """Choose one full, non-border, median-sized view per physical class."""

    by_class: dict[str, list[TargetSource]] = defaultdict(list)
    for annotation_path in sorted(annotations_dir.glob("frame_*.json")):
        payload = read_json(annotation_path)
        frame_id = int(payload["frame"])
        image_path = images_dir / f"frame_{frame_id:06d}.png"
        for annotation in payload["annotations"]:
            x1, y1, x2, y2 = map(int, annotation["bbox"])
            if x1 <= 0 or y1 <= 0 or x2 >= SOURCE_SIZE[0] or y2 >= SOURCE_SIZE[1]:
                continue
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            by_class[annotation["object_id"]].append(
                TargetSource(
                    object_id=annotation["object_id"],
                    frame_id=frame_id,
                    bbox=(x1, y1, x2, y2),
                    image_path=str(image_path),
                )
            )
    selected: list[TargetSource] = []
    for object_id in sorted(by_class):
        rows = sorted(
            by_class[object_id],
            key=lambda row: ((row.bbox[2] - row.bbox[0]) * (row.bbox[3] - row.bbox[1]), row.frame_id),
        )
        selected.append(rows[len(rows) // 2])
    missing = sorted(set(SELECTED_CLASSES) - {row.object_id for row in selected})
    if missing:
        raise RuntimeError(f"Missing requested source classes: {missing}")
    return [row for row in selected if row.object_id in SELECTED_CLASSES]


def resized_target_patch(source: TargetSource, short_side_l0: int) -> np.ndarray:
    image = cv2.imread(source.image_path, cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (SOURCE_SIZE[1], SOURCE_SIZE[0]):
        raise RuntimeError(f"Invalid Helsinki source image: {source.image_path}")
    x1, y1, x2, y2 = source.bbox
    patch = image[y1:y2, x1:x2]
    height, width = patch.shape[:2]
    source_short = min(width, height)
    scale = (short_side_l0 * 4) / source_short
    out_width = max(4, int(round(width * scale / 4.0)) * 4)
    out_height = max(4, int(round(height * scale / 4.0)) * 4)
    # Force the requested short side exactly while preserving aspect as closely as possible.
    if width <= height:
        out_width = short_side_l0 * 4
    else:
        out_height = short_side_l0 * 4
    interpolation = cv2.INTER_AREA if out_width < width or out_height < height else cv2.INTER_CUBIC
    return cv2.resize(patch, (out_width, out_height), interpolation=interpolation)


def gradient_map(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def placement_for_pair(
    helsinki_l0: np.ndarray,
    hosted_l0: np.ndarray,
    patch_size_l0: tuple[int, int],
    helsinki_boxes_l0: Sequence[Sequence[float]],
    salt: int,
) -> tuple[int, int]:
    """Choose a quiet shared location, diversified among the five best cells."""

    width, height = patch_size_l0
    margin_x, margin_y = max(32, width + 8), max(32, height + 8)
    xs = np.linspace(margin_x, L0_SIZE[0] - margin_x - width, 12).astype(int)
    ys = np.linspace(margin_y, L0_SIZE[1] - margin_y - height, 7).astype(int)
    grad_h, grad_v = gradient_map(helsinki_l0), gradient_map(hosted_l0)
    options: list[tuple[float, int, int]] = []
    for y in ys:
        for x in xs:
            box = (x, y, x + width, y + height)
            if overlaps_any(box, helsinki_boxes_l0):
                continue
            pad = 8
            ysli = slice(max(0, y - pad), min(L0_SIZE[1], y + height + pad))
            xsli = slice(max(0, x - pad), min(L0_SIZE[0], x + width + pad))
            score = float(grad_h[ysli, xsli].mean() + grad_v[ysli, xsli].mean())
            options.append((score, int(x), int(y)))
    if not options:
        raise RuntimeError("No non-overlapping placement available")
    options.sort()
    _, x, y = options[salt % min(5, len(options))]
    return x, y


def render_composite(background_source: np.ndarray, patch: np.ndarray, x_l0: int, y_l0: int):
    x_source, y_source = x_l0 * 4, y_l0 * 4
    height, width = patch.shape[:2]
    composite = background_source.copy()
    composite[y_source:y_source + height, x_source:x_source + width] = patch
    rendered = cv2.resize(composite, L0_SIZE, interpolation=cv2.INTER_AREA)
    target_box = (
        float(x_l0),
        float(y_l0),
        float(x_l0 + width // 4),
        float(y_l0 + height // 4),
    )
    return rendered, target_box


def candidate_summary(candidates: Sequence[Candidate], threshold: float) -> tuple[int | None, float | None]:
    matches = [candidate for candidate in candidates if candidate.iou >= threshold]
    if not matches:
        return None, None
    best_ranked = min(matches, key=lambda candidate: candidate.rank)
    return best_ranked.rank, best_ranked.score


def percentile(values: Sequence[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def bootstrap_ci(values: Sequence[float], rng: np.random.Generator, iterations: int = 10000):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return [None, None]
    draws = rng.integers(0, len(array), size=(iterations, len(array)))
    means = array[draws].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def aggregate(rows: Sequence[dict]) -> dict:
    result: dict[str, dict] = {}
    for family in ("helsinki", "hosted_validation"):
        result[family] = {}
        for size_bin in (*TARGET_SHORT_SIDES, "all"):
            selected = [
                row for row in rows
                if row["background_family"] == family
                and (size_bin == "all" or row["size_bin"] == size_bin)
            ]
            metrics = {"n": len(selected)}
            for iou_label in ("030", "050"):
                for k in (1, 5, 20):
                    metrics[f"recall_at_{k}_iou_{iou_label}"] = float(np.mean([
                        row[f"rank_at_iou_{iou_label}"] != ""
                        and int(row[f"rank_at_iou_{iou_label}"]) <= k
                        for row in selected
                    ]))
            for key in ("target_score_iou_030", "max_iou", "proposal_count_before_topk", "background_proposal_count"):
                values = [float(row[key]) for row in selected]
                metrics[f"{key}_median"] = percentile(values, 50)
                metrics[f"{key}_p10"] = percentile(values, 10)
                metrics[f"{key}_p90"] = percentile(values, 90)
            ranks = [
                int(row["rank_at_iou_030"])
                for row in selected if row["rank_at_iou_030"] != ""
            ]
            metrics["matched_rank_iou_030_median"] = percentile(ranks, 50)
            metrics["matched_rank_iou_030_p90"] = percentile(ranks, 90)
            metrics["production_threshold_cross_rate_iou_030"] = float(np.mean([
                row["crosses_production_threshold"] for row in selected
            ]))
            result[family][size_bin] = metrics
    return result


def paired_analysis(pair_rows: Sequence[dict]) -> dict:
    rng = np.random.default_rng(SEED + 1)
    result: dict[str, dict] = {}
    for size_bin in (*TARGET_SHORT_SIDES, "all"):
        selected = [row for row in pair_rows if size_bin == "all" or row["size_bin"] == size_bin]
        metrics: dict[str, object] = {"n": len(selected)}
        for key in ("rank_delta_iou_030", "score_delta_iou_030", "iou_delta"):
            values = [float(row[key]) for row in selected]
            metrics[f"{key}_mean"] = float(np.mean(values))
            metrics[f"{key}_median"] = float(np.median(values))
            metrics[f"{key}_mean_95ci"] = bootstrap_ci(values, rng)
        ratios = [float(row["score_ratio_iou_030"]) for row in selected if row["score_ratio_iou_030"] != ""]
        metrics["score_ratio_iou_030_median"] = percentile(ratios, 50)
        for k in (1, 5, 20):
            metrics[f"top{k}_loss_rate_iou_030"] = float(np.mean([
                row[f"crosses_out_of_top{k}_iou_030"] for row in selected
            ]))
            metrics[f"top{k}_gain_rate_iou_030"] = float(np.mean([
                row[f"crosses_into_top{k}_iou_030"] for row in selected
            ]))
        metrics["production_threshold_loss_rate"] = float(np.mean([
            row["crosses_below_production_threshold"] for row in selected
        ]))
        metrics["production_threshold_gain_rate"] = float(np.mean([
            row["crosses_above_production_threshold"] for row in selected
        ]))
        result[size_bin] = metrics
    return result


def classify(aggregate_metrics: dict, paired_metrics: dict) -> tuple[str, str]:
    hel = aggregate_metrics["helsinki"]["all"]
    hosted = aggregate_metrics["hosted_validation"]["all"]
    paired = paired_metrics["all"]
    h20 = hel["recall_at_20_iou_030"]
    v20 = hosted["recall_at_20_iou_030"]
    retained = v20 / h20 if h20 else 0.0
    score_ratio = paired["score_ratio_iou_030_median"]
    rank_delta = paired["rank_delta_iou_030_median"]
    if h20 >= 0.5 and retained >= 0.8 and score_ratio is not None and score_ratio <= 0.5 and rank_delta <= 5:
        return "CASE A", "calibration / threshold dominant"
    if h20 >= 0.5 and (retained <= 0.7 or paired["top20_loss_rate_iou_030"] >= 0.2):
        return "CASE B", "representation / background-conditioned objectness failure"
    return "CASE C", "background hypothesis weak or not isolated strongly enough"


def make_contact_sheet(examples: Sequence[dict], output_path: Path) -> None:
    panels = []
    for example in examples:
        row_panels = []
        for family in ("helsinki", "hosted_validation"):
            image = example[family].copy()
            x1, y1, x2, y2 = map(int, example["bbox"])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            crop_w, crop_h = 240, 150
            left = max(0, min(L0_SIZE[0] - crop_w, cx - crop_w // 2))
            top = max(0, min(L0_SIZE[1] - crop_h, cy - crop_h // 2))
            crop = image[top:top + crop_h, left:left + crop_w].copy()
            cv2.rectangle(crop, (x1 - left, y1 - top), (x2 - left, y2 - top), (0, 255, 255), 2)
            crop = cv2.resize(crop, (480, 300), interpolation=cv2.INTER_NEAREST)
            label = f"{family} | {example['size_bin']} | {example['object_id']}"
            cv2.rectangle(crop, (0, 0), (480, 30), (0, 0, 0), -1)
            cv2.putText(crop, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            row_panels.append(crop)
        panels.append(np.hstack(row_panels))
    sheet = np.vstack(panels)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Could not write {output_path}")


def markdown_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def build_report(summary: dict, manifest: dict) -> str:
    aggregate_metrics = summary["aggregate"]
    paired = summary["paired"]["all"]
    classification = summary["classification"]
    lines = [
        "# Frozen V2 paired background-transfer replay",
        "",
        "## Executive result",
        "",
        f"**{classification['case']} — {classification['label'].upper()}.** {classification['explanation']}",
        "",
        "This is a controlled compositing result, not a measurement of hidden Validation recall. There is no hosted ground truth, and hosted captures are used only as scenery evidence.",
        "",
        "## Experiment identity",
        "",
        f"- Branch: `{manifest['git']['branch']}`",
        f"- Experiment input HEAD: `{manifest['git']['input_head']}`",
        f"- Frozen V2 lineage: `{manifest['frozen_v2_commit']}`",
        f"- Random seed: `{manifest['random_seed']}`",
        f"- Paired examples: **{manifest['counts']['pairs']}** ({manifest['counts']['composites']} composites)",
        f"- Source Helsinki targets: **{manifest['counts']['source_targets']}**, one selected view for each of eight physical classes",
        f"- Hosted backgrounds: **{manifest['counts']['hosted_backgrounds']}**, time-stratified across scene types",
        f"- Scale-bin counts: `{manifest['counts']['pairs_by_scale_bin']}`",
        f"- Runtime: OpenCV `{manifest['versions']['opencv']}`, NumPy `{manifest['versions']['numpy']}`, ONNX Runtime `{manifest['versions']['onnxruntime']}` on `{manifest['versions']['onnx_providers'][0]}`",
        "",
        "## Primary recall metrics",
        "",
        "| Background | Scale | R@1 IoU .30 | R@5 IoU .30 | R@20 IoU .30 | R@1 IoU .50 | R@5 IoU .50 | R@20 IoU .50 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family in ("helsinki", "hosted_validation"):
        for size_bin in (*TARGET_SHORT_SIDES, "all"):
            row = aggregate_metrics[family][size_bin]
            lines.append(
                f"| {family} | {size_bin} | {markdown_pct(row['recall_at_1_iou_030'])} | {markdown_pct(row['recall_at_5_iou_030'])} | {markdown_pct(row['recall_at_20_iou_030'])} | {markdown_pct(row['recall_at_1_iou_050'])} | {markdown_pct(row['recall_at_5_iou_050'])} | {markdown_pct(row['recall_at_20_iou_050'])} |"
            )
    rank_ci = paired["rank_delta_iou_030_mean_95ci"]
    score_ci = paired["score_delta_iou_030_mean_95ci"]
    iou_ci = paired["iou_delta_mean_95ci"]
    lines.extend([
        "",
        "## Paired result",
        "",
        "Deltas are hosted-minus-Helsinki for the exact same target patch and geometry. Rank uses the first score-ranked proposal at IoU >= 0.30; a missing match is assigned one past the larger proposal count in that pair.",
        "",
        f"- Target-rank delta: median **{paired['rank_delta_iou_030_median']:.2f}**, mean **{paired['rank_delta_iou_030_mean']:.2f}** (paired bootstrap 95% CI **{rank_ci[0]:.2f} to {rank_ci[1]:.2f}**). Positive is worse on hosted scenery.",
        f"- Target-score delta: median **{paired['score_delta_iou_030_median']:.6f}**, mean **{paired['score_delta_iou_030_mean']:.6f}** (95% CI **{score_ci[0]:.6f} to {score_ci[1]:.6f}**).",
        f"- Target-score ratio (hosted/Helsinki): median **{paired['score_ratio_iou_030_median'] if paired['score_ratio_iou_030_median'] is not None else 'n/a'}**.",
        f"- Best-IoU delta: median **{paired['iou_delta_median']:.4f}**, mean **{paired['iou_delta_mean']:.4f}** (95% CI **{iou_ci[0]:.4f} to {iou_ci[1]:.4f}**).",
        f"- Crossed out of top-20 at IoU >= 0.30: **{markdown_pct(paired['top20_loss_rate_iou_030'])}**; crossed into top-20: **{markdown_pct(paired['top20_gain_rate_iou_030'])}**.",
        f"- Crossed below production threshold 0.01: **{markdown_pct(paired['production_threshold_loss_rate'])}**; crossed above: **{markdown_pct(paired['production_threshold_gain_rate'])}**.",
        "",
        f"**Control-sensitivity caveat:** Helsinki-composite Recall@20 at IoU 0.30 was only {markdown_pct(aggregate_metrics['helsinki']['all']['recall_at_20_iou_030'])}. The synthetic copy-paste control therefore does not recreate a high-rank local baseline. Case C is supported by the absence—and reversal—of a hosted-background degradation, but the low control sensitivity limits how strongly this probe can exclude subtler context effects.",
        "",
        "## Diagnostic distributions",
        "",
        "All values are p10 / median / p90. Target score is the highest-ranked match at IoU >= 0.30; matched rank is conditional on such a match existing. Background proposals mean candidates below IoU 0.30 relative to the inserted target, not labeled negatives.",
        "",
        "| Background | Scale | Matched rank median / p90 | Target score | Best target IoU | Proposals before top-K | Background/non-target proposals |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for family in ("helsinki", "hosted_validation"):
        for size_bin in (*TARGET_SHORT_SIDES, "all"):
            row = aggregate_metrics[family][size_bin]
            lines.append(
                f"| {family} | {size_bin} | {row['matched_rank_iou_030_median']} / {row['matched_rank_iou_030_p90']} | "
                f"{row['target_score_iou_030_p10']:.6f} / {row['target_score_iou_030_median']:.6f} / {row['target_score_iou_030_p90']:.6f} | "
                f"{row['max_iou_p10']:.3f} / {row['max_iou_median']:.3f} / {row['max_iou_p90']:.3f} | "
                f"{row['proposal_count_before_topk_p10']:.1f} / {row['proposal_count_before_topk_median']:.1f} / {row['proposal_count_before_topk_p90']:.1f} | "
                f"{row['background_proposal_count_p10']:.1f} / {row['background_proposal_count_median']:.1f} / {row['background_proposal_count_p90']:.1f} |"
            )
    lines.extend([
        "",
        "Full rank, score, IoU, proposal-volume, background-proposal, boundary-crossing, scale, class, and background records are in `pair_metrics.csv`; every post-NMS diagnostic candidate is in `candidate_metrics.csv`.",
        "",
        "## Rendering and controls",
        "",
        "- Helsinki scenery stays at its native 3840×2160 source resolution.",
        "- Each hosted 960×540 capture is expanded to 3840×2160 with nearest-neighbor replication. The subsequent 4× `INTER_AREA` render therefore preserves the original hosted scenery pixels outside the pasted rectangle.",
        "- A tight annotated Helsinki bbox is cropped, resized at source resolution, and pasted at a 4-source-pixel-aligned location. The complete 3840×2160 composite is then rendered to 960×540 with `cv2.INTER_AREA`.",
        "- The same resized patch bytes, x/y location, zero rotation, bbox geometry, and render path are used in each pair. A byte equality assertion verifies the final transmitted target rectangle in every pair.",
        "- Placement is selected from a fixed grid among low-gradient regions in both backgrounds and excludes known Helsinki annotation boxes. This reduces unnecessary overwriting of salient structures; it does not assert that hosted regions are target-free.",
        "- The frozen ONNX head is decoded at `1e-4`, with V2's class pooling and IoU 0.55 NMS, but without its confidence 0.01 floor or 32-proposal truncation.",
        "",
        "## Bounding-box caveat",
        "",
        "The pasted target is a tight rectangular annotation crop, not a segmentation cutout. It can contain original Helsinki background pixels inside the bbox. Because that exact rectangle is reused byte-for-byte within each pair, the experiment isolates sensitivity to surrounding/wider scenery context, not pure foreground/background isolation.",
        "",
        "Hosted images are not negative labels. They may contain hidden challenge targets, so all non-matching candidates are called background/non-target *relative to the inserted synthetic target only*.",
        "",
        "## Interpretation gate",
        "",
        "The gate was fixed in code before interpretation: Case A requires Helsinki R@20 >= 0.50, at least 80% R@20 retention, median score ratio <= 0.5, and median rank degradation <= 5. Case B requires Helsinki R@20 >= 0.50 plus either <=70% R@20 retention or >=20% paired top-20 losses. All other outcomes are Case C.",
        "",
        f"**Final classification: {classification['case']} — {classification['label']}.**",
        "",
        "## Limitations",
        "",
        "- No hosted ground truth exists; nothing here estimates actual hidden Validation recall, target prevalence, target scale, or class accuracy.",
        "- Hosted backgrounds originate as transmitted 960×540 captures. Nearest-neighbor expansion makes source-resolution compositing deterministic but cannot reconstruct source detail that was never captured.",
        "- Rectangular crops retain some local Helsinki context inside each bbox.",
        "- Repeated use of one physical Helsinki instance per class avoids adjacent-frame inflation but does not test new physical-instance generalization.",
        "- Existing unlabeled objects in either background can compete in rank; paired analysis and four diverse scenes expose rather than label that competition.",
        "",
        "## Exactly one recommended next action",
        "",
        f"**{classification['recommended_action']}**",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "& 'drone-flyby/.venv/Scripts/python.exe' 'drone-flyby/architecture_experiments/v3_background_transfer_probe/run_background_transfer_probe.py'",
        "```",
        "",
        "The script fails closed on branch mismatch, artifact hash drift, missing/wrong-sized inputs, pair pixel inequality, unbalanced bins, or inconsistent row counts. Generated composites are reproducible in the ignored `cache/` directory when `--save-composites` is supplied.",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-composites", action="store_true")
    parser.add_argument("--quick", action="store_true", help="One class/background per bin; smoke test only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    experiment_dir = script_path.parent
    drone_dir = experiment_dir.parents[1]
    repo = drone_dir.parent
    branch = git(repo, "branch", "--show-current")
    if branch != "drone-v3-background-transfer-probe":
        raise RuntimeError(f"Refusing to run on branch {branch!r}")

    model_dir = drone_dir / "models"
    artifact_hashes = {}
    for name, expected in EXPECTED_HASHES.items():
        path = model_dir / name
        actual = sha256(path)
        artifact_hashes[name] = {"expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected:
            raise RuntimeError(f"Frozen artifact hash mismatch for {name}: {actual}")

    images_dir = drone_dir / "src" / "helsinki" / "images"
    annotations_dir = drone_dir / "src" / "helsinki" / "annotations"
    hosted_dir = drone_dir / "hosted_telemetry" / "v2_validation_6a86911a" / "run_20260919_005220" / "images"
    targets = select_target_sources(annotations_dir, images_dir)
    if args.quick:
        targets = targets[:1]
        hosted_ids = HOSTED_FRAME_IDS[:1]
        helsinki_background_ids = HELSINKI_BACKGROUND_IDS[:1]
    else:
        hosted_ids = HOSTED_FRAME_IDS
        helsinki_background_ids = HELSINKI_BACKGROUND_IDS

    hosted_backgrounds = {}
    for frame_id in hosted_ids:
        path = hosted_dir / f"{frame_id:05d}.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (L0_SIZE[1], L0_SIZE[0]):
            raise RuntimeError(f"Missing/wrong-sized hosted background: {path}")
        hosted_backgrounds[frame_id] = image

    helsinki_sources = {}
    helsinki_annotations = {}
    for frame_id in helsinki_background_ids:
        path = images_dir / f"frame_{frame_id:06d}.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (SOURCE_SIZE[1], SOURCE_SIZE[0]):
            raise RuntimeError(f"Missing/wrong-sized Helsinki background: {path}")
        helsinki_sources[frame_id] = image
        payload = read_json(annotations_dir / f"frame_{frame_id:06d}.json")
        helsinki_annotations[frame_id] = [
            tuple(float(value) / 4.0 for value in row["bbox"])
            for row in payload["annotations"]
        ]

    decoder = DiagnosticDecoder(model_dir / "drone_yolo11n_l0.onnx")
    cache_dir = experiment_dir / "cache"
    if args.save_composites:
        cache_dir.mkdir(exist_ok=True)

    composite_rows: list[dict] = []
    candidate_rows: list[dict] = []
    pair_rows: list[dict] = []
    qc_examples: list[dict] = []
    pair_index = 0
    for target_index, target in enumerate(targets):
        for size_index, (size_bin, short_side_l0) in enumerate(TARGET_SHORT_SIDES.items()):
            patch = resized_target_patch(target, short_side_l0)
            patch_h_l0, patch_w_l0 = patch.shape[0] // 4, patch.shape[1] // 4
            for background_index, (hosted_id, helsinki_id) in enumerate(zip(hosted_ids, helsinki_background_ids)):
                pair_id = f"p{pair_index:04d}"
                helsinki_source = helsinki_sources[helsinki_id]
                helsinki_l0 = cv2.resize(helsinki_source, L0_SIZE, interpolation=cv2.INTER_AREA)
                hosted_l0 = hosted_backgrounds[hosted_id]
                x_l0, y_l0 = placement_for_pair(
                    helsinki_l0,
                    hosted_l0,
                    (patch_w_l0, patch_h_l0),
                    helsinki_annotations[helsinki_id],
                    salt=target_index * 17 + size_index * 7 + background_index,
                )
                hosted_source = cv2.resize(hosted_l0, SOURCE_SIZE, interpolation=cv2.INTER_NEAREST)
                rendered = {}
                rows_for_pair = {}
                for family, background_source, background_id, background_type in (
                    ("helsinki", helsinki_source, f"frame_{helsinki_id:06d}", "helsinki_forest_lakeshore"),
                    ("hosted_validation", hosted_source, f"{hosted_id:05d}", HOSTED_SCENE_TYPES[hosted_id]),
                ):
                    image, target_box = render_composite(background_source, patch, x_l0, y_l0)
                    rendered[family] = image
                    candidates = decoder.decode(image, target_box)
                    rank030, score030 = candidate_summary(candidates, 0.30)
                    rank050, score050 = candidate_summary(candidates, 0.50)
                    max_iou = max((candidate.iou for candidate in candidates), default=0.0)
                    best_iou_candidate = max(candidates, key=lambda item: item.iou, default=None)
                    row = {
                        "pair_id": pair_id,
                        "background_family": family,
                        "background_id": background_id,
                        "background_type": background_type,
                        "size_bin": size_bin,
                        "target_short_side_l0": min(patch_w_l0, patch_h_l0),
                        "source_class": target.object_id,
                        "source_frame": target.frame_id,
                        "source_bbox_x1": target.bbox[0],
                        "source_bbox_y1": target.bbox[1],
                        "source_bbox_x2": target.bbox[2],
                        "source_bbox_y2": target.bbox[3],
                        "target_x1": target_box[0],
                        "target_y1": target_box[1],
                        "target_x2": target_box[2],
                        "target_y2": target_box[3],
                        "proposal_count_before_topk": len(candidates),
                        "background_proposal_count": sum(candidate.iou < 0.30 for candidate in candidates),
                        "rank_at_iou_030": rank030 if rank030 is not None else "",
                        "target_score_iou_030": score030 if score030 is not None else 0.0,
                        "rank_at_iou_050": rank050 if rank050 is not None else "",
                        "target_score_iou_050": score050 if score050 is not None else 0.0,
                        "max_iou": max_iou,
                        "best_iou_candidate_rank": best_iou_candidate.rank if best_iou_candidate else "",
                        "best_iou_candidate_score": best_iou_candidate.score if best_iou_candidate else 0.0,
                        "crosses_production_threshold": bool(score030 is not None and score030 >= PRODUCTION_THRESHOLD),
                    }
                    composite_rows.append(row)
                    rows_for_pair[family] = row
                    for candidate in candidates:
                        candidate_rows.append({
                            "pair_id": pair_id,
                            "background_family": family,
                            "candidate_rank": candidate.rank,
                            "candidate_score": round(candidate.score, 8),
                            "candidate_x1": round(candidate.bbox[0], 5),
                            "candidate_y1": round(candidate.bbox[1], 5),
                            "candidate_x2": round(candidate.bbox[2], 5),
                            "candidate_y2": round(candidate.bbox[3], 5),
                            "target_iou": round(candidate.iou, 8),
                        })
                    if args.save_composites:
                        cv2.imwrite(str(cache_dir / f"{pair_id}_{family}.png"), image)

                x1, y1, x2, y2 = map(int, target_box)
                if not np.array_equal(
                    rendered["helsinki"][y1:y2, x1:x2],
                    rendered["hosted_validation"][y1:y2, x1:x2],
                ):
                    raise RuntimeError(f"Pair invariant failed: target pixels differ in {pair_id}")

                hel, hosted = rows_for_pair["helsinki"], rows_for_pair["hosted_validation"]
                missing_rank = max(hel["proposal_count_before_topk"], hosted["proposal_count_before_topk"]) + 1
                hel_rank = int(hel["rank_at_iou_030"]) if hel["rank_at_iou_030"] != "" else missing_rank
                hosted_rank = int(hosted["rank_at_iou_030"]) if hosted["rank_at_iou_030"] != "" else missing_rank
                pair_row = {
                    "pair_id": pair_id,
                    "size_bin": size_bin,
                    "target_short_side_l0": min(patch_w_l0, patch_h_l0),
                    "source_class": target.object_id,
                    "source_frame": target.frame_id,
                    "helsinki_background_id": hel["background_id"],
                    "hosted_background_id": hosted["background_id"],
                    "hosted_background_type": hosted["background_type"],
                    "target_x1": target_box[0],
                    "target_y1": target_box[1],
                    "target_x2": target_box[2],
                    "target_y2": target_box[3],
                    "target_pixels_equal": True,
                    "helsinki_rank_iou_030": hel["rank_at_iou_030"],
                    "hosted_rank_iou_030": hosted["rank_at_iou_030"],
                    "rank_delta_iou_030": hosted_rank - hel_rank,
                    "helsinki_score_iou_030": hel["target_score_iou_030"],
                    "hosted_score_iou_030": hosted["target_score_iou_030"],
                    "score_delta_iou_030": hosted["target_score_iou_030"] - hel["target_score_iou_030"],
                    "score_ratio_iou_030": (
                        hosted["target_score_iou_030"] / hel["target_score_iou_030"]
                        if hel["target_score_iou_030"] > 0 else ""
                    ),
                    "helsinki_max_iou": hel["max_iou"],
                    "hosted_max_iou": hosted["max_iou"],
                    "iou_delta": hosted["max_iou"] - hel["max_iou"],
                    "helsinki_proposal_count": hel["proposal_count_before_topk"],
                    "hosted_proposal_count": hosted["proposal_count_before_topk"],
                    "helsinki_background_proposals": hel["background_proposal_count"],
                    "hosted_background_proposals": hosted["background_proposal_count"],
                }
                for k in (1, 5, 20):
                    hel_inside = hel["rank_at_iou_030"] != "" and int(hel["rank_at_iou_030"]) <= k
                    hosted_inside = hosted["rank_at_iou_030"] != "" and int(hosted["rank_at_iou_030"]) <= k
                    pair_row[f"crosses_out_of_top{k}_iou_030"] = bool(hel_inside and not hosted_inside)
                    pair_row[f"crosses_into_top{k}_iou_030"] = bool(not hel_inside and hosted_inside)
                pair_row["crosses_below_production_threshold"] = bool(
                    hel["crosses_production_threshold"] and not hosted["crosses_production_threshold"]
                )
                pair_row["crosses_above_production_threshold"] = bool(
                    not hel["crosses_production_threshold"] and hosted["crosses_production_threshold"]
                )
                pair_rows.append(pair_row)
                if target_index == 0 and background_index == size_index % len(hosted_ids):
                    qc_examples.append({
                        "size_bin": size_bin,
                        "object_id": target.object_id,
                        "bbox": target_box,
                        "helsinki": rendered["helsinki"],
                        "hosted_validation": rendered["hosted_validation"],
                    })
                pair_index += 1
                print(f"[{pair_index}] {pair_id} {target.object_id} {size_bin} hosted={hosted_id}", flush=True)

    counts_by_bin = Counter(row["size_bin"] for row in pair_rows)
    if not args.quick and set(counts_by_bin.values()) != {32}:
        raise RuntimeError(f"Unbalanced scale bins: {counts_by_bin}")
    if len(composite_rows) != 2 * len(pair_rows):
        raise RuntimeError("Composite/pair row mismatch")
    aggregate_metrics = aggregate(composite_rows)
    paired_metrics = paired_analysis(pair_rows)
    case, label = classify(aggregate_metrics, paired_metrics)
    if case == "CASE A":
        recommended_action = "Replace V2's fixed 0.01 discovery floor with a diagnostic-supported low-floor, fixed-proposal-budget policy in the next separately authorized V3 design."
        explanation = "Targets generally retained top-20 rank while their absolute activation fell on hosted scenery."
    elif case == "CASE B":
        recommended_action = "Train one class-agnostic discovery model with broad background randomization/copy-paste in the next separately authorized V3 experiment."
        explanation = "Identical target pixels frequently lost top-20 rank on hosted scenery, so threshold adjustment alone is not sufficient."
    else:
        recommended_action = "Run one paired physical-instance-transfer probe that holds scale and scenery fixed while swapping Helsinki target instances for genuinely independent target instances."
        explanation = (
            "Hosted scenery did not degrade the identical inserted patches: aggregate Recall@20 at IoU 0.30 was higher, "
            "top-20 gains exceeded losses, and paired rank/score uncertainty included no change. The observed hosted failure "
            "therefore is not explained mainly by wider scenery context in this controlled probe."
        )

    summary = {
        "classification": {
            "case": case,
            "label": label,
            "explanation": explanation,
            "recommended_action": recommended_action,
        },
        "aggregate": aggregate_metrics,
        "paired": paired_metrics,
    }
    manifest = {
        "experiment": "frozen_v2_paired_background_transfer_replay",
        "frozen_v2_commit": "6fb764e5b9fb5f7eefcb832c72b35553165219e5",
        "git": {"branch": branch, "input_head": git(repo, "rev-parse", "HEAD")},
        "random_seed": SEED,
        "artifact_hashes": artifact_hashes,
        "config": {
            "frozen_v2_production_proposer": {
                "backend": "yolo",
                "budget": 32,
                "confidence_floor": 0.01,
                "nms_iou": 0.55,
                "imgsz": 960,
                "onnx_threads": 4,
                "merge_iou": 0.45,
                "score_pooling": "maximum class response; identity discarded",
            },
            "diagnostic_decode_floor": DIAGNOSTIC_FLOOR,
            "production_confidence_floor": PRODUCTION_THRESHOLD,
            "nms_iou": NMS_IOU,
            "production_budget_not_applied": True,
            "source_size": list(SOURCE_SIZE),
            "detector_input_size": list(L0_SIZE),
            "hosted_source_expansion": "cv2.INTER_NEAREST 960x540 -> 3840x2160",
            "l0_render": "cv2.INTER_AREA 3840x2160 -> 960x540",
            "target_rotation_degrees": 0,
            "target_source_alignment_pixels": 4,
            "target_short_sides_l0": TARGET_SHORT_SIDES,
            "hosted_frame_ids": list(hosted_ids),
            "helsinki_background_frame_ids": list(helsinki_background_ids),
            "hosted_scene_types": {str(key): value for key, value in HOSTED_SCENE_TYPES.items() if key in hosted_ids},
            "source_selection": "one non-border median-area annotated view per class",
            "placement": "shared fixed grid; one of five lowest combined-gradient cells; excludes known Helsinki boxes",
        },
        "counts": {
            "source_targets": len(targets),
            "hosted_backgrounds": len(hosted_ids),
            "helsinki_backgrounds": len(helsinki_background_ids),
            "pairs": len(pair_rows),
            "composites": len(composite_rows),
            "candidates": len(candidate_rows),
            "pairs_by_scale_bin": dict(sorted(counts_by_bin.items())),
        },
        "target_sources": [asdict(target) for target in targets],
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "onnx_providers": decoder.session.get_providers(),
        },
        "limitations": {
            "rectangular_patches_include_local_helsinki_background": True,
            "hosted_ground_truth_available": False,
            "hosted_images_used_as_negative_labels": False,
            "claim_scope": "sensitivity to surrounding/wider scenery context only",
        },
    }

    write_csv(experiment_dir / "pair_metrics.csv", pair_rows, list(pair_rows[0]))
    write_csv(experiment_dir / "candidate_metrics.csv", candidate_rows, list(candidate_rows[0]))
    write_json(experiment_dir / "background_transfer_summary.json", summary)
    write_json(experiment_dir / "experiment_manifest.json", manifest)
    if not args.quick:
        make_contact_sheet(qc_examples, experiment_dir / "representative_pairs.png")
        (experiment_dir / "BACKGROUND_TRANSFER_REPORT.md").write_text(
            build_report(summary, manifest), encoding="utf-8", newline="\n"
        )
    print(json.dumps({"classification": summary["classification"], "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
