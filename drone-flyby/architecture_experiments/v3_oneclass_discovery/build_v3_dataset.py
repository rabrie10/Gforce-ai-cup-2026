"""Build the deterministic one-class V3 discovery dataset.

All labels are class 0 (``target``). Synthetic targets are composed at
3840x2160 and only then rendered to challenge L0 with cv2.INTER_AREA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
DRONE = HERE.parents[1]
HELSINKI = DRONE / "src" / "helsinki"
HOSTED = DRONE / "hosted_telemetry" / "v2_validation_6a86911a" / "run_20260919_005220" / "images"
OUT = HERE / "generated" / "dataset"
SOURCE_SIZE = (3840, 2160)
L0_SIZE = (960, 540)
SEED = 20260919
TRAIN_TARGET_FRAMES = tuple(range(0, 19))
EVAL_TARGET_FRAMES = tuple(range(19, 25))
TRAIN_BACKGROUND_IDS = tuple(range(0, 150, 5))
EVAL_BACKGROUND_IDS = tuple(range(150, 250, 5))
CONTEXT_MULTIPLIERS = (1.0, 1.25, 1.5, 2.0, 4.0)
SIZE_BINS = {
    "lt8": (4.0, 7.75),
    "8to16": (8.0, 16.0),
    "gt16": (16.25, 32.0),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_annotations(frame: int) -> list[dict]:
    path = HELSINKI / "annotations" / f"frame_{frame:06d}.json"
    return json.loads(path.read_text(encoding="utf-8"))["annotations"]


def image(frame: int) -> np.ndarray:
    path = HELSINKI / "images" / f"frame_{frame:06d}.png"
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None or value.shape[:2] != (SOURCE_SIZE[1], SOURCE_SIZE[0]):
        raise RuntimeError(f"Missing/wrong-sized Helsinki image: {path}")
    return value


def hosted_texture(frame: int, rng: random.Random) -> np.ndarray:
    """Use hosted imagery as scenery texture, never as a target-free example.

    Downsampling to 120x68 before source expansion removes coherent target-scale
    detail; a random cyclic offset avoids a fixed correspondence across samples.
    Every resulting image receives an inserted, labelled target.
    """
    path = HOSTED / f"{frame:05d}.png"
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None or value.shape[:2] != (L0_SIZE[1], L0_SIZE[0]):
        raise RuntimeError(f"Missing/wrong-sized hosted image: {path}")
    texture = cv2.resize(value, (120, 68), interpolation=cv2.INTER_AREA)
    texture = cv2.resize(texture, SOURCE_SIZE, interpolation=cv2.INTER_CUBIC)
    # A small reflect-padded affine offset varies alignment without creating
    # the wrap seam that np.roll would put into the training signal.
    dx, dy = rng.uniform(-400, 400), rng.uniform(-240, 240)
    return cv2.warpAffine(texture, np.float32([[1, 0, dx], [0, 1, dy]]), SOURCE_SIZE,
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def context_crop(source: np.ndarray, bbox: list[int], multiplier: float) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    x1, y1, x2, y2 = map(float, bbox)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    width, height = x2 - x1, y2 - y1
    rx1 = max(0, math.floor(cx - width * multiplier / 2))
    ry1 = max(0, math.floor(cy - height * multiplier / 2))
    rx2 = min(SOURCE_SIZE[0], math.ceil(cx + width * multiplier / 2))
    ry2 = min(SOURCE_SIZE[1], math.ceil(cy + height * multiplier / 2))
    patch = source[ry1:ry2, rx1:rx2].copy()
    return patch, (x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1)


def feather_mask(height: int, width: int) -> np.ndarray:
    edge = max(4, min(192, round(min(height, width) * 0.28)))
    y = np.minimum(np.arange(height), np.arange(height)[::-1]).astype(np.float32)
    x = np.minimum(np.arange(width), np.arange(width)[::-1]).astype(np.float32)
    ramp_y = np.clip((y + 1) / edge, 0, 1)
    ramp_x = np.clip((x + 1) / edge, 0, 1)
    mask = np.minimum(ramp_y[:, None], ramp_x[None, :])
    return (0.5 - 0.5 * np.cos(mask * np.pi))[..., None]


def photometric(value: np.ndarray, rng: random.Random) -> np.ndarray:
    work = value.astype(np.float32)
    work = (work - 127.5) * rng.uniform(0.65, 1.45) + 127.5 + rng.uniform(-45, 45)
    hsv = cv2.cvtColor(np.uint8(np.clip(work, 0, 255)), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-12, 12)) % 180
    hsv[..., 1] *= rng.uniform(0.55, 1.5)
    hsv[..., 2] *= rng.uniform(0.75, 1.25)
    work = cv2.cvtColor(np.uint8(np.clip(hsv, 0, 255)), cv2.COLOR_HSV2BGR)
    temperature = rng.uniform(-0.18, 0.18)
    work = work.astype(np.float32)
    work[..., 2] *= 1.0 + temperature
    work[..., 0] *= 1.0 - temperature
    work = np.uint8(np.clip(work, 0, 255))
    if rng.random() < 0.45:
        sigma = rng.uniform(0.25, 1.4)
        work = cv2.GaussianBlur(work, (0, 0), sigma)
    elif rng.random() < 0.35:
        blur = cv2.GaussianBlur(work, (0, 0), rng.uniform(0.4, 1.0))
        work = cv2.addWeighted(work, rng.uniform(1.15, 1.6), blur, rng.uniform(-0.6, -0.15), 0)
    if rng.random() < 0.55:
        quality = rng.randint(35, 92)
        ok, encoded = cv2.imencode(".jpg", work, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            work = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return work


def yolo_line(box: tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = box
    return f"0 {(x1+x2)/(2*L0_SIZE[0]):.8f} {(y1+y2)/(2*L0_SIZE[1]):.8f} {(x2-x1)/L0_SIZE[0]:.8f} {(y2-y1)/L0_SIZE[1]:.8f}"


def write_example(split: str, name: str, value: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> None:
    image_dir, label_dir = OUT / "images" / split, OUT / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_dir / f"{name}.jpg"), value, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(name)
    (label_dir / f"{name}.txt").write_text("\n".join(yolo_line(b) for b in boxes) + "\n", encoding="utf-8")


def build_native(records: list[dict]) -> None:
    for split, frames in (("train", TRAIN_TARGET_FRAMES), ("native_eval", EVAL_TARGET_FRAMES)):
        for frame in frames:
            source = image(frame)
            rendered = cv2.resize(source, L0_SIZE, interpolation=cv2.INTER_AREA)
            boxes = [tuple(float(v) / 4 for v in row["bbox"]) for row in read_annotations(frame)]
            write_example(split, f"native_{frame:06d}", rendered, boxes)
            records.append({"split": split, "kind": "native", "name": f"native_{frame:06d}", "target_frame": frame,
                            "target_count": len(boxes), "background_family": "helsinki_native"})


def build_synthetic(split: str, count_per_bin: int, target_frames: tuple[int, ...], background_ids: tuple[int, ...], records: list[dict]) -> None:
    rng = random.Random(SEED + (0 if split == "train" else 100000))
    sources = [(frame, row) for frame in target_frames for row in read_annotations(frame)]
    source_cache: dict[int, np.ndarray] = {}
    for bin_index, (size_bin, limits) in enumerate(SIZE_BINS.items()):
        for index in range(count_per_bin):
            local = random.Random(rng.randrange(1 << 62))
            frame, annotation = sources[(index * 37 + bin_index * 11) % len(sources)]
            background_id = background_ids[(index * 7 + bin_index * 3) % len(background_ids)]
            multiplier = CONTEXT_MULTIPLIERS[(index + 2 * bin_index) % len(CONTEXT_MULTIPLIERS)]
            target_short_l0 = local.uniform(*limits)
            if frame not in source_cache:
                source_cache[frame] = image(frame)
            patch, inner = context_crop(source_cache[frame], annotation["bbox"], multiplier)
            inner_short = min(inner[2] - inner[0], inner[3] - inner[1])
            scale = target_short_l0 * 4.0 / inner_short
            out_w, out_h = max(2, round(patch.shape[1] * scale)), max(2, round(patch.shape[0] * scale))
            patch = cv2.resize(patch, (out_w, out_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            patch = photometric(patch, local)
            ix1, iy1, ix2, iy2 = (value * scale for value in inner)
            max_x = SOURCE_SIZE[0] - out_w
            max_y = SOURCE_SIZE[1] - out_h
            if max_x < 0 or max_y < 0:
                raise RuntimeError(f"Patch exceeds source canvas: {out_w}x{out_h}")
            px = local.uniform(0, max_x)
            py = local.uniform(0, max_y)
            x, y = int(math.floor(px)), int(math.floor(py))
            dx, dy = px - x, py - y
            if dx or dy:
                patch = cv2.warpAffine(patch, np.float32([[1, 0, dx], [0, 1, dy]]), (out_w, out_h),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            background = hosted_texture(background_id, local)
            alpha = feather_mask(out_h, out_w)
            region = background[y:y+out_h, x:x+out_w].astype(np.float32)
            background[y:y+out_h, x:x+out_w] = np.uint8(np.clip(alpha * patch + (1-alpha) * region, 0, 255))
            rendered = cv2.resize(background, L0_SIZE, interpolation=cv2.INTER_AREA)
            box = ((x + dx + ix1) / 4, (y + dy + iy1) / 4, (x + dx + ix2) / 4, (y + dy + iy2) / 4)
            name = f"synthetic_{size_bin}_{index:04d}"
            write_example(split, name, rendered, [box])
            records.append({"split": split, "kind": "synthetic", "name": name, "size_bin": size_bin,
                            "target_short_side_l0": min(box[2]-box[0], box[3]-box[1]), "target_frame": frame,
                            "source_class": annotation["object_id"], "background_family": "hosted_texture",
                            "background_frame": background_id, "context_multiplier": multiplier,
                            "subpixel_offset_source": [round(dx, 6), round(dy, 6)], "target_count": 1})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-per-bin", type=int, default=30)
    parser.add_argument("--eval-per-bin", type=int, default=30)
    args = parser.parse_args()
    if OUT.exists():
        # Only remove known generated files under the experiment-owned directory.
        for path in sorted(OUT.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    records: list[dict] = []
    build_native(records)
    build_synthetic("train", args.train_per_bin, TRAIN_TARGET_FRAMES, TRAIN_BACKGROUND_IDS, records)
    build_synthetic("synthetic_eval", args.eval_per_bin, EVAL_TARGET_FRAMES, EVAL_BACKGROUND_IDS, records)
    dataset_yaml = OUT / "dataset.yaml"
    dataset_yaml.write_text(f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/synthetic_eval\nnames:\n  0: target\n", encoding="utf-8")
    counts = Counter((r["split"], r["kind"]) for r in records)
    manifest = {
        "schema_version": 1, "seed": SEED, "source_size": list(SOURCE_SIZE), "l0_size": list(L0_SIZE),
        "l0_render": "cv2.INTER_AREA after all source-resolution compositing", "class_mapping": {"0": "target"},
        "helsinki_target_view_split": {"train_frames": list(TRAIN_TARGET_FRAMES), "eval_frames": list(EVAL_TARGET_FRAMES),
            "caveat": "Views are held apart, but they depict the same physical instances; this is not instance generalization."},
        "hosted_background_split": {"train_frame_ids": list(TRAIN_BACKGROUND_IDS), "eval_frame_ids": list(EVAL_BACKGROUND_IDS),
            "policy": "contiguous non-overlapping time blocks of the 5-frame sampled capture"},
        "hosted_usage": "low-pass scenery texture only; never an empty/target-free label; every hosted-derived sample has an inserted labelled target",
        "compositing": {"patch": "rectangular target/context crop", "context_multipliers": list(CONTEXT_MULTIPLIERS),
            "edge": "raised-cosine feather up to 24 source pixels", "placement": "uniform source-resolution fractional position",
            "photometric": {"contrast": [0.65,1.45], "brightness_offset": [-45,45], "saturation": [0.55,1.5],
                "hue_opencv_units": [-12,12], "value": [0.75,1.25], "temperature": [-0.18,0.18],
                "gaussian_blur_sigma": [0.25,1.4], "mild_unsharp": True, "jpeg_quality": [35,92]}},
        "target_size_bins_l0": {key: list(value) for key,value in SIZE_BINS.items()},
        "counts": {f"{a}:{b}": n for (a,b),n in sorted(counts.items())}, "records": records,
        "dataset_yaml": str(dataset_yaml.resolve()),
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": manifest["counts"], "dataset": str(dataset_yaml)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
