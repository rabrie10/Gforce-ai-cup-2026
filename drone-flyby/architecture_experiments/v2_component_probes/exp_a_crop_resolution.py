"""Probe A+B - oracle crop recognition at true L0/L1/L2, across representations.

Question A: does true camera resolution materially restore identity signal?
Question B: on the *identical* crops, which representation carries that signal?

Protocol
--------
Ground-truth boxes are used only to place the crop, never to answer it. For
every GT appearance and every level, the crop is rendered through the exact
transmission path the evaluator would use: source region -> INTER_AREA
downsample by the level factor -> resize to the recognizer input. Upsampling
afterwards cannot recreate detail the level never carried, which is the whole
point of the comparison.

Identity is decided by nearest-exemplar cosine similarity against a gallery of
other appearances, with a temporal exclusion band: for a query at frame f, no
appearance within ``--gap`` frames of f may act as a reference.

The three crop variants
-----------------------
The first run of this probe returned 0.927 top-1 at L0 from raw 16x16 grey
pixels, on objects two to fifteen pixels across. That is not identity signal.
The objects are static on the ground and only the camera moves, so every crop of
a given object contains the *same patch of terrain*, and a nearest-neighbour
classifier can answer from the background alone. The probe therefore renders
each appearance three ways:

``full``
    Object plus surrounding context, as a live recognizer would see it.
``object_only``
    Everything outside the GT box replaced by the crop's border mean. This is
    the closest available measurement of identity carried by object pixels.
``background_only``
    The GT box replaced by the same border mean. The object is gone, so any
    accuracy here is pure context memorization and is the control that says how
    much of ``full`` to believe.

``object_only`` is the number that answers question A. ``background_only`` is
the number that says whether ``full`` means anything at all.

Honesty boundary
----------------
The supplied scene holds exactly one physical instance per class, so even
``object_only`` measures robustness to viewpoint, scale and detail loss *within
one flight*. It is NOT cross-instance or cross-sequence generalization
evidence, and no absolute accuracy here may be quoted as such. What the probe
supports is the relative comparison between L0, L1 and L2 under an identical
protocol, and between representations on identical pixels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtos import OBJECT_CLASSES  # noqa: E402
from utils import frame_numbers, load_annotations, load_frame  # noqa: E402
from v2.embedders import build_embedder  # noqa: E402
from v2.geometry import LEVEL_DOWNSCALE, box_short_side, expand_box  # noqa: E402


LEVELS = (0, 1, 2)
VARIANTS = ('full', 'object_only', 'background_only')
SIZE_EDGES = (0.0, 4.0, 8.0, 16.0, 32.0, float('inf'))
SIZE_LABELS = ('<4', '4-<8', '8-<16', '16-<32', '>=32')


def size_bucket(short_side: float) -> str:
    for index in range(len(SIZE_LABELS)):
        if SIZE_EDGES[index] <= short_side < SIZE_EDGES[index + 1]:
            return SIZE_LABELS[index]
    return SIZE_LABELS[-1]


def render_with_object_rect(
    frame: np.ndarray,
    box: Tuple[float, float, float, float],
    level: int,
    crop_size: int,
    context: float,
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Render one appearance at ``level`` and report where the object landed.

    Returns the crop plus the object's rectangle inside it, so the masked
    variants can be built from exactly the same pixels.
    """
    downscale = LEVEL_DOWNSCALE[level]
    region = expand_box(box, context, minimum_side=downscale * 2.0)
    x1, y1 = max(0, int(round(region[0]))), max(0, int(round(region[1])))
    x2 = min(frame.shape[1], int(round(region[2])))
    y2 = min(frame.shape[0], int(round(region[3])))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    patch = frame[y1:y2, x1:x2]
    if downscale > 1.0:
        target = (
            max(1, int(round(patch.shape[1] / downscale))),
            max(1, int(round(patch.shape[0] / downscale))),
        )
        patch = cv2.resize(patch, target, interpolation=cv2.INTER_AREA)
    interpolation = cv2.INTER_AREA if patch.shape[0] > crop_size else cv2.INTER_LINEAR
    crop = cv2.resize(patch, (crop_size, crop_size), interpolation=interpolation)

    scale_x = crop_size / float(x2 - x1)
    scale_y = crop_size / float(y2 - y1)
    rect = (
        int(np.clip(round((box[0] - x1) * scale_x), 0, crop_size - 1)),
        int(np.clip(round((box[1] - y1) * scale_y), 0, crop_size - 1)),
        int(np.clip(round((box[2] - x1) * scale_x), 1, crop_size)),
        int(np.clip(round((box[3] - y1) * scale_y), 1, crop_size)),
    )
    return crop, rect


def border_fill(crop: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    """Mean colour of the pixels outside the object rectangle."""
    mask = np.ones(crop.shape[:2], dtype=bool)
    mask[rect[1]:rect[3], rect[0]:rect[2]] = False
    if not mask.any():
        return np.array(crop.reshape(-1, 3).mean(axis=0), dtype=np.float32)
    return crop[mask].mean(axis=0).astype(np.float32)


NEUTRAL_FILL = np.array([128, 128, 128], dtype=np.uint8)


def build_variants(
    crop: np.ndarray, rect: Tuple[int, int, int, int]
) -> Dict[str, np.ndarray]:
    """Full crop, object with context erased, and context with object erased.

    ``object_only`` is filled with a **constant neutral grey**, not with the
    surrounding terrain's mean colour. An earlier revision used the border mean
    and leaked the very thing the variant exists to remove: on a flight where
    every object sits on its own fixed patch of ground, that mean colour is a
    near-unique fingerprint of the object's location.
    """
    object_only = np.empty_like(crop)
    object_only[:, :] = NEUTRAL_FILL
    object_only[rect[1]:rect[3], rect[0]:rect[2]] = crop[rect[1]:rect[3], rect[0]:rect[2]]

    background_only = crop.copy()
    background_only[rect[1]:rect[3], rect[0]:rect[2]] = border_fill(crop, rect)
    return {'full': crop, 'object_only': object_only, 'background_only': background_only}


def collect_appearances(scene: str, crop_size: int, context: float) -> List[dict]:
    """Render every GT appearance at all levels, in all three variants, once."""
    appearances: List[dict] = []
    for frame in frame_numbers(scene):
        image = load_frame(frame, scene)
        for annotation in load_annotations(frame, scene):
            box = tuple(float(value) for value in annotation['bbox'])
            crops: Dict[Tuple[int, str], np.ndarray] = {}
            complete = True
            for level in LEVELS:
                rendered = render_with_object_rect(image, box, level, crop_size, context)
                if rendered is None:
                    complete = False
                    break
                variants = build_variants(*rendered)
                for name, array in variants.items():
                    crops[(level, name)] = array
            if not complete:
                continue
            appearances.append(
                {
                    'frame': frame,
                    'object_id': annotation['object_id'],
                    'bbox': box,
                    'source_short_side': box_short_side(box),
                    'crops': crops,
                }
            )
    return appearances


def evaluate(
    features: np.ndarray,
    labels: List[str],
    frames: List[int],
    gap: int,
    top_k: int,
) -> dict:
    """Leave-one-frame-out nearest-exemplar identity, with a temporal band."""
    similarity = features @ features.T
    frame_array = np.asarray(frames)
    label_array = np.asarray(labels)
    class_names = list(OBJECT_CLASSES)

    predictions: List[str] = []
    margins: List[float] = []
    eligible = np.zeros(len(labels), dtype=bool)

    for index in range(len(labels)):
        allowed = np.abs(frame_array - frame_array[index]) >= gap
        if not allowed.any():
            predictions.append('')
            margins.append(0.0)
            continue
        eligible[index] = True
        row = similarity[index]
        scores = np.full(len(class_names), -np.inf, dtype=np.float64)
        for class_index, name in enumerate(class_names):
            mask = allowed & (label_array == name)
            if not mask.any():
                continue
            values = np.sort(row[mask])[::-1][:top_k]
            scores[class_index] = float(values.mean())
        best = int(np.argmax(scores))
        ordered = np.sort(scores[np.isfinite(scores)])[::-1]
        margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0])
        predictions.append(class_names[best])
        margins.append(margin)

    correct = np.array(
        [bool(eligible[i] and predictions[i] == labels[i]) for i in range(len(labels))]
    )
    return {
        'eligible': eligible,
        'correct': correct,
        'predictions': predictions,
        'margins': np.asarray(margins),
    }


def summarize(result: dict, appearances: List[dict], level: int) -> dict:
    eligible = result['eligible']
    correct = result['correct']
    total = int(eligible.sum())
    hits = int(correct.sum())

    per_class: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    per_size: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for index, appearance in enumerate(appearances):
        if not eligible[index]:
            continue
        bucket = size_bucket(appearance['source_short_side'] / LEVEL_DOWNSCALE[level])
        per_class[appearance['object_id']][1] += 1
        per_size[bucket][1] += 1
        if correct[index]:
            per_class[appearance['object_id']][0] += 1
            per_size[bucket][0] += 1

    return {
        'level': level,
        'eligible': total,
        'correct': hits,
        'accuracy': hits / total if total else 0.0,
        'classes_with_any_correct': sum(1 for v in per_class.values() if v[0] > 0),
        'classes_present': len(per_class),
        'mean_margin': float(result['margins'][eligible].mean()) if total else 0.0,
        'per_class': {
            name: {'correct': v[0], 'n': v[1], 'accuracy': v[0] / v[1] if v[1] else 0.0}
            for name, v in sorted(per_class.items())
        },
        'per_size': {
            name: {
                'correct': per_size[name][0],
                'n': per_size[name][1],
                'accuracy': per_size[name][0] / per_size[name][1] if per_size[name][1] else 0.0,
            }
            for name in SIZE_LABELS
            if name in per_size
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='helsinki')
    parser.add_argument('--crop-size', type=int, default=64)
    parser.add_argument('--context', type=float, default=1.6)
    parser.add_argument('--gap', type=int, default=4,
                        help='minimum frame separation between query and reference')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--representations', default='pixels,resnet18,resnet50,yolo_head')
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--output', default=str(Path(__file__).parent / 'exp_a_results.json'))
    arguments = parser.parse_args()

    variants = [name.strip() for name in arguments.variants.split(',') if name.strip()]

    print('Rendering appearances at all three levels, in all variants ...')
    appearances = collect_appearances(arguments.scene, arguments.crop_size, arguments.context)
    labels = [appearance['object_id'] for appearance in appearances]
    frames = [appearance['frame'] for appearance in appearances]
    print(f'  {len(appearances)} appearances, {len(set(labels))} classes, '
          f'{len(set(frames))} frames')

    report = {
        'protocol': {
            'scene': arguments.scene,
            'crop_size': arguments.crop_size,
            'context': arguments.context,
            'temporal_gap': arguments.gap,
            'top_k': arguments.top_k,
            'appearances': len(appearances),
            'variants': variants,
            'note': (
                'One physical instance per class and objects static on the ground, so '
                'the full-crop numbers are confounded by terrain context. Read '
                'object_only against background_only; never quote any of these as '
                'cross-instance generalization evidence.'
            ),
        },
        'results': {},
    }

    for name in [n.strip() for n in arguments.representations.split(',') if n.strip()]:
        print(f'\n=== representation: {name} ===')
        try:
            embedder = build_embedder(name)
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f'  unavailable: {type(exc).__name__}: {exc}')
            report['results'][name] = {'error': f'{type(exc).__name__}: {exc}'}
            continue

        entry: Dict[str, dict] = {'dimension': embedder.dimension, 'variants': {}}
        for variant in variants:
            entry['variants'][variant] = {}
            row = []
            for level in LEVELS:
                crops = [appearance['crops'][(level, variant)] for appearance in appearances]
                started = time.perf_counter()
                features = np.concatenate(
                    [embedder.embed(crops[i:i + 64]) for i in range(0, len(crops), 64)]
                )
                embed_ms = (time.perf_counter() - started) * 1000.0
                result = evaluate(features, labels, frames, arguments.gap, arguments.top_k)
                summary = summarize(result, appearances, level)
                summary['embed_ms_per_crop'] = embed_ms / max(1, len(crops))
                entry['variants'][variant][str(level)] = summary
                row.append(
                    f'L{level} {summary["accuracy"]:.3f} '
                    f'[{summary["classes_with_any_correct"]}/{summary["classes_present"]}cls]'
                )
            print(f'  {variant:16s} ' + '   '.join(row))
        report['results'][name] = entry

    Path(arguments.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nWrote {arguments.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
