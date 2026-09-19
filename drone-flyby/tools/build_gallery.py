"""Build the cached reference gallery from the supplied Helsinki frames.

Run once, offline, before deployment. The product is a small ``.npz`` of class
prototypes that the endpoint memory-maps at startup; nothing here runs in the
request path.

The crops are produced through **exactly** the query-time path - render the
960x540 view the evaluator would transmit, then cut the object out of that view
with the same context and the same background normalization - so a reference
carries the same detail loss and the same colour treatment a live query does.
Building references from the 4K source directly would quietly give the gallery
information the camera never transmits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtos import OBJECT_CLASSES  # noqa: E402
from utils import frame_numbers, load_annotations, load_frame  # noqa: E402
from v2.embedders import build_embedder  # noqa: E402
from v2.config import ProposalConfig  # noqa: E402
from v2.gallery import (  # noqa: E402
    BACKGROUND_CLASS,
    ReferenceGallery,
    augment,
    prepare_crop,
    spherical_kmeans,
)
from v2.geometry import (  # noqa: E402
    box_center,
    clamp_center,
    crop_from_view,
    render_level_view,
    source_region,
    source_to_view,
)
from v2.proposals import ProposalEngine  # noqa: E402


BACKGROUND_GRID = {0: [(1920, 1080)],
                   1: [(960, 540), (2880, 540), (960, 1620), (2880, 1620)],
                   2: [(480, 270), (3360, 270), (480, 1890), (3360, 1890)]}


def mine_background(
    scene: str,
    levels: List[int],
    crop_size: int,
    context: float,
    background_normalize: bool,
) -> List[np.ndarray]:
    """Collect crops from proposals that land on no object at all.

    Random terrain patches would teach the rejector about terrain in general.
    What it actually has to reject is *this proposal engine's own false
    positives*: the particular field edges, tree crowns and shadow boundaries
    that score highly under multi-scale local contrast. Mining them directly
    makes the reject option a discriminator against the real error
    distribution rather than against an imagined one.
    """
    engine = ProposalEngine(ProposalConfig(budget=60))
    crops: List[np.ndarray] = []
    for frame in frame_numbers(scene):
        image = load_frame(frame, scene)
        boxes = [tuple(float(v) for v in row['bbox'])
                 for row in load_annotations(frame, scene)]
        for level in levels:
            for camera in BACKGROUND_GRID.get(level, []):
                view = render_level_view(image, level, *camera)
                region = source_region(level, *camera)
                local_boxes = [source_to_view(box, region) for box in boxes]
                for proposal in engine.propose(view, budget=60):
                    # Generous exclusion: anything even loosely near a real
                    # object is left out, so no prototype is secretly an object.
                    if any(
                        _overlaps_generously(proposal.box, local)
                        for local in local_boxes
                    ):
                        continue
                    crop = crop_from_view(view, proposal.box, crop_size, context)
                    if crop is not None:
                        crops.append(prepare_crop(crop, background_normalize))
    return crops


def _overlaps_generously(proposal, target) -> bool:
    pad = 0.75 * max(target[2] - target[0], target[3] - target[1]) + 8.0
    return not (
        proposal[2] < target[0] - pad
        or proposal[0] > target[2] + pad
        or proposal[3] < target[1] - pad
        or proposal[1] > target[3] + pad
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='helsinki')
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--crop-size', type=int, default=64)
    parser.add_argument('--context', type=float, default=1.25)
    parser.add_argument('--levels', default='0,1,2')
    parser.add_argument('--rotations', type=int, default=12)
    parser.add_argument('--prototypes', type=int, default=24,
                        help='prototypes kept per class per level')
    parser.add_argument('--background-prototypes', type=int, default=160,
                        help='terrain prototypes forming the reject option')
    parser.add_argument('--no-background-normalize', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', default=str(ROOT / 'models' / 'reference_gallery.npz'))
    arguments = parser.parse_args()

    levels = [int(value) for value in arguments.levels.split(',')]
    background_normalize = not arguments.no_background_normalize
    embedder = build_embedder(arguments.backbone, threads=arguments.threads)

    started = time.perf_counter()
    buckets: Dict[tuple, List[np.ndarray]] = defaultdict(list)
    appearances = 0

    for frame in frame_numbers(arguments.scene):
        image = load_frame(frame, arguments.scene)
        annotations = load_annotations(frame, arguments.scene)
        for level in levels:
            view_cache: Dict[tuple, np.ndarray] = {}
            for annotation in annotations:
                box = tuple(float(value) for value in annotation['bbox'])
                centre = box_center(box)
                camera = clamp_center(level, centre[0], centre[1])
                if camera not in view_cache:
                    view_cache[camera] = render_level_view(image, level, *camera)
                view = view_cache[camera]
                region = source_region(level, *camera)
                local = source_to_view(box, region)
                crop = crop_from_view(view, local, arguments.crop_size, arguments.context)
                if crop is None:
                    continue
                base = prepare_crop(crop, background_normalize)
                variants = augment(base, rotations=arguments.rotations)
                buckets[(annotation['object_id'], level)].extend(variants)
                appearances += 1

    print(f'Collected {appearances} appearance renders in '
          f'{time.perf_counter() - started:.1f}s')

    background_crops: List[np.ndarray] = []
    if arguments.background_prototypes > 0:
        background_crops = mine_background(
            arguments.scene, levels, arguments.crop_size, arguments.context,
            background_normalize,
        )
        print(f'Mined {len(background_crops)} background crops from the proposal '
              f"engine's own false positives")

    print('Embedding ...')

    features: List[np.ndarray] = []
    class_indices: List[int] = []
    level_values: List[int] = []
    for (object_id, level), crops in sorted(buckets.items()):
        embedded = np.concatenate(
            [embedder.embed(crops[i:i + 64]) for i in range(0, len(crops), 64)]
        )
        prototypes = spherical_kmeans(embedded, arguments.prototypes)
        features.append(prototypes.astype(np.float32))
        class_indices.extend([OBJECT_CLASSES.index(object_id)] * len(prototypes))
        level_values.extend([level] * len(prototypes))
        print(f'  {object_id:16s} L{level}  {len(crops):5d} crops -> {len(prototypes)} prototypes')

    if background_crops:
        embedded = np.concatenate(
            [embedder.embed(background_crops[i:i + 64])
             for i in range(0, len(background_crops), 64)]
        )
        prototypes = spherical_kmeans(embedded, arguments.background_prototypes)
        features.append(prototypes.astype(np.float32))
        class_indices.extend([BACKGROUND_CLASS] * len(prototypes))
        # Background is level-agnostic: terrain looks like terrain at every
        # level, and the query-side level mask must never exclude the reject
        # option or every crop becomes an object again.
        level_values.extend([0] * len(prototypes))
        print(f'  {"background":16s} --  {len(background_crops):5d} crops -> '
              f'{len(prototypes)} prototypes')

    gallery = ReferenceGallery(
        features=np.concatenate(features),
        class_index=np.asarray(class_indices, dtype=np.int32),
        level=np.asarray(level_values, dtype=np.int8),
        backbone=arguments.backbone,
        crop_size=arguments.crop_size,
        context=arguments.context,
        background_normalize=background_normalize,
    )
    gallery.save(arguments.output)
    description = gallery.describe()
    print(json.dumps(description, indent=2))
    print(f'Wrote {arguments.output} '
          f'({Path(arguments.output).stat().st_size / 1e6:.2f} MB) in '
          f'{time.perf_counter() - started:.1f}s')
    if description['classes_covered'] != len(OBJECT_CLASSES):
        print('WARNING: not every class has prototypes', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
