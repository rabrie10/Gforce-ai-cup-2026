"""The cached multi-view reference gallery and the crop preprocessing it shares.

Probe A produced the single most important negative result of this build: on the
supplied flight, a crop with the **object erased** still classifies at 0.92-0.95
top-1 under every representation tested. Objects are static on the ground and
only the camera moves, so each object keeps the same patch of terrain underneath
it for its whole life, and any appearance matcher can answer from the ground
instead of the object. That is a mechanism for exactly the catastrophic transfer
failure Baseline 1 measured - the terrain does not come with us to a new flight.

Two preprocessing decisions follow directly, and they are the reason this module
exists rather than a two-line crop helper:

* **Tight crops.** Context is trimmed to a little over the box so terrain is a
  minority of the pixels.
* **Background normalization.** The median colour of the crop's border ring is
  subtracted before embedding, so the representation describes the object's
  contrast structure rather than the absolute colour of the ground it sits on.
  A green field and a brown field produce the same normalized object.

Prototypes are built offline with rotation, flip and photometric augmentation,
because the reference flight fixes each asset's orientation relative to the
camera and a different flight will not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from dtos import OBJECT_CLASSES


logger = logging.getLogger(__name__)

GALLERY_VERSION = 3
BACKGROUND_CLASS = -1
NEUTRAL = 128.0


# --------------------------------------------------------------------------- #
# Shared crop preprocessing - identical at build time and at query time
# --------------------------------------------------------------------------- #

def border_median(crop: np.ndarray, ring: int = 3) -> np.ndarray:
    """Median colour of the crop's outer ring, used as the local background."""
    if crop.shape[0] <= 2 * ring or crop.shape[1] <= 2 * ring:
        return np.median(crop.reshape(-1, crop.shape[-1]), axis=0)
    top = crop[:ring].reshape(-1, crop.shape[-1])
    bottom = crop[-ring:].reshape(-1, crop.shape[-1])
    left = crop[:, :ring].reshape(-1, crop.shape[-1])
    right = crop[:, -ring:].reshape(-1, crop.shape[-1])
    return np.median(np.concatenate([top, bottom, left, right]), axis=0)


def normalize_background(crop: np.ndarray) -> np.ndarray:
    """Re-centre a crop on its own local background colour.

    The output is still an 8-bit BGR image so it can go straight into any
    embedder, but its terrain is now mid-grey whatever the terrain actually was.
    """
    background = border_median(crop).astype(np.float32)
    shifted = crop.astype(np.float32) - background + NEUTRAL
    return np.clip(shifted, 0, 255).astype(np.uint8)


def prepare_crop(crop: np.ndarray, background_normalize: bool = True) -> np.ndarray:
    return normalize_background(crop) if background_normalize else crop


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #

def augment(crop: np.ndarray, rotations: int = 12, photometric: bool = True) -> List[np.ndarray]:
    """Rotations, a mirror and mild photometric jitter.

    The reference flight approaches every asset from one heading, so without
    rotation augmentation the gallery encodes "this asset seen from the north"
    rather than "this asset". A different sequence will not oblige.
    """
    variants: List[np.ndarray] = []
    size = crop.shape[0]
    centre = (size / 2.0 - 0.5, size / 2.0 - 0.5)
    for index in range(max(1, rotations)):
        angle = 360.0 * index / max(1, rotations)
        if angle == 0.0:
            rotated = crop
        else:
            matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
            rotated = cv2.warpAffine(
                crop, matrix, (size, size),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
        variants.append(rotated)
        variants.append(cv2.flip(rotated, 1))

    if not photometric:
        return variants

    jittered: List[np.ndarray] = []
    for variant in variants:
        jittered.append(variant)
    for variant in variants[::4]:
        for alpha, beta in ((0.82, -10.0), (1.20, 12.0)):
            jittered.append(
                cv2.convertScaleAbs(variant, alpha=alpha, beta=beta)
            )
    return jittered


# --------------------------------------------------------------------------- #
# Prototype reduction
# --------------------------------------------------------------------------- #

def spherical_kmeans(
    features: np.ndarray, clusters: int, iterations: int = 25, seed: int = 20260919
) -> np.ndarray:
    """Cluster L2-normalized rows on the unit sphere, deterministically.

    Prototypes rather than raw exemplars keep the query matmul inside the CPU
    budget: a gallery augmented twenty-fold would otherwise dominate the frame.
    """
    if len(features) <= clusters:
        return features.copy()
    generator = np.random.default_rng(seed)
    # k-means++ style seeding on cosine distance.
    centres = [features[generator.integers(len(features))]]
    for _ in range(clusters - 1):
        similarity = features @ np.stack(centres).T
        # Floating-point cosine can sit a hair above 1.0, which would make the
        # seeding distribution negative and numpy reject it outright.
        distance = np.clip(1.0 - similarity.max(axis=1), 0.0, None)
        total = float(distance.sum())
        if total <= 1e-9:
            centres.append(features[generator.integers(len(features))])
            continue
        centres.append(features[int(generator.choice(len(features), p=distance / total))])
    centroids = np.stack(centres)

    for _ in range(iterations):
        assignment = np.argmax(features @ centroids.T, axis=1)
        updated = np.zeros_like(centroids)
        for index in range(clusters):
            members = features[assignment == index]
            updated[index] = members.mean(axis=0) if len(members) else centroids[index]
        norms = np.linalg.norm(updated, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        updated = updated / norms
        if np.allclose(updated, centroids, atol=1e-6):
            centroids = updated
            break
        centroids = updated
    return centroids


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #

@dataclass
class ReferenceGallery:
    """Class prototypes per resolution level, ready for cosine scoring."""

    features: np.ndarray            # (N, D) float32, L2-normalized
    # Index into OBJECT_CLASSES, or BACKGROUND_CLASS (-1) for a terrain
    # prototype. Without the reject option every grass blob a proposal lands on
    # is confidently named something, which measured 2.4% track precision.
    class_index: np.ndarray
    level: np.ndarray               # (N,) int8
    backbone: str
    crop_size: int
    context: float
    background_normalize: bool
    version: int = GALLERY_VERSION

    @property
    def dimension(self) -> int:
        return int(self.features.shape[1])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features.astype(np.float32),
            class_index=self.class_index.astype(np.int32),
            level=self.level.astype(np.int8),
            backbone=np.array(self.backbone),
            crop_size=np.array(self.crop_size),
            context=np.array(self.context),
            background_normalize=np.array(self.background_normalize),
            version=np.array(self.version),
            classes=np.array(list(OBJECT_CLASSES)),
        )

    @classmethod
    def load(cls, path: str | Path) -> 'ReferenceGallery':
        data = np.load(Path(path), allow_pickle=False)
        stored = [str(name) for name in data['classes']]
        if tuple(stored) != OBJECT_CLASSES:
            raise ValueError('Gallery class order does not match OBJECT_CLASSES')
        return cls(
            features=data['features'].astype(np.float32),
            class_index=data['class_index'].astype(np.int32),
            level=data['level'].astype(np.int8),
            backbone=str(data['backbone']),
            crop_size=int(data['crop_size']),
            context=float(data['context']),
            background_normalize=bool(data['background_normalize']),
            version=int(data['version']),
        )

    def describe(self) -> dict:
        counts = {
            OBJECT_CLASSES[index]: int((self.class_index == index).sum())
            for index in range(len(OBJECT_CLASSES))
        }
        background = int((self.class_index == BACKGROUND_CLASS).sum())
        return {
            'version': self.version,
            'backbone': self.backbone,
            'dimension': self.dimension,
            'prototypes': int(len(self.features)),
            'crop_size': self.crop_size,
            'context': self.context,
            'background_normalize': self.background_normalize,
            'per_level': {
                str(level): int((self.level == level).sum()) for level in (0, 1, 2)
            },
            'per_class': counts,
            'background_prototypes': background,
            'classes_covered': sum(1 for value in counts.values() if value > 0),
        }
