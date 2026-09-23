"""Transfer-oriented crop identity: frozen features against a cached gallery.

Baseline 1's 16-way supervised head reached 0.3837 mAP on the flight it was
fitted to and 0.0016 hosted. Nothing here is fitted to that flight. Identity is
a similarity to cached prototypes, computed from a frozen ImageNet backbone on
background-normalized, object-centred crops, and it returns a **posterior**
rather than a label so the track bank can accumulate and, when better evidence
arrives, overturn it.

The whole gallery is loaded and the backbone warmed once at startup. Nothing in
this module touches the network, and everything in the query path is one batched
forward pass plus one matrix multiply.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from dtos import OBJECT_CLASSES
from v2.config import CONFIG, RecognizerConfig
from v2.embedders import Embedder, build_embedder
from v2.gallery import BACKGROUND_CLASS, ReferenceGallery, prepare_crop


logger = logging.getLogger(__name__)

NUM_CLASSES = len(OBJECT_CLASSES)
UNIFORM = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES, dtype=np.float32)


@dataclass
class RecognitionResult:
    posterior: np.ndarray
    top_similarity: float
    margin: float
    # How much more this crop looks like some object than like terrain. Without
    # it the softmax names every grass blob a proposal lands on, which is what
    # drove track precision to 2.4% before background prototypes existed.
    objectness: float = 1.0
    background_similarity: float = 0.0

    @property
    def best_index(self) -> int:
        return int(np.argmax(self.posterior))

    @property
    def best_class(self) -> str:
        return OBJECT_CLASSES[self.best_index]


class CropRecognizer:
    """Batched prototype matching over object-centred crops."""

    def __init__(self, config: Optional[RecognizerConfig] = None) -> None:
        self.config = config or CONFIG.recognizer
        self.gallery: Optional[ReferenceGallery] = None
        self.embedder: Optional[Embedder] = None
        self.available = False
        self.has_background = False
        self.load_error: Optional[str] = None
        self._level_masks: dict = {}

    # -- lifecycle ---------------------------------------------------------- #

    def load(self) -> None:
        if not self.config.enabled:
            self.load_error = 'disabled'
            return
        path = Path(self.config.gallery_path)
        if not path.is_file():
            self.load_error = f'gallery missing: {path}'
            logger.warning(
                'Reference gallery not found at %s; identity falls back to uniform', path
            )
            return
        try:
            gallery = ReferenceGallery.load(path)
            embedder = build_embedder(gallery.backbone, threads=self.config.torch_threads)
            if embedder.dimension != gallery.dimension:
                raise ValueError(
                    f'Gallery dimension {gallery.dimension} does not match '
                    f'{gallery.backbone} output {embedder.dimension}'
                )
        except Exception as exc:
            self.load_error = f'{type(exc).__name__}: {exc}'
            logger.exception('Reference gallery could not be loaded')
            return

        self.gallery = gallery
        self.embedder = embedder
        # Prototypes observed at a level at least as detailed as the query are
        # the fair comparison; a query made at L0 should not be judged against
        # native-resolution references alone.
        for level in (0, 1, 2):
            self._level_masks[level] = np.where(gallery.level <= level)[0]
            if len(self._level_masks[level]) == 0:
                self._level_masks[level] = np.arange(len(gallery.features))
        self.has_background = bool((gallery.class_index == BACKGROUND_CLASS).any())
        if not self.has_background:
            logger.warning(
                'Gallery carries no background prototypes; every proposal will be '
                'named an object. Rebuild with tools/build_gallery.py.'
            )
        self.available = True
        self.load_error = None
        logger.info('Reference gallery loaded: %s', gallery.describe())

    def warmup(self) -> None:
        if not self.available or self.embedder is None:
            return
        dummy = np.full((self.crop_size, self.crop_size, 3), 128, dtype=np.uint8)
        self.recognize([dummy] * 4, level=1)

    @property
    def crop_size(self) -> int:
        """Query crops must match the gallery's geometry, not the config's.

        A gallery built at one context and queried at another compares
        different framings of the same object and silently degrades.
        """
        return self.gallery.crop_size if self.gallery is not None else self.config.crop_size

    @property
    def crop_context(self) -> float:
        return self.gallery.context if self.gallery is not None else self.config.context

    def describe(self) -> dict:
        if not self.available or self.gallery is None:
            return {'available': False, 'error': self.load_error}
        return {'available': True, **self.gallery.describe()}

    # -- query -------------------------------------------------------------- #

    def recognize(
        self, crops: Sequence[np.ndarray], level: int
    ) -> Tuple[List[RecognitionResult], float]:
        """Score a batch of crops. Never raises into the request path."""
        started = time.perf_counter()
        if not crops:
            return [], 0.0
        if not self.available or self.embedder is None or self.gallery is None:
            return (
                [RecognitionResult(UNIFORM.copy(), 0.0, 0.0) for _ in crops],
                (time.perf_counter() - started) * 1000.0,
            )

        try:
            prepared = [
                prepare_crop(crop, self.gallery.background_normalize) for crop in crops
            ]
            features = self._embed(prepared)
            results = self._score(features, level)
        except Exception:
            logger.exception('Recognizer failed; returning uniform posteriors')
            results = [RecognitionResult(UNIFORM.copy(), 0.0, 0.0) for _ in crops]
        return results, (time.perf_counter() - started) * 1000.0

    def _embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        batch = max(1, self.config.batch)
        blocks = [
            self.embedder.embed(crops[start:start + batch])
            for start in range(0, len(crops), batch)
        ]
        return np.concatenate(blocks) if blocks else np.zeros((0, 1), dtype=np.float32)

    def _score(self, features: np.ndarray, level: int) -> List[RecognitionResult]:
        gallery = self.gallery
        indices = self._level_masks.get(level, np.arange(len(gallery.features)))
        similarity = features @ gallery.features[indices].T
        class_index = gallery.class_index[indices]

        top_k = max(1, self.config.top_k)

        def pooled(columns: np.ndarray) -> np.ndarray:
            """Mean of the k best-matching prototypes in a column group.

            One lucky prototype should not decide an identity, and a group with
            more prototypes should not win on count alone.
            """
            block = similarity[:, columns]
            k = min(top_k, block.shape[1])
            return np.partition(block, -k, axis=1)[:, -k:].mean(axis=1)

        scores = np.full((len(features), NUM_CLASSES), -1.0, dtype=np.float32)
        for target in range(NUM_CLASSES):
            columns = np.where(class_index == target)[0]
            if len(columns) == 0:
                continue
            scores[:, target] = pooled(columns)

        background_columns = np.where(class_index == BACKGROUND_CLASS)[0]
        if len(background_columns):
            background = pooled(background_columns)
        else:
            background = np.full(len(features), -1.0, dtype=np.float32)

        results: List[RecognitionResult] = []
        temperature = max(1e-3, self.config.temperature)
        objectness_temperature = max(1e-3, self.config.objectness_temperature)
        for index, row in enumerate(scores):
            present = row > -1.0
            if not present.any():
                results.append(RecognitionResult(UNIFORM.copy(), 0.0, 0.0, 0.0, 0.0))
                continue
            logits = np.where(present, row / temperature, -np.inf)
            logits = logits - np.max(logits[present])
            weights = np.where(present, np.exp(logits), 0.0)
            total = weights.sum()
            posterior = (
                (weights / total).astype(np.float32) if total > 0 else UNIFORM.copy()
            )
            ordered = np.sort(row[present])[::-1]
            margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0])
            background_similarity = float(background[index])
            if len(background_columns):
                gap = (float(ordered[0]) - background_similarity) / objectness_temperature
                objectness = float(1.0 / (1.0 + np.exp(-np.clip(gap, -30.0, 30.0))))
            else:
                objectness = 1.0
            results.append(
                RecognitionResult(
                    posterior=posterior,
                    top_similarity=float(ordered[0]),
                    margin=margin,
                    objectness=objectness,
                    background_similarity=background_similarity,
                )
            )
        return results
