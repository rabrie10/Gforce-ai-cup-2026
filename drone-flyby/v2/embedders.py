"""Interchangeable crop representations.

All of these consume the *same* crop array and return L2-normalized rows, so
the identical-crop representation probe can swap one for another without any
other difference in the comparison, and the pipeline can adopt whichever wins
by changing one environment variable.

The frozen-backbone entries exist because Baseline 1 established that a 16-way
head fine-tuned on one physical instance per class does not transfer. A
representation nobody fitted to this scene cannot memorize it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent

# ImageNet statistics, in RGB order.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return matrix / norms


class Embedder:
    """Maps a batch of BGR crops to L2-normalized feature rows."""

    name = 'base'
    dimension = 0

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def warmup(self) -> None:
        dummy = np.zeros((32, 32, 3), dtype=np.uint8)
        self.embed([dummy, dummy])


class PixelEmbedder(Embedder):
    """Downsampled grey pixels. The control every learned feature must beat."""

    name = 'pixels'

    def __init__(self, side: int = 16) -> None:
        self.side = side
        self.dimension = side * side

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        rows = np.empty((len(crops), self.dimension), dtype=np.float32)
        for index, crop in enumerate(crops):
            grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(grey, (self.side, self.side), interpolation=cv2.INTER_AREA)
            vector = small.astype(np.float32).reshape(-1)
            rows[index] = vector - vector.mean()
        return _l2_normalize(rows)


class HandcraftedEmbedder(Embedder):
    """Colour histogram plus oriented-gradient histogram.

    Cheap, deterministic and completely free of learned priors. It is here to
    show how much of any frozen-backbone advantage is really just "it sees
    colour and edges".
    """

    name = 'handcrafted'

    def __init__(self, colour_bins: int = 8, orientation_bins: int = 9, cells: int = 4) -> None:
        self.colour_bins = colour_bins
        self.orientation_bins = orientation_bins
        self.cells = cells
        self.dimension = colour_bins * 3 + orientation_bins * cells * cells

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        rows = np.empty((len(crops), self.dimension), dtype=np.float32)
        for index, crop in enumerate(crops):
            parts: List[np.ndarray] = []
            for channel in range(3):
                histogram = cv2.calcHist(
                    [crop], [channel], None, [self.colour_bins], [0, 256]
                ).reshape(-1)
                parts.append(histogram / max(1.0, histogram.sum()))

            grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = np.hypot(gx, gy)
            angle = (np.arctan2(gy, gx) % np.pi) / np.pi * self.orientation_bins
            bin_index = np.clip(angle.astype(np.int32), 0, self.orientation_bins - 1)

            height, width = grey.shape
            cell_h = max(1, height // self.cells)
            cell_w = max(1, width // self.cells)
            for row in range(self.cells):
                for column in range(self.cells):
                    y0, y1 = row * cell_h, min(height, (row + 1) * cell_h)
                    x0, x1 = column * cell_w, min(width, (column + 1) * cell_w)
                    cell_bins = bin_index[y0:y1, x0:x1].reshape(-1)
                    cell_magnitude = magnitude[y0:y1, x0:x1].reshape(-1)
                    histogram = np.bincount(
                        cell_bins, weights=cell_magnitude, minlength=self.orientation_bins
                    ).astype(np.float32)
                    parts.append(histogram / max(1e-6, float(histogram.sum())))
            rows[index] = np.concatenate(parts)
        return _l2_normalize(rows)


class TorchvisionEmbedder(Embedder):
    """A frozen ImageNet backbone, global-pooled, in evaluation mode.

    Nothing here is trained on Helsinki. That is the point: the features carry
    generic shape and texture statistics, so identity comes from similarity to
    a reference gallery rather than from a decision boundary fitted to the one
    instance of each class this project happens to own.
    """

    def __init__(self, architecture: str = 'resnet18', threads: int = 4) -> None:
        import torch
        import torchvision

        torch.set_num_threads(int(threads))
        self.architecture = architecture
        self.name = architecture
        self._torch = torch

        builders = {
            'resnet18': (torchvision.models.resnet18, 'ResNet18_Weights', 512),
            'resnet34': (torchvision.models.resnet34, 'ResNet34_Weights', 512),
            'resnet50': (torchvision.models.resnet50, 'ResNet50_Weights', 2048),
            'mobilenet_v3_small': (
                torchvision.models.mobilenet_v3_small,
                'MobileNet_V3_Small_Weights',
                576,
            ),
            'shufflenet_v2_x0_5': (
                torchvision.models.shufflenet_v2_x0_5,
                'ShuffleNet_V2_X0_5_Weights',
                1024,
            ),
        }
        if architecture not in builders:
            raise ValueError(f'Unsupported backbone {architecture!r}')
        builder, weights_enum, dimension = builders[architecture]
        weights = getattr(torchvision.models, weights_enum).DEFAULT
        model = builder(weights=weights)

        if architecture.startswith(('resnet', 'shufflenet')):
            # Everything up to the classifier; the global pool stays.
            modules = list(model.children())[:-1]
            self.model = torch.nn.Sequential(*modules)
        else:
            self.model = torch.nn.Sequential(
                model.features, torch.nn.AdaptiveAvgPool2d(1)
            )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.dimension = dimension

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dimension), dtype=np.float32)
        torch = self._torch
        batch = np.stack(
            [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops]
        ).astype(np.float32) / 255.0
        batch = (batch - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2)))
        with torch.inference_mode():
            features = self.model(tensor)
        rows = features.reshape(features.shape[0], -1).numpy().astype(np.float32)
        return _l2_normalize(rows)


class YoloHeadEmbedder(Embedder):
    """The supervised Baseline 1 head, scored on the identical crop.

    Present only so the representation comparison is fair: the question is
    whether the supervised branch carries signal the frozen one does not, on
    exactly the same pixels. It is not a candidate for the default path.
    """

    name = 'yolo_head'

    def __init__(self, weights: Optional[str] = None, imgsz: int = 160) -> None:
        from ultralytics import YOLO

        from dtos import OBJECT_CLASSES

        path = Path(weights or (ROOT / 'models' / 'drone_yolo11n_l0.pt'))
        self.model = YOLO(str(path))
        self.imgsz = imgsz
        self.classes = OBJECT_CLASSES
        self.dimension = len(OBJECT_CLASSES)

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        rows = np.zeros((len(crops), self.dimension), dtype=np.float32)
        if not crops:
            return rows
        results = self.model.predict(
            source=[
                cv2.resize(crop, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
                for crop in crops
            ],
            imgsz=self.imgsz,
            conf=0.001,
            iou=0.7,
            device='cpu',
            verbose=False,
        )
        for index, result in enumerate(results):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            confidences = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for class_index, confidence in zip(classes, confidences):
                if 0 <= class_index < self.dimension:
                    rows[index, class_index] = max(rows[index, class_index], float(confidence))
        # Rows with no detection stay all-zero; normalizing them would invent a
        # direction, so they are left as the honest "no evidence" vector.
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        return rows / norms


def build_embedder(name: str, threads: int = 4) -> Embedder:
    """Resolve a representation name to an instance."""
    if name == 'pixels':
        return PixelEmbedder()
    if name == 'handcrafted':
        return HandcraftedEmbedder()
    if name == 'yolo_head':
        return YoloHeadEmbedder()
    return TorchvisionEmbedder(name, threads=threads)
