"""Fine-tune one Ultralytics nano detector on the full L0 Helsinki set."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO


TRAINING_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = TRAINING_ROOT / "generated" / "l0_helsinki" / "dataset.yaml"
DEFAULT_PROJECT = TRAINING_ROOT / "runs"
DEFAULT_NAME = "yolo11n_l0_all25"
MODEL_DESTINATION = TRAINING_ROOT.parent / "models" / "drone_yolo11n_l0.pt"
ARTIFACTS = TRAINING_ROOT / "artifacts"


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--name", default=DEFAULT_NAME)
    arguments = parser.parse_args()

    if not arguments.data.is_file():
        raise FileNotFoundError(
            f"Missing {arguments.data}. Run training/data_prep.py before training."
        )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    DEFAULT_PROJECT.mkdir(parents=True, exist_ok=True)
    MODEL_DESTINATION.parent.mkdir(parents=True, exist_ok=True)

    configuration = {
        "model_checkpoint": arguments.model,
        "pretrained_source": (
            "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
            if arguments.model == "yolo11n.pt" else "Ultralytics asset resolved by YOLO()"
        ),
        "data": str(arguments.data.resolve()),
        "epochs": arguments.epochs,
        "batch": arguments.batch,
        "imgsz": arguments.imgsz,
        "device": arguments.device,
        "optimizer": "auto (Ultralytics selection)",
        "learning_rate": "Ultralytics auto/default",
        "augmentations": {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0,
            "close_mosaic": 10,
        },
        "seed": arguments.seed,
        "deterministic": True,
        "rectangular_batches": True,
        "workers": 0,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python")
        },
        "torch_threads": torch.get_num_threads(),
    }
    metadata_path = ARTIFACTS / "training_metadata.json"
    metadata_path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")

    started = time.perf_counter()
    model = YOLO(arguments.model)
    result = model.train(
        data=str(arguments.data.resolve()),
        epochs=arguments.epochs,
        batch=arguments.batch,
        imgsz=arguments.imgsz,
        device=arguments.device,
        workers=0,
        project=str(DEFAULT_PROJECT.resolve()),
        name=arguments.name,
        exist_ok=True,
        seed=arguments.seed,
        deterministic=True,
        optimizer="auto",
        pretrained=True,
        rect=True,
        amp=False,
        plots=True,
        val=True,
        verbose=True,
    )
    wall_seconds = time.perf_counter() - started
    run_dir = Path(result.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training finished without {best}")
    shutil.copy2(best, MODEL_DESTINATION)
    results_csv = run_dir / "results.csv"
    if results_csv.is_file():
        shutil.copy2(results_csv, ARTIFACTS / "training_results.csv")
    args_yaml = run_dir / "args.yaml"
    if args_yaml.is_file():
        shutil.copy2(args_yaml, ARTIFACTS / "training_args.yaml")

    configuration.update(
        {
            "status": "PASS",
            "training_wall_seconds": wall_seconds,
            "run_directory": str(run_dir.resolve()),
            "selected_checkpoint": str(best.resolve()),
            "deployed_weight": str(MODEL_DESTINATION.resolve()),
            "weight_bytes": MODEL_DESTINATION.stat().st_size,
        }
    )
    metadata_path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "wall_seconds": wall_seconds,
        "weight": str(MODEL_DESTINATION),
        "weight_bytes": MODEL_DESTINATION.stat().st_size,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
