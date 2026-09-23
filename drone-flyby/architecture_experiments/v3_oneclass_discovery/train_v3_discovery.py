"""Train, export, and hash V3 one-class discovery candidates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
DRONE = HERE.parents[1]
DATA = HERE / "generated" / "dataset" / "dataset.yaml"
RUNS = HERE / "runs"
MODELS = HERE / "models"
V2 = DRONE / "models" / "drone_yolo11n_l0.pt"
P2 = HERE / "yolo11n_p2_target.yaml"
SEED = 20260919


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_model(name: str) -> tuple[YOLO, str]:
    if name == "standard":
        return YOLO(str(V2)), "frozen V2 YOLO11n weights; detection head adapted from 16 classes to target"
    if name == "p2":
        model = YOLO(str(P2))
        model.load(str(V2))
        return model, "YOLO11n-P2 YAML with compatible frozen V2 backbone/head weights transferred where shapes match"
    raise ValueError(name)


def train(name: str, epochs: int, batch: int, imgsz: int) -> dict:
    model, initialization = candidate_model(name)
    started = time.perf_counter()
    result = model.train(
        data=str(DATA.resolve()), epochs=epochs, batch=batch, imgsz=imgsz, device="cpu", workers=0,
        project=str(RUNS.resolve()), name=f"v3_{name}", exist_ok=True, seed=SEED, deterministic=True,
        optimizer="AdamW", lr0=0.001, lrf=0.05, weight_decay=0.0005, patience=8,
        pretrained=True, rect=False, amp=False, cache=False, plots=True, val=True, verbose=True,
        hsv_h=0.015, hsv_s=0.45, hsv_v=0.35, degrees=2.0, translate=0.12, scale=0.35,
        shear=1.0, perspective=0.0002, flipud=0.0, fliplr=0.5, mosaic=0.5, mixup=0.0,
        close_mosaic=5,
    )
    wall = time.perf_counter() - started
    run_dir = Path(result.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training did not produce {best}")
    MODELS.mkdir(parents=True, exist_ok=True)
    pt = MODELS / f"v3_oneclass_{name}.pt"
    shutil.copy2(best, pt)
    export_model = YOLO(str(pt))
    exported = Path(export_model.export(format="onnx", imgsz=imgsz, dynamic=True, simplify=True,
                                        opset=17, device="cpu", half=False))
    onnx = MODELS / f"v3_oneclass_{name}.onnx"
    if exported.resolve() != onnx.resolve():
        shutil.move(str(exported), onnx)
    return {
        "candidate": name, "status": "PASS", "initialization": initialization,
        "epochs_requested": epochs, "batch": batch, "imgsz": imgsz, "seed": SEED,
        "training_wall_seconds": wall, "run_directory": str(run_dir.resolve()),
        "pt": {"path": str(pt.resolve()), "bytes": pt.stat().st_size, "sha256": sha256(pt)},
        "onnx": {"path": str(onnx.resolve()), "bytes": onnx.stat().st_size, "sha256": sha256(onnx)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("standard", "p2", "all"), default="all")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()
    if not DATA.is_file():
        raise FileNotFoundError(f"Run build_v3_dataset.py first: {DATA}")
    torch.set_num_threads(min(12, max(1, torch.get_num_threads())))
    selected = ("standard", "p2") if args.candidate == "all" else (args.candidate,)
    records = [train(name, args.epochs, args.batch, args.imgsz) for name in selected]
    manifest_path = HERE / "training_manifest.json"
    prior = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    by_name = {row["candidate"]: row for row in prior.get("candidates", [])}
    by_name.update({row["candidate"]: row for row in records})
    manifest = {
        "schema_version": 1, "seed": SEED, "deterministic": True,
        "dataset_manifest_sha256": sha256(HERE / "dataset_manifest.json"),
        "frozen_v2_initialization": {"path": str(V2.resolve()), "sha256": sha256(V2)},
        "training_policy": "select by held-out discovery recall, never training loss",
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "ultralytics": importlib.metadata.version("ultralytics"), "device": "cpu", "torch_threads": torch.get_num_threads()},
        "candidates": [by_name[name] for name in sorted(by_name)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
