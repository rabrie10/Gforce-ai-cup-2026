"""Fine-tune a one-class discovery detector on genuine multiscale Helsinki observations.
Starts from COCO yolo11s.pt; imgsz 960 (native view long side) for small-object recall."""
import os
from ultralytics import YOLO
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/yolo")
YOLO("/workspace/assets/yolo11s.pt").train(
    data="/workspace/assets/dataset_ms/dataset.yaml",
    epochs=100, imgsz=960, batch=16, device=0, workers=8,
    seed=20260919, deterministic=True, optimizer="AdamW",
    lr0=0.002, lrf=0.05, weight_decay=0.01, freeze=0, patience=30,
    project="/workspace/training", name="ms1", exist_ok=True,
    pretrained=True, single_cls=True,
    mosaic=1.0, close_mosaic=15, mixup=0.1, copy_paste=0.0,
    degrees=12.0, translate=0.1, scale=0.5, shear=1.0, perspective=0.0,
    fliplr=0.5, flipud=0.15, hsv_h=0.015, hsv_s=0.4, hsv_v=0.35,
    plots=False, cache=False, val=True, verbose=False)
print("MS_TRAIN_COMPLETE")
