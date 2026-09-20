"""Domain-adaptation detector training on RTX 4090. Starts from COCO yolo11s.pt
(not the overfit Helsinki weights). Strong augmentation + composited targets +
hard negatives. Genuine calibration split as val (Helsinki-domain reference only)."""
import os
from ultralytics import YOLO

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/yolo")
model = YOLO("/workspace/assets/yolo11s.pt")
model.train(
    data="/workspace/assets/dataset_da/dataset.yaml",
    epochs=60, imgsz=640, batch=32, device=0, workers=8,
    seed=20260919, deterministic=True, optimizer="AdamW",
    lr0=0.002, lrf=0.05, weight_decay=0.01, freeze=0, patience=20,
    project="/workspace/training", name="da1", exist_ok=True,
    pretrained=True, single_cls=True,
    mosaic=1.0, close_mosaic=15, mixup=0.15, copy_paste=0.0,
    degrees=25.0, translate=0.12, scale=0.6, shear=2.0, perspective=0.0005,
    fliplr=0.5, flipud=0.3, hsv_h=0.02, hsv_s=0.5, hsv_v=0.4,
    plots=True, cache=False, val=True, verbose=False,
)
print("DA_TRAIN_COMPLETE")
