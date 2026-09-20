"""V7 discovery detector: one-class YOLO11s on the object-centric domain-randomised dataset.
Same architecture/interface as ms1 so the V6 pipeline can swap the checkpoint.
Geometric augmentation is left mild here because orientation/scale randomisation is already
baked into the dataset; hard wall-clock cap so the run cannot overrun the deadline."""
import os
from ultralytics import YOLO
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/yolo")
YOLO("/workspace/assets/yolo11s.pt").train(
    data="/workspace/assets/dataset_v7/dataset.yaml",
    epochs=60, imgsz=960, batch=16, device=0, workers=8,
    seed=20260920, deterministic=True, optimizer="AdamW",
    lr0=0.002, lrf=0.05, weight_decay=0.01, freeze=0, patience=20,
    project="/workspace/training", name="v7a", exist_ok=True,
    pretrained=True, single_cls=True,
    mosaic=1.0, close_mosaic=10, mixup=0.1, copy_paste=0.0,
    degrees=0.0, translate=0.1, scale=0.5, shear=1.0, perspective=0.0,
    fliplr=0.5, flipud=0.5, hsv_h=0.015, hsv_s=0.4, hsv_v=0.35,
    time=0.55,                      # hard wall-clock cap (hours)
    plots=False, cache=False, val=True, verbose=False)
print("V7_TRAIN_COMPLETE")
