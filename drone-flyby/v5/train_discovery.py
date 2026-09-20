"""One direct pretrained YOLO11s adaptation, with native L0/L1/L2 supervision."""
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO
from v5.common import require_azure, save_json, sha256


def main():
    require_azure()
    p = argparse.ArgumentParser()
    p.add_argument('--assets', type=Path, default=Path('/assets'))
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--resume', type=Path, help='Resume this exact training run checkpoint on the selected device')
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    weights = a.assets/'yolo11s.pt'
    model = YOLO(str(weights))
    if a.resume:
        model = YOLO(str(a.resume))
        model.train(resume=True, device=a.device, batch=a.batch)
        best = Path(model.trainer.best)
        shutil.copy2(best, a.assets/'discovery.pt')
        YOLO(str(a.assets/'discovery.pt')).export(format='onnx',imgsz=640,batch=1,
            dynamic=False,simplify=False,opset=17,device='cpu')
        print('DISCOVERY_COMPLETE',sha256(a.assets/'discovery.onnx'),flush=True)
        return
    settings = dict(data=str(a.assets/'dataset_v52/dataset.yaml'), epochs=a.epochs,
        imgsz=640, batch=a.batch, device=a.device, workers=2, seed=20260919,
        deterministic=True, optimizer='AdamW', lr0=0.001, lrf=0.05,
        weight_decay=0.01, freeze=10, patience=12, project=str(a.assets/'training'),
        name='discovery', exist_ok=False, pretrained=True, single_cls=True,
        # Real camera crops supply scale variation. Avoid object-breaking mosaic,
        # mixup and segmentation-free patch compositing.
        mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=15., translate=0.08,
        scale=0.35, shear=0., perspective=0., fliplr=0.5, flipud=0.2,
        hsv_h=0.015, hsv_s=0.45, hsv_v=0.35, plots=True, cache=False)
    save_json(a.assets/'discovery_training.json', {'initial_sha256': sha256(weights), 'settings': settings})
    model.train(**settings)
    best = a.assets/'training/discovery/weights/best.pt'
    shutil.copy2(best, a.assets/'discovery.pt')
    trained = YOLO(str(a.assets/'discovery.pt'))
    exported = trained.export(format='onnx', imgsz=640, batch=1, dynamic=False,
                              simplify=False, opset=17, device='cpu')
    assert Path(exported) == a.assets/'discovery.onnx'
    print('DISCOVERY_COMPLETE', sha256(a.assets/'discovery.onnx'), flush=True)


if __name__ == '__main__':
    main()
