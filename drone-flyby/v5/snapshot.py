"""Immutable diagnostic snapshot of the ONE ongoing full-strength training run.

This is an explicitly partial-training review artifact, not a replacement system.
Run while the V5 training container is paused, after the visual head exists.
"""
import argparse
import json
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO
from v5.common import require_azure,save_json,sha256


def main():
    require_azure()
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--assets',type=Path,default=Path('/assets'))
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    for name in ['visual_head.npz','dinov2_224.onnx','dinov2_140.onnx','dinov2_98.onnx','dinov2_vits14.pth','yolo11s.pt']:
        shutil.copy2(a.assets/name,a.out/name)
    checkpoint=a.assets/'training/discovery/weights/best.pt'
    shutil.copy2(checkpoint,a.out/'discovery.pt')
    shutil.copy2(a.assets/'training/discovery/weights/last.pt',a.out/'resume_last.pt')
    shutil.copy2(a.assets/'training/discovery/results.csv',a.out/'training_results.csv')
    model=YOLO(str(a.out/'discovery.pt'))
    epoch=int(model.ckpt.get('epoch',-1))+1
    torch.set_num_threads(2)
    model.export(format='onnx',imgsz=640,batch=1,dynamic=False,simplify=False,opset=17,device='cpu')
    names=['discovery.pt','discovery.onnx','visual_head.npz','dinov2_224.onnx','dinov2_140.onnx','dinov2_98.onnx','dinov2_vits14.pth','yolo11s.pt','resume_last.pt']
    save_json(a.out/'manifest.json',{'version':'5.2-review-snapshot','training_complete':False,
        'selected_detector_epoch':epoch,'planned_detector_epochs':30,
        'sha256':{n:sha256(a.out/n) for n in names},
        'source_run':str(a.assets/'training/discovery'),
        'note':'Full requested models, partial detector adaptation. Do not describe as converged or production-ready.'})
    print((a.out/'manifest.json').read_text())


if __name__=='__main__':
    main()
