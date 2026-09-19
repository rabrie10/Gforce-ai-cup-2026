"""Azure-only measurement before committing to detector/encoder input sizes."""
import argparse
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from v5.common import require_azure, save_json, sha256, timing


def main():
    require_azure()
    p = argparse.ArgumentParser()
    p.add_argument('--assets', type=Path, default=Path('/assets'))
    p.add_argument('--out', type=Path, default=Path('/results/feasibility.json'))
    a = p.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    a.assets.mkdir(parents=True, exist_ok=True)
    os.chdir(a.assets)
    model = YOLO('yolo11s.pt')
    dino_path = a.assets / 'dinov2'
    if not dino_path.exists():
        subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/facebookresearch/dinov2.git', str(dino_path)], check=True)
    revision = subprocess.check_output(['git', '-C', str(dino_path), 'rev-parse', 'HEAD'], text=True).strip()
    encoder = torch.hub.load(str(dino_path), 'dinov2_vits14', source='local', pretrained=True).eval()
    torch.save(encoder.state_dict(), a.assets / 'dinov2_vits14.pth')
    image = cv2.imread('/hosted/00060.png')
    if image is None:
        raise FileNotFoundError('Representative recorded hosted frame /hosted/00060.png')
    rows = []
    for size in (640, 960):
        values = []
        for i in range(12):
            start = time.perf_counter()
            model.predict(image, imgsz=size, device='cpu', conf=0.01, verbose=False)
            if i > 1:
                values.append((time.perf_counter() - start) * 1000)
        rows.append({'model': 'yolo11s', 'size': size, 'batch': 1, **timing(values)})
        save_json(a.out, {'rows': rows, 'dinov2_revision': revision})
    crops = [cv2.resize(image[200:270, 300:370], (s, s)) for s in (98, 140, 224)]
    with torch.inference_mode():
        for size, crop in zip((98, 140, 224), crops):
            for batch in (1, 8):
                x = torch.from_numpy(crop[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255)[None].repeat(batch, 1, 1, 1)
                values = []
                for i in range(12):
                    start = time.perf_counter()
                    encoder(x)
                    if i > 1:
                        values.append((time.perf_counter() - start) * 1000)
                rows.append({'model': 'dinov2_vits14', 'size': size, 'batch': batch, **timing(values)})
                save_json(a.out, {'rows': rows, 'dinov2_revision': revision})
    save_json(a.out, {'rows': rows, 'dinov2_revision': revision, 'threads': 2,
                      'cpu_quota': 2, 'note': 'Representative real image, pretrained unadapted detector; not endpoint latency',
                      'sha256': {n: sha256(a.assets / n) for n in ('yolo11s.pt', 'dinov2_vits14.pth')}})
    print(a.out.read_text(), flush=True)


if __name__ == '__main__':
    main()
