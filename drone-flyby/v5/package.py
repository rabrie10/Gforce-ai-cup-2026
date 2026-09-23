"""Create a reproducible manifest once trained artifacts exist."""
import argparse
import json
import subprocess
from pathlib import Path
from dtos import OBJECT_CLASSES
from v5.common import save_json,sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--assets',type=Path,default=Path('/assets'))
    a=p.parse_args()
    names=['discovery.pt','discovery.onnx','visual_head.npz','dinov2_vits14.pth',
           'dinov2_98.onnx','dinov2_140.onnx','dinov2_224.onnx','yolo11s.pt']
    hashes={n:sha256(a.assets/n) for n in names}
    save_json(a.assets/'manifest.json',{'version':'5.2','classes':OBJECT_CLASSES,
        'sha256':hashes,'discovery':'general-pretrained YOLO11s adapted on genuine multiresolution annotated views',
        'visual':'frozen DINOv2 ViT-S/14 with regularized direct 16-class and target heads',
        'sources':{'yolo':'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt',
                   'dino':'https://github.com/facebookresearch/dinov2',
                   'dino_revision':subprocess.check_output(['git','-C',str(a.assets/'dinov2'),'rev-parse','HEAD'],text=True).strip()},
        'training_data_sha256':sha256(a.assets/'dataset_v52/provenance.json'),
        'visual_data_sha256':sha256(a.assets/'visual_provenance.json'),
        'dependencies':subprocess.check_output(['pip','freeze'],text=True).splitlines()})
    print((a.assets/'manifest.json').read_text())


if __name__=='__main__':
    main()
