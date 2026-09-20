"""Offline-only verified Helsinki views. Never imported by the endpoint."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dtos import OBJECT_CLASSES
from v2.geometry import clamp_center, render_level_view, source_region, source_to_view
from v5.common import require_azure, save_json, sha256

CALIBRATION = {4, 9, 14, 18}


def split_for(frame):
    # Hangar/medium_plane first appear after frame 18. Cover all 16 classes
    # in training, and explicitly report a 21..24 frame (not instance) holdout.
    return 'test' if frame >= 21 else ('calibration' if frame in CALIBRATION else 'train')


def read_scene(scene):
    for label_path in sorted((Path(scene) / 'annotations').glob('*.json')):
        frame = int(label_path.stem.split('_')[-1])
        image_path = Path(scene) / 'images' / (label_path.stem + '.png')
        image = cv2.imread(str(image_path))
        if image is None or image.shape[:2] != (2160, 3840):
            raise ValueError(f'Expected original 4K source: {image_path}')
        labels = json.loads(label_path.read_text())['annotations']
        yield frame, image, labels, image_path, label_path


def intersects(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def negative_boxes(image, labels, rng, count=12):
    """Photographic regions excluded from ALL annotated boxes, with a 32px guard.

    Verified relative to the supplied exhaustive Helsinki annotations, not detector
    outputs. Never assumes unlabelled hosted regions are negatives.
    """
    boxes = []
    for _ in range(500):
        side = int(rng.choice([32, 48, 80, 128, 256, 512]))
        x, y = int(rng.integers(0, 3840-side)), int(rng.integers(0, 2160-side))
        box = (x, y, x+side, y+side)
        guard = (x-32, y-32, x+side+32, y+side+32)
        if not any(intersects(guard, a['bbox']) for a in labels):
            boxes.append(box)
            if len(boxes) == count:
                break
    return boxes


def build(scene, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    records, sources = [], []
    rng = np.random.default_rng(20260919)
    for frame, image, labels, image_path, label_path in read_scene(scene):
        split = split_for(frame)
        sources.append({'frame': frame, 'image': str(image_path), 'image_sha256': sha256(image_path),
                        'annotations': str(label_path), 'annotations_sha256': sha256(label_path), 'split': split})
        views = [(0, 1920, 1080)]
        # Genuine 4K crop views, distributed across targets and object scales.
        for level in (1, 2):
            selected = labels[::max(1, len(labels)//4)][:4]
            for annotation in selected:
                x1, y1, x2, y2 = annotation['bbox']
                cx, cy = clamp_center(level, (x1+x2)/2+rng.uniform(-80, 80), (y1+y2)/2+rng.uniform(-50, 50))
                views.append((level, cx, cy))
        for index, (level, cx, cy) in enumerate(views):
            region = source_region(level, cx, cy)
            view = render_level_view(image, level, cx, cy)
            boxes = []
            for ann in labels:
                if not intersects(region, ann['bbox']):
                    continue
                local = source_to_view(ann['bbox'], region)
                box = [max(0., local[0]), max(0., local[1]), min(960., local[2]), min(540., local[3])]
                if box[2] > box[0] and box[3] > box[1]:
                    boxes.append(box)
            name = f'f{frame:03d}_L{level}_{index}'
            write_view(out, split, name, view, boxes)
            records.append({'name': name, 'split': split, 'source_frame': frame, 'level': level,
                            'source_region': region, 'boxes': boxes, 'kind': 'annotated_native_camera_view'})
        # Real textured negatives; no compositing, synthetic target patches or masks.
        for index, box in enumerate(negative_boxes(image, labels, rng, 2)):
            x1,y1,x2,y2 = box
            view = cv2.resize(image[y1:y2,x1:x2], (960,540), interpolation=cv2.INTER_LINEAR)
            name = f'f{frame:03d}_negative_{index}'
            write_view(out, split, name, view, [])
            records.append({'name': name, 'split': split, 'source_frame': frame, 'source_region': box,
                            'boxes': [], 'kind': 'photographic_annotated_target_exclusion_32px',
                            'resampled_background': True})
    save_json(out/'provenance.json', {'seed': 20260919, 'sources': sources, 'views': records,
        'limitations': ['One physical instance/class; frame splits do not establish instance generalization.',
                       'No trustworthy segmentation masks or external terrain labels; no compositing.',
                       'Negative verification is against supplied exhaustive source annotations.']})
    (out/'dataset.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/calibration\ntest: images/test\nnames:\n  0: target\n')
    print(json.dumps({'views': len(records), 'dataset': str(out)}, indent=2), flush=True)


def write_view(out, split, name, image, boxes):
    for sub in ('images', 'labels'):
        (out/sub/split).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out/'images'/split/(name+'.png')), image)
    lines = [f'0 {(b[0]+b[2])/1920:.8f} {(b[1]+b[3])/1080:.8f} {(b[2]-b[0])/960:.8f} {(b[3]-b[1])/540:.8f}' for b in boxes]
    (out/'labels'/split/(name+'.txt')).write_text('\n'.join(lines))


if __name__ == '__main__':
    require_azure()
    p = argparse.ArgumentParser()
    p.add_argument('--scene', type=Path, default=Path('/work/src/helsinki'))
    p.add_argument('--out', type=Path, default=Path('/assets/dataset_v52'))
    a = p.parse_args()
    build(a.scene, a.out)
