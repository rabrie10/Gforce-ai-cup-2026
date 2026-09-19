"""Frozen pretrained DINOv2 + regularized direct 16-class and target heads.

Inference reads only local model artifacts. Oracle boxes belong to the independent
offline evaluator, never to this component or its integration.
"""
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from dtos import OBJECT_CLASSES
from v2.geometry import crop_from_view
from v5.common import require_azure


def softmax(x):
    z = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return z / np.maximum(z.sum(axis=-1, keepdims=True), 1e-12)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


class DinoEncoder:
    def __init__(self, assets, size=224, threads=2, device='cpu'):
        require_azure()
        self.size = size
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        providers = ['CPUExecutionProvider']
        if device.startswith('cuda'):
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('CUDA requested but GPU ONNX Runtime/driver is unavailable')
            providers = [('CUDAExecutionProvider', {'device_id':int(device.split(':')[-1]) if ':' in device else 0}), 'CPUExecutionProvider']
        elif device != 'cpu':
            raise ValueError(f'Unsupported device: {device}')
        self.session = ort.InferenceSession(str(Path(assets)/f'dinov2_{size}.onnx'),
            sess_options=options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def encode(self, crops):
        if not crops:
            return np.empty((0,384), dtype=np.float32)
        x = np.stack([cv2.resize(c, (self.size,self.size))[:,:,::-1] for c in crops]).astype(np.float32)/255.
        x = (x - np.array([.485,.456,.406], dtype=np.float32)) / np.array([.229,.224,.225], dtype=np.float32)
        x = np.ascontiguousarray(x.transpose(0,3,1,2))
        return unit(self.session.run(None, {self.input_name:x})[0])


class VisualExpert:
    def __init__(self, assets, encoder=None, device='cpu', head_name='visual_head.npz', threads=2):
        self.assets = Path(assets)
        self.head = dict(np.load(self.assets/head_name, allow_pickle=False))
        if self.head['classes'].tolist() != list(OBJECT_CLASSES):
            raise ValueError('Visual head official class order mismatch')
        self.size = int(self.head['size'])
        self.context = float(self.head['context'])
        self.encoder = encoder or DinoEncoder(assets, self.size, device=device, threads=threads)

    def classify_features(self, features):
        h = self.head
        direct = softmax(features @ h['weight'].T + h['bias'])
        similarities = features @ h['references'].T
        scores = np.stack([np.max(similarities[:,h['reference_classes']==i], axis=1) for i in range(16)], axis=1)
        reference = softmax(scores / float(h['reference_temperature']))
        target_logits = features @ h['target_weight'].T + h['target_bias']
        target = 1./(1.+np.exp(-np.clip(target_logits[:,0], -30,30)))
        # Keep direct path primary. Reference disagreement remains visible.
        posterior = .8*direct + .2*reference
        results = []
        for i, row in enumerate(posterior):
            top = np.argsort(row)[::-1][:3]
            margin = float(row[top[0]]-row[top[1]])
            entropy = float(-(row*np.log(np.maximum(row,1e-12))).sum()/np.log(16))
            results.append({'class_scores':row.tolist(), 'direct_scores':direct[i].tolist(),
                'reference_scores':reference[i].tolist(), 'reference_similarity':scores[i].tolist(),
                'top3':[{'class':OBJECT_CLASSES[j], 'score':float(row[j])} for j in top],
                'margin':margin, 'entropy':entropy, 'target_probability':float(target[i]),
                'state':'unknown' if target[i]<.5 else ('tentative' if margin<.12 or entropy>.65 else 'confident'),
                'head_reference_agree':bool(direct[i].argmax()==reference[i].argmax())})
        return results

    def classify(self, image, boxes):
        start = time.perf_counter()
        crops = [crop_from_view(image, box, self.size, self.context) for box in boxes]
        if any(c is None for c in crops):
            raise ValueError('Invalid object crop')
        if not crops:
            return [], np.empty((0,384), dtype=np.float32), {'crop_ms':0.,'encoder_head_ms':0.}
        cropped = time.perf_counter()
        features = self.encoder.encode(crops)
        results = self.classify_features(features)
        return results, features, {'crop_ms':(cropped-start)*1000,
                                  'encoder_head_ms':(time.perf_counter()-cropped)*1000}
