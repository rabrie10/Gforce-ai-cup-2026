"""Broad raw one-class YOLO11s proposals; independent NMS/ranking diagnostics."""
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from v2.geometry import iou, view_to_source
from v5.common import require_azure


class Discovery:
    def __init__(self, assets, budget=32, threshold=.005, threads=2, multiscale=False, device='cpu'):
        require_azure()
        self.budget = budget
        self.threshold = threshold
        self.multiscale = multiscale
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        providers = ['CPUExecutionProvider']
        if device.startswith('cuda'):
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('CUDA requested but GPU ONNX Runtime/driver is unavailable')
            providers = [('CUDAExecutionProvider', {'device_id':int(device.split(':')[-1]) if ':' in device else 0}), 'CPUExecutionProvider']
        elif device != 'cpu':
            raise ValueError(f'Unsupported device: {device}')
        self.session = ort.InferenceSession(str(Path(assets)/'discovery.onnx'),
            sess_options=opts, providers=providers)
        self.input = self.session.get_inputs()[0]
        self.size = int(self.input.shape[-1])

    def _pass(self, image, offset=(0,0), source='full'):
        h,w = image.shape[:2]
        scale = min(self.size/w, self.size/h)
        nw,nh = round(w*scale),round(h*scale)
        dx,dy = (self.size-nw)//2,(self.size-nh)//2
        canvas = np.full((self.size,self.size,3),114,dtype=np.uint8)
        canvas[dy:dy+nh,dx:dx+nw] = cv2.resize(image,(nw,nh))
        x = np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1)[None], dtype=np.float32)/255.
        pred = self.session.run(None,{self.input.name:x})[0][0].T
        if pred.shape[1] != 5:
            raise ValueError(f'Expected one-class YOLO [xywh,target], got {pred.shape}')
        raw=[]
        for row in pred[pred[:,4]>=self.threshold]:
            cx,cy,bw,bh,score = map(float,row)
            if not np.isfinite(row).all() or bw<=0 or bh<=0 or not 0<=score<=1:
                continue
            box = [max(0.,(cx-bw/2-dx)/scale)+offset[0], max(0.,(cy-bh/2-dy)/scale)+offset[1],
                   min(float(w),(cx+bw/2-dx)/scale)+offset[0], min(float(h),(cy+bh/2-dy)/scale)+offset[1]]
            if np.isfinite(box).all() and box[2]>box[0] and box[3]>box[1]:
                raw.append({'local_box':box, 'discovery_score':score, 'candidate_source':source})
        return raw

    def propose(self,image,region):
        start=time.perf_counter()
        raw=self._pass(image)
        # Complementary spatial scale from the same detector; independently switchable.
        # This magnifies existing L0 pixels, never claims new native camera information.
        if self.multiscale:
            h,w=image.shape[:2]
            for x,y in ((0,0),(w//2,0),(0,h//2),(w//2,h//2)):
                raw.extend(self._pass(image[y:y+h//2,x:x+w//2],(x,y),'tile'))
        inferred=time.perf_counter()
        raw.sort(key=lambda p:p['discovery_score'], reverse=True)
        selected=[]
        for raw_rank,p in enumerate(raw,1):
            p.update(raw_rank=raw_rank, source_box=list(view_to_source(p['local_box'],region)),
                     candidate_rank=None, selected=False, filtered_reason='nms',emitted=False,
                     class_scores=None,top3=None,target_probability=None,margin=None,entropy=None,
                     state='not_classified')
            p['global_box']=[v/s for v,s in zip(p['source_box'],(3840,2160,3840,2160))]
            if any(iou(p['local_box'],q['local_box'])>.55 for q in selected):
                continue
            p['candidate_rank']=len(selected)+1
            p['selected']=len(selected)<self.budget
            p['filtered_reason']=None if p['selected'] else 'recognition_budget'
            selected.append(p)
        return raw, [p for p in selected if p['selected']], {
            'discovery_inference_ms':(inferred-start)*1000,
            'discovery_ranking_ms':(time.perf_counter()-inferred)*1000,
            'raw_count':len(raw),'nms_count':len(selected),'selected_count':min(len(selected),self.budget)}
