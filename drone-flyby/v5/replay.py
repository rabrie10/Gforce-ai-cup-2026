"""Offline Azure replay of recorded hosted images; labels never enter predict()."""
import argparse
import json
import resource
from pathlib import Path

import cv2
import numpy as np

from dtos import DroneFlybyPredictRequestDto
from local_evaluator import Camera, build_request
from utils import encode_image
from v5.common import require_azure,save_json,sha256,timing
from v5.pipeline import Pipeline,Config


def draw(image, boxes, caption, color=(0,220,255)):
    canvas=image.copy()
    for box,label in boxes:
        x1,y1,x2,y2=[int(round(x)) for x in box]
        cv2.rectangle(canvas,(x1,y1),(x2,y2),color,1)
        if label:
            x,y=max(0,min(750,x1)),max(14,min(529,y1-3))
            cv2.putText(canvas,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(canvas,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.38,color,1,cv2.LINE_AA)
    header=np.full((42,960,3),24,np.uint8)
    cv2.putText(header,caption,(12,27),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
    return np.concatenate([header,canvas],axis=0)


def replay_hosted(config,inputs,out,indices):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    pipeline=Pipeline(config)
    rows=[]
    for index in indices:
        matches=sorted((Path(inputs)/'images').glob(f'{index:05d}*.png'))
        if len(matches)!=1:
            rows.append({'capture_index':index,'missing_or_ambiguous_inputs':[str(p) for p in matches]})
            continue
        path=matches[0]
        metadata_path=Path(inputs)/'frames'/(path.stem+'.json')
        metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if metadata.get('view',{}).get('resolution_level',0)!=0:
            raise ValueError('Hosted L0 replay received a non-L0 image')
        image=cv2.imread(str(path))
        if image is None or image.shape[:2]!=(540,960):
            raise ValueError(f'Invalid saved L0: {path}')
        source_frame=metadata.get('frame',index)
        payload=build_request(source_frame,index,Camera(),encode_image(image),None)
        # Sparse hosted samples are deliberately independent L0 observations.
        payload['sequence_id']=f'hosted-independent-{index}'
        request=DroneFlybyPredictRequestDto.model_validate(payload)
        response=pipeline.predict(request)
        diag=pipeline.last_diagnostics
        if 'error' in diag:
            raise RuntimeError(diag['error'])
        raw=diag['candidates']
        selected=[p for p in raw if p['selected']]
        raw_boxes=[(p['local_box'],'') for p in raw[:300]]
        accepted=[(p['local_box'],f'{p["candidate_rank"]}: {p["top3"][0]["class"]} {p["target_probability"]:.2f}') for p in selected]
        final=[([b*scale for b,scale in zip(p.bbox,(960,540,960,540))],f'{p.object_id} {p.confidence:.2f}') for p in response.annotations]
        v3=[]
        for prediction in metadata.get('emitted',[]):
            box=prediction.get('box',prediction.get('bbox'))
            if box is None:
                continue
            # Telemetry emitted boxes are source pixels; normalized boxes carry
            # explicit bbox values in some versions. Record the chosen contract.
            scale=(960,540,960,540) if max(box)<=1 else (.25,.25,.25,.25)
            name=prediction.get('class',prediction.get('object_id',str(prediction.get('top','?'))))
            v3.append(([b*s for b,s in zip(box,scale)],str(name)))
        panels=[draw(image,[],f'A. Recorded hosted index {index}; source frame {source_frame}'),
                draw(image,v3,'B. Saved V3 emitted predictions' if v3 else 'B. Saved V3 emitted predictions unavailable'),
                draw(image,raw_boxes,f'C. V5 raw proposals: first {min(300,len(raw))} / {len(raw)}'),
                draw(image,accepted,f'D. V5 ranked candidates ({len(selected)})'),
                draw(image,final,f'E. V5 emitted predictions ({len(final)})',(70,240,70)),
                draw(image,[],'F. No verified machine-readable manual boxes available')]
        overlay=np.concatenate([np.concatenate(panels[:3],axis=1),np.concatenate(panels[3:],axis=1)],axis=0)
        cv2.imwrite(str(out/f'hosted_{index:05d}_comparison.png'),overlay)
        cv2.imwrite(str(out/f'hosted_{index:05d}_raw.png'),image)
        # Large quadrants expose aircraft/vehicle clusters even when discovery misses.
        quadrants=[]
        for x,y in ((0,0),(480,0),(0,270),(480,270)):
            patch=cv2.resize(image[y:y+270,x:x+480],(960,540),interpolation=cv2.INTER_NEAREST)
            quadrants.append(draw(patch,[],f'Native L0 inspection x={x} y={y}; display magnification only'))
        cv2.imwrite(str(out/f'hosted_{index:05d}_inspection.png'),
            np.concatenate([np.concatenate(quadrants[:2],axis=1),np.concatenate(quadrants[2:],axis=1)],axis=0))
        diag.update({'input':str(path),'input_sha256':sha256(path),'capture_index':index,
                     'source_frame':source_frame,'saved_v3_metadata':str(metadata_path),
                     'hosted_use':'development diagnostic; no official ground truth',
                     'manual_annotations':'unavailable; analyst ROIs from different captures not transferred',
                     'raw_overlay_limit':300,'configuration':vars(config),'model_manifest':pipeline.manifest})
        save_json(out/f'hosted_{index:05d}_candidates.json',diag)
        rows.append({'capture_index':index,'source_frame':source_frame,'raw':len(raw),'selected':len(selected),
            'emitted':len(final),'timing':diag['timing']})
        print(f'HOSTED_OVERLAY {index} emitted={len(final)}',flush=True)
    save_json(out/'summary.json',{'frames':rows,'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'pipeline_ms':timing([r['timing']['total_ms'] for r in rows if 'timing' in r]),
        'accuracy':'Not measurable without verified hosted annotations', 'temporal':'independent sparse samples'})


if __name__=='__main__':
    require_azure()
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('/results/hosted'))
    p.add_argument('--indices',type=int,nargs='+',default=[60,125,0,185,245])
    a=p.parse_args()
    replay_hosted(Config(temporal=False,active_camera=False),a.inputs,a.out,a.indices)
