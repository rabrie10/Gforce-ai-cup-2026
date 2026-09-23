"""Separate discovery metrics, integrated official scorer, and HTTP runtime."""
import argparse
import json
import os
import resource
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from dtos import DroneFlybyPredictRequestDto
from local_evaluator import Camera,build_request,render_view,score,replay
from utils import global_bbox_to_source
from v2.geometry import iou
from v5.common import require_azure,save_json,timing
from v5.data import read_scene,split_for
from v5.discovery import Discovery
from v5.pipeline import Pipeline,Config


def discovery_eval(assets,out,budget=32,multiscale=False):
    detector=Discovery(assets,budget=budget,multiscale=multiscale)
    rows=[]
    for frame,image,labels,_,_ in read_scene('/work/src/helsinki'):
        view=cv2.resize(image,(960,540),interpolation=cv2.INTER_AREA)
        raw,selected,stages=detector.propose(view,[0,0,3840,2160])
        for ann in labels:
            raw_iou=max([iou(p['source_box'],ann['bbox']) for p in raw],default=0.)
            ranked_iou=max([iou(p['source_box'],ann['bbox']) for p in selected],default=0.)
            rows.append({'frame':frame,'split':split_for(frame),'class':ann['object_id'],'box':ann['bbox'],
                         'raw_iou':raw_iou,'ranked_iou':ranked_iou,'short_source_side':min(ann['bbox'][2]-ann['bbox'][0],ann['bbox'][3]-ann['bbox'][1])})
        save_json(out/f'discovery_frame_{frame:03d}.json',{'raw':raw,'timing':stages,
            'selected_unmatched':sum(max([iou(p['source_box'],ann['bbox']) for ann in labels],default=0.)<.5 for p in selected),
            'note':'Unmatched includes background, duplicates and poor localization; not pure background FP.'})
    summary={}
    for split in ('train','calibration','test','all'):
        subset=[r for r in rows if split=='all' or r['split']==split]
        summary[split]={'n':len(subset),'raw_recall50':float(np.mean([r['raw_iou']>=.5 for r in subset])),
            'ranked_recall50':float(np.mean([r['ranked_iou']>=.5 for r in subset])),
            'mean_best_raw_iou':float(np.mean([r['raw_iou'] for r in subset]))}
    save_json(out/'discovery_summary.json',{'summary':summary,'objects':rows,'budget':budget,
        'physical_instance_generalization':False,'multiscale':multiscale,
        'miss_categories':['no raw IoU50 candidate','rank/budget loss','poor localization']})
    negative=[]
    for path in sorted((Path(os.getenv('V5_DATASET','/assets/dataset_v52'))/'images/test').glob('*negative*.png')):
        raw,selected,stages=detector.propose(cv2.imread(str(path)),[0,0,3840,2160])
        negative.append({'image':str(path),'raw_count':len(raw),'selected_count':len(selected),'timing':stages})
    save_json(out/'discovery_background.json',{'photographic_negative_views':negative,
        'verification':'No supplied annotated target within source crop plus 32px guard',
        'mean_selected_false_positives':float(np.mean([r['selected_count'] for r in negative])) if negative else None})


def integrated_eval(config,out,url):
    pipeline=Pipeline(config)
    predictions={}
    rows=[]
    for index,(frame,image,labels,_,_) in enumerate(read_scene('/work/src/helsinki')):
        payload=build_request(frame,index,Camera(),render_view(image,Camera()),None)
        request=DroneFlybyPredictRequestDto.model_validate(payload)
        response=pipeline.predict(request)
        if 'error' in pipeline.last_diagnostics:
            raise RuntimeError(pipeline.last_diagnostics['error'])
        predictions[frame]=[{'object_id':p.object_id,'bbox':global_bbox_to_source(p.bbox),'confidence':p.confidence} for p in response.annotations]
        save_json(out/f'integrated_{frame:03d}.json',pipeline.last_diagnostics)
        rows.append(pipeline.last_diagnostics['timing'])
    map50,per_class=score('helsinki',predictions)
    save_json(out/'integrated_summary.json',{'map50':map50,'per_class':per_class,
        'pipeline_ms':timing([r['total_ms'] for r in rows]),'configuration':vars(config),
        'note':'All frames scored by unchanged official local scorer; includes training frames. See component split metrics.',
        'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
    if url:
        http=[]
        for realtime in (False,True):
            requests.post(url.rsplit('/',1)[0]+'/reset',timeout=10).raise_for_status()
            prediction,stats=replay(url,'helsinki',realtime,0.,False)
            result,classes=score('helsinki',prediction)
            http.append({'realtime':realtime,'map50':result,'statistics':vars(stats),
                'round_trip_ms':timing(stats.round_trip_ms),
                'skipped_fraction':stats.frames_skipped/max(1,stats.frames_total)})
        save_json(out/'http_summary.json',{'runs':http,'metrics':requests.get(url.rsplit('/',1)[0]+'/metrics',timeout=10).json()})


if __name__=='__main__':
    require_azure()
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path('/results/evaluation'))
    p.add_argument('--url')
    p.add_argument('--discovery-only',action='store_true')
    p.add_argument('--multiscale',action='store_true')
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    config=Config(multiscale=a.multiscale)
    discovery_eval(config.assets,a.out,config.candidate_budget,config.multiscale)
    if not a.discovery_only:
        integrated_eval(config,a.out,a.url)
