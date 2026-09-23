"""Frozen DINO features, regularized heads and discovery-independent diagnostics."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from dtos import OBJECT_CLASSES
from v2.geometry import clamp_center, crop_from_view, render_level_view, source_region, source_to_view
from v5.common import require_azure, save_json, sha256
from v5.data import negative_boxes, read_scene, split_for
from v5.visual_expert import DinoEncoder, VisualExpert


def augment(crop, rng):
    """In-plane and mild photometric changes, not synthetic 3D viewpoints."""
    yield crop
    for k in range(1,8):
        changed = np.rot90(crop,k%4).copy()
        if k>=4:
            changed = cv2.flip(changed,1)
        changed = changed.astype(np.float32)*rng.uniform(.8,1.2)+rng.uniform(-25,25,size=(1,1,3))
        yield np.uint8(np.clip(changed,0,255))


def main():
    require_azure()
    p=argparse.ArgumentParser()
    p.add_argument('--assets',type=Path,default=Path('/assets'))
    p.add_argument('--scene',type=Path,default=Path('/work/src/helsinki'))
    p.add_argument('--out',type=Path,default=Path('/results/visual'))
    p.add_argument('--size',type=int,default=224)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'examples').mkdir(exist_ok=True)
    encoder=DinoEncoder(a.assets,a.size,device=a.device)
    rng=np.random.default_rng(20260919)
    context=1.12
    records=[]
    features=[]
    pending=[]

    def add(crop,row):
        records.append(row)
        pending.append(crop)
        if len(pending)==8:
            features.extend(encoder.encode(pending))
            pending.clear()

    for frame,image,labels,image_path,label_path in read_scene(a.scene):
        split=split_for(frame)
        for ai,ann in enumerate(labels):
            idx=OBJECT_CLASSES.index(ann['object_id'])
            for level in (0,1,2):
                box=ann['bbox']
                cx,cy=clamp_center(level,(box[0]+box[2])/2,(box[1]+box[3])/2)
                region=source_region(level,cx,cy)
                view=render_level_view(image,level,cx,cy)
                local=source_to_view(box,region)
                crop=crop_from_view(view,local,a.size,context)
                base={'frame':frame,'class_index':idx,'class':ann['object_id'],'level':level,
                      'split':split,'kind':'object','source_box':box,'annotation_file':str(label_path)}
                variations=augment(crop,rng) if split=='train' else [crop]
                for aug,changed in enumerate(variations):
                    add(changed,{**base,'augmentation':aug})
                # Remove the entire annotated object rectangle before cropping.
                # This is an ablation/control, never a claimed segmentation mask.
                erased=view.copy()
                x1,y1,x2,y2=local
                x1,y1=max(0,int(np.floor(x1))-1),max(0,int(np.floor(y1))-1)
                x2,y2=min(960,int(np.ceil(x2))+1),min(540,int(np.ceil(y2))+1)
                ring=crop.reshape(-1,3)
                border=np.concatenate([crop[:3].reshape(-1,3),crop[-3:].reshape(-1,3),crop[:,:3].reshape(-1,3),crop[:,-3:].reshape(-1,3)])
                erased[y1:y2,x1:x2]=np.median(border,axis=0).astype(np.uint8)
                control=crop_from_view(erased,local,a.size,context)
                add(control,{**base,'kind':'object_erased_control','augmentation':0})
                if split!='train':
                    tile=np.concatenate([cv2.resize(crop,(196,196)),cv2.resize(control,(196,196))],axis=1)
                    cv2.imwrite(str(a.out/'examples'/f'{frame:03d}_{ann["object_id"]}_L{level}.png'),tile)
        for ni,box in enumerate(negative_boxes(image,labels,rng,12)):
            for level in (0,1,2):
                cx,cy=clamp_center(level,(box[0]+box[2])/2,(box[1]+box[3])/2)
                region=source_region(level,cx,cy)
                view=render_level_view(image,level,cx,cy)
                crop=crop_from_view(view,source_to_view(box,region),a.size,context)
                add(crop,{'frame':frame,'class_index':-1,'class':'background','level':level,
                    'split':split,'kind':'photographic_negative','source_box':box,'augmentation':0})
        print(f'VISUAL_FEATURES frame={frame} rows={len(records)}',flush=True)
    if pending:
        features.extend(encoder.encode(pending))
    x=np.asarray(features,dtype=np.float32)
    save_json(a.assets/'visual_provenance.json',records)
    np.savez_compressed(a.assets/'visual_features.npz',features=x)
    train=np.array([r['split']=='train' for r in records])
    obj=np.array([r['kind']=='object' for r in records])
    y=np.array([r['class_index'] for r in records])
    mask=train & obj
    assert set(y[mask])==set(range(16)), 'Training frames must include all official classes'
    classifier=LogisticRegression(C=10.,class_weight='balanced',max_iter=1500,solver='lbfgs',random_state=20260919)
    classifier.fit(x[mask],y[mask])
    target=LogisticRegression(C=3.,class_weight='balanced',max_iter=1500,random_state=20260919)
    target.fit(x[train],obj[train].astype(int))
    refs=mask & np.array([r['augmentation']==0 for r in records])
    np.savez_compressed(a.assets/'visual_head.npz',weight=classifier.coef_.astype(np.float32),
        bias=classifier.intercept_.astype(np.float32),target_weight=target.coef_.astype(np.float32),
        target_bias=target.intercept_.astype(np.float32),references=x[refs],reference_classes=y[refs],
        classes=np.array(OBJECT_CLASSES),size=a.size,context=context,reference_temperature=.07)
    expert=VisualExpert(a.assets,encoder)
    result=expert.classify_features(x)
    metrics={}
    for split in ('train','calibration','test'):
        for level in (0,1,2):
            for kind in ('object','object_erased_control','photographic_negative'):
                ix=[i for i,r in enumerate(records) if r['split']==split and r['level']==level and r['kind']==kind and r['augmentation']==0]
                if not ix:
                    continue
                pred=np.array([np.argmax(result[i]['class_scores']) for i in ix])
                direct=np.array([np.argmax(result[i]['direct_scores']) for i in ix])
                reference=np.array([np.argmax(result[i]['reference_scores']) for i in ix])
                truth=y[ix]
                top3=np.array([truth[k] in np.argsort(result[i]['class_scores'])[-3:] for k,i in enumerate(ix)])
                metrics[f'{split}/L{level}/{kind}']={'n':len(ix),'top1':float(np.mean(pred==truth)),
                    'top3':float(top3.mean()),'direct_top1':float(np.mean(direct==truth)),
                    'reference_top1':float(np.mean(reference==truth)),
                    'target_accept_rate':float(np.mean([result[i]['target_probability']>=.5 for i in ix])),
                    'mean_entropy':float(np.mean([result[i]['entropy'] for i in ix])),
                    'confusion_matrix':confusion_matrix(truth,pred,labels=list(range(16))).tolist() if kind!='photographic_negative' else None}
    save_json(a.out/'metrics.json',{'metrics':metrics,'classes':OBJECT_CLASSES,
        'physical_instance_generalization':False,'discovery_used':False,
        'control':'Entire verified object box replaced by border median; surrounding terrain retained.',
        'training':'Frozen pretrained DINOv2; regularized 16-way + binary logistic heads. C=10 and C=3 fixed a priori.',
        'sha256':{'head':sha256(a.assets/'visual_head.npz')}})
    save_json(a.out/'predictions.json',[{**r,**s} for r,s in zip(records,result) if r['augmentation']==0])
    print('VISUAL_COMPLETE',json.dumps({k:{q:v for q,v in value.items() if q!='confusion_matrix'} for k,value in metrics.items()}),flush=True)


if __name__=='__main__':
    main()
