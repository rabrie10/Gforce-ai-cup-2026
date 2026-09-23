"""Deterministic Helsinki scene diagnostics; no learning, network, or evaluator runs."""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import math
import sys
import platform
from collections import Counter, defaultdict

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT))
from dtos import OBJECT_CLASSES, SOURCE_REGION_SIZES
from utils import source_region_for_view
from local_evaluator import Camera, render_view
from utils import decode_image

DATA = ROOT / 'src/helsinki'
W, H, FPS = 3840, 2160, 3


def write_csv(name, rows):
    assert rows, name
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), min=float(a.min()), p10=float(np.percentile(a, 10)),
                median=float(np.median(a)), mean=float(a.mean()),
                p90=float(np.percentile(a, 90)), max=float(a.max()))


def summarize(rows, keys, group='class'):
    result = []
    for c in ['ALL'] + sorted({r[group] for r in rows}):
        subset = rows if c == 'ALL' else [r for r in rows if r[group] == c]
        for key in keys:
            result.append({'class': c, 'metric': key, **stats([r[key] for r in subset])})
    return result


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda x: max(0, x[2]-x[0]) * max(0, x[3]-x[1])
    union = area(a) + area(b) - iw*ih
    return iw*ih/union if union > 0 else 0.0


def label(img, s, x, y, size=.42, color=(230, 230, 230)):
    cv2.putText(img, str(s), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def save(path, im):
    assert cv2.imwrite(str(OUT / path), im, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def region(row, level):
    width, height = SOURCE_REGION_SIZES[level]
    cx = int(np.clip(round(row['cx']), width//2, W-width//2))
    cy = int(np.clip(round(row['cy']), height//2, H-height//2))
    return source_region_for_view(level, cx, cy), cx, cy


def contains(region_, b):
    return region_[0] <= b[0] and region_[1] <= b[1] and region_[2] >= b[2] and region_[3] >= b[3]


def main():
    (OUT / 'figures').mkdir(exist_ok=True)
    (OUT / 'galleries').mkdir(exist_ok=True)
    inputs = sorted(DATA.rglob('*.*')) + [ROOT / 'dtos.py', ROOT / 'utils.py', ROOT / 'local_evaluator.py', ROOT / 'evaluator_probes/PROBE_RESULTS.md', ROOT / 'evaluator_probes/probe_results.json']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    meta = json.loads((DATA/'run_metadata.json').read_text())
    images = {int(p.stem.split('_')[-1]): p for p in (DATA/'images').glob('*.png')}
    annotations = {int(p.stem.split('_')[-1]): p for p in (DATA/'annotations').glob('*.json')}
    assert set(images) == set(annotations) == set(range(25))
    assert len(list((DATA/'images').iterdir())) == len(images)
    assert len(list((DATA/'annotations').iterdir())) == len(annotations)
    assert meta['capture']['num_frames'] == 25
    assert meta['total_objects'] == 16 and meta['object_totals'] == dict.fromkeys(OBJECT_CLASSES, 1)
    by_frame, by_class = defaultdict(list), defaultdict(list)
    rows, inventory = [], []
    for frame, path in sorted(annotations.items()):
        doc = json.loads(path.read_text())
        assert doc['frame'] == frame
        assert set(doc) == {'frame', 'pose', 'annotations', 'object_counts'}
        assert Counter(a['object_id'] for a in doc['annotations']) == doc['object_counts']
        assert all(n == 1 for n in doc['object_counts'].values())
        inventory.append(dict(frame=frame, image=images[frame].name, annotation=path.name,
                              annotations=len(doc['annotations']), **{f'pose_{k}':v for k,v in doc['pose'].items()}))
        for ann in doc['annotations']:
            assert set(ann) == {'object_id', 'bbox'}
            c = ann['object_id']; assert c in OBJECT_CLASSES
            x1,y1,x2,y2 = ann['bbox']
            assert all(isinstance(v, int) for v in ann['bbox'])
            assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
            w,h = x2-x1,y2-y1
            r = dict(frame=frame, **{'class':c}, x1=x1,y1=y1,x2=x2,y2=y2,
                     width=w,height=h,area=w*h,width_fraction=w/W,height_fraction=h/H,
                     area_fraction=w*h/(W*H),aspect_ratio=w/h,cx=(x1+x2)/2,cy=(y1+y2)/2,
                     edge_left=x1,edge_top=y1,edge_right=W-x2,edge_bottom=H-y2,
                     min_edge=min(x1,y1,W-x2,H-y2),boundary_touch=int(min(x1,y1,W-x2,H-y2)<=1))
            for level,factor in enumerate([4,2,1]):
                r.update({f'L{level}_width':w/factor,f'L{level}_height':h/factor,f'L{level}_short':min(w,h)/factor})
                short = min(w,h)/factor
                r[f'L{level}_bucket'] = '<4' if short<4 else '4-<8' if short<8 else '8-<16' if short<16 else '16-32' if short<=32 else '>32'
            rows.append(r); by_frame[frame].append(r); by_class[c].append(r)
    assert len(rows) == len({(r['frame'],r['class']) for r in rows}) == 259
    numeric = [k for k in rows[0] if k not in ['frame','class'] and not k.endswith('bucket')]
    summary = summarize(rows,numeric)
    scale_buckets = []
    for c in ['ALL']+sorted(by_class):
        rr = rows if c=='ALL' else by_class[c]
        for l in range(3):
            for b in ['<4','4-<8','8-<16','16-32','>32']:
                chosen=[r for r in rr if r[f'L{l}_bucket']==b]
                scale_buckets.append(dict(**{'class':c},level=l,bucket=b,count=len(chosen),
                    boundary_touch=sum(r['boundary_touch'] for r in chosen),
                    class_frames=';'.join(f"{r['class']}:{r['frame']}" for r in chosen)))
    temporal=[]; motion=[]
    edge_names=['left','top','right','bottom']
    for c, rr in sorted(by_class.items()):
        first,last=rr[0],rr[-1]
        gaps=sorted(set(range(first['frame'],last['frame']+1))-{r['frame'] for r in rr})
        def side(r):
            return edge_names[int(np.argmin([r['edge_'+e] for e in edge_names]))]
        temporal.append(dict(**{'class':c},first_frame=first['frame'],last_frame=last['frame'],
            visible_frames=len(rr),contiguous=not gaps,gaps=';'.join(map(str,gaps)),
            sampled_duration_s=len(rr)/FPS,first_last_span_s=(last['frame']-first['frame'])/FPS,
            first_nearest_edge=side(first),first_edge_distance=first['min_edge'],
            last_nearest_edge=side(last),last_edge_distance=last['min_edge'],
            entry='left-censored' if first['frame']==0 else ('touching-'+side(first) if first['boundary_touch'] else 'unknown-nearest-'+side(first)),
            exit='right-censored' if last['frame']==24 else ('touching-'+side(last) if last['boundary_touch'] else 'unknown-nearest-'+side(last))))
        for a,b in zip(rr,rr[1:]):
            if b['frame'] != a['frame']+1: continue
            dx,dy=b['cx']-a['cx'],b['cy']-a['cy']
            motion.append(dict(**{'class':c},frame=a['frame'],next_frame=b['frame'],dx=dx,dy=dy,
                displacement=math.hypot(dx,dy),dx_normalized=dx/W,dy_normalized=dy/H,
                normalized_displacement=math.hypot(dx/W,dy/H),
                width_change=b['width']-a['width'],height_change=b['height']-a['height'],
                area_change=b['area']-a['area'],area_ratio=b['area']/a['area'],
                scale_ratio=math.sqrt(b['area']/a['area']),velocity_px_frame=math.hypot(dx,dy),
                velocity_px_second=FPS*math.hypot(dx,dy),
                boundary_pair=int(a['boundary_touch'] or b['boundary_touch'])))
    stale=[]; survival=[]
    bbox=lambda r: np.array([r[k] for k in ['x1','y1','x2','y2']],dtype=float)
    for c,rr in sorted(by_class.items()):
        lookup={r['frame']:r for r in rr}
        for a in rr:
            t=a['frame']; prev=lookup.get(t-1)
            for method in ['hold','velocity']:
                if method=='velocity' and prev is None: continue
                valid_prefix=0; failed=False; n=0
                for h in range(1,25-t):
                    if t+h not in lookup: break
                    b=lookup[t+h]
                    pred=bbox(a) if method=='hold' else bbox(a)+h*(bbox(a)-bbox(prev))
                    # Linear corners equal center/size extrapolation. Clip to legal global frame.
                    pred=np.clip(pred,[0,0,0,0],[W,H,W,H])
                    legal=pred[2]>pred[0] and pred[3]>pred[1]
                    overlap=iou(pred,bbox(b)) if legal else 0.
                    n+=1
                    if overlap>=.5 and not failed: valid_prefix=h
                    else: failed=True
                    stale.append(dict(**{'class':c},origin_frame=t,target_frame=t+h,horizon=h,method=method,
                        common_cohort=prev is not None,iou=overlap,valid_at_050=overlap>=.5,
                        prediction_legal=bool(legal),origin_boundary=a['boundary_touch'],target_boundary=b['boundary_touch']))
                if n:
                    survival.append(dict(**{'class':c},origin_frame=t,method=method,available_future=n,
                        valid_prefix_frames=valid_prefix,valid_prefix_s=valid_prefix/FPS,
                        failure_observed=valid_prefix<n,right_censored=valid_prefix==n))
    stale_summary=[]
    for cohort in ['all_available','paired_history']:
        for c in ['ALL']+sorted(by_class):
            for method in ['hold','velocity']:
                for h in range(1,25):
                    ss=[r for r in stale if (c=='ALL' or r['class']==c) and r['method']==method and r['horizon']==h and (cohort=='all_available' or r['common_cohort'])]
                    if ss: stale_summary.append(dict(**{'class':c},cohort=cohort,method=method,horizon=h,
                        count=len(ss),valid_count=sum(r['valid_at_050'] for r in ss),
                        valid_fraction=sum(r['valid_at_050'] for r in ss)/len(ss),mean_iou=float(np.mean([r['iou'] for r in ss])),median_iou=float(np.median([r['iou'] for r in ss]))))
    pairs=[]; centered=[]
    for f,rr in sorted(by_frame.items()):
        for a,b in itertools.combinations(sorted(rr,key=lambda r:r['class']),2):
            union_w=max(a['x2'],b['x2'])-min(a['x1'],b['x1'])
            union_h=max(a['y2'],b['y2'])-min(a['y1'],b['y1'])
            pairs.append(dict(frame=f,class_a=a['class'],class_b=b['class'],
                center_distance=math.hypot(a['cx']-b['cx'],a['cy']-b['cy']),union_width=union_w,union_height=union_h,
                fits_L1=union_w<=1920 and union_h<=1080,fits_L2=union_w<=960 and union_h<=540))
        for a in rr:
            for l in [1,2]:
                reg,cx,cy=region(a,l)
                full=[b['class'] for b in rr if contains(reg,bbox(b))]
                partial=[b['class'] for b in rr if iou(reg,bbox(b))>0]
                nextfull=[b['class'] for b in by_frame.get(f+1,[]) if contains(reg,bbox(b))]
                centered.append(dict(frame=f,target=a['class'],level=l,cx=cx,cy=cy,
                    source_region=';'.join(map(str,reg)),full_count=len(full),intersect_count=len(partial),
                    other_full_count=len([c for c in full if c!=a['class']]),
                    full_classes=';'.join(full),next_frame_full_classes=';'.join(nextfull),
                    next_frame_target_retained=a['class'] in nextfull if f<24 else '',
                    target_fits=a['class'] in full))
    pair_summary=[]
    for a,b in sorted({(r['class_a'],r['class_b']) for r in pairs}):
        pp=[r for r in pairs if r['class_a']==a and r['class_b']==b]
        pair_summary.append(dict(class_a=a,class_b=b,co_visible_frames=len(pp),
            fits_L1=sum(r['fits_L1'] for r in pp),fits_L2=sum(r['fits_L2'] for r in pp),
            median_center_distance=float(np.median([r['center_distance'] for r in pp]))))
    # Visuals: exact rendered views, then padded object context, expanded only by nearest neighbor.
    contact=np.full((5*240,5*400,3),24,np.uint8)
    comparison_rows={}; gallery_rows=defaultdict(list); representatives={}
    for c,rr in by_class.items():
        candidates=[r for r in rr if not r['boundary_touch']] or rr
        representatives[c]=min(candidates,key=lambda r:abs(r['frame']-np.median([v['frame'] for v in candidates])))['frame']
    for f,p in sorted(images.items()):
        im=cv2.imread(str(p)); assert im is not None and im.shape==(H,W,3)
        inventory[f].update(width=W,height=H)
        l0=decode_image(render_view(im,Camera()))
        assert np.array_equal(l0,cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA))
        thumb=cv2.resize(im,(384,216),interpolation=cv2.INTER_AREA)
        for a in by_frame[f]:
            b=[int(a[k]/10) for k in ['x1','y1','x2','y2']]
            cv2.rectangle(thumb,tuple(b[:2]),tuple(b[2:]),(0,220,255),1)
            label(thumb,a['class'],min(b[0],285),max(10,b[1]-3),.23,(0,255,255))
        cy,cx=(f//5)*240,(f%5)*400
        contact[cy+20:cy+236,cx:cx+384]=thumb; label(contact,f'frame {f}',cx+4,cy+15)
        for a in by_frame[f]:
            strip=np.full((250,4*224,3),30,np.uint8)
            box=bbox(a).astype(int)
            patch_box=np.array([max(0,box[0]-8),max(0,box[1]-8),min(W,box[2]+8),min(H,box[3]+8)])
            source=im[patch_box[1]:patch_box[3],patch_box[0]:patch_box[2]]
            patches=[source]
            for l in range(3):
                if l==0: reg=(0,0,W,H); view=l0
                else:
                    reg,cx,cy=region(a,l)
                    view=decode_image(render_view(im,Camera(l,cx,cy)))
                factor=4//(2**l)
                pb=np.array([math.floor((patch_box[0]-reg[0])/factor),math.floor((patch_box[1]-reg[1])/factor),
                             math.ceil((patch_box[2]-reg[0])/factor),math.ceil((patch_box[3]-reg[1])/factor)])
                pb=np.clip(pb,[0,0,0,0],[960,540,960,540])
                patch=view[pb[1]:pb[3],pb[0]:pb[2]]
                patches.append(cv2.resize(patch,None,fx=factor,fy=factor,interpolation=cv2.INTER_NEAREST))
            for j,patch in enumerate(patches):
                ph,pw=patch.shape[:2]; assert ph<=224 and pw<=224
                x=j*224+(224-pw)//2; y=28+(216-ph)//2
                strip[y:y+ph,x:x+pw]=patch
                title='source 1:1' if j==0 else f'L{j-1} pixels x{[4,2,1][j-1]}'
                label(strip,title,j*224+5,17)
            label(strip,f"f{f} {a['class']} | source {a['width']}x{a['height']} | L0 {a['L0_width']:g}x{a['L0_height']:g}",5,246,.37)
            gallery_rows[a['class']].append(strip)
            if f==representatives[a['class']]: comparison_rows[a['class']]=strip
        print(f'Inspected geometry/rendered frame {f}',flush=True)
    save('figures/sequence_contact.png',contact)
    gallery_index=[]
    for c,strips in sorted(gallery_rows.items()):
        for page in range(math.ceil(len(strips)/5)):
            name=f'galleries/{c}_{page+1}.png'
            save(name,np.vstack(strips[page*5:(page+1)*5]))
            gallery_index.append(dict(**{'class':c},page=page+1,path=name,frames=';'.join(str(r['frame']) for r in by_class[c][page*5:(page+1)*5])))
    for page in range(4):
        cc=sorted(by_class)[page*4:(page+1)*4]
        save(f'figures/comparison_{page+1}.png',np.vstack([comparison_rows[c] for c in cc]))
    timeline=np.full((560,1160,3),24,np.uint8)
    for i,c in enumerate(sorted(by_class)):
        y=42+i*31; label(timeline,c,8,y+15)
        for f in range(25):
            if i==0: label(timeline,f,230+f*35,24)
            cv2.rectangle(timeline,(230+f*35,y),(258+f*35,y+22),(65,165,70) if any(r['frame']==f for r in by_class[c]) else (65,65,65),-1)
    save('figures/visibility_timeline.png',timeline)
    trajectories=np.full((600,1050,3),24,np.uint8)
    for i,(c,rr) in enumerate(sorted(by_class.items())):
        color=tuple(int(v) for v in cv2.cvtColor(np.uint8([[[i*11,200,255]]]),cv2.COLOR_HSV2BGR)[0,0])
        points=np.array([(int(r['cx']/4),int(r['cy']/4)) for r in rr])
        cv2.polylines(trajectories,[points],False,color,2)
        for pt in points: cv2.circle(trajectories,tuple(pt),2,color,-1)
        label(trajectories,c,points[0][0],max(12,points[0][1]),.35,color)
        label(trajectories,str(rr[-1]['frame']),points[-1][0],points[-1][1],.32,color)
    cv2.rectangle(trajectories,(0,0),(960,540),(180,180,180),1)
    label(trajectories,'Source coordinates / 4; dots = frames; class at start, final frame number at end',10,580)
    save('figures/trajectories.png',trajectories)
    outputs={'annotation_metrics.csv':rows,'class_summary.csv':summary,'frame_inventory.csv':inventory,
        'scale_buckets.csv':scale_buckets,'temporal_metrics.csv':temporal,'motion_metrics.csv':motion,
        'motion_summary.csv':summarize(motion,[k for k in motion[0] if k not in ['class','frame','next_frame']]),
        'stale_box_iou.csv':stale,'stale_summary.csv':stale_summary,'survival_horizons.csv':survival,
        'camera_pairs.csv':pairs,'camera_pair_summary.csv':pair_summary,'camera_centered.csv':centered,
        'gallery_index.csv':gallery_index}
    for name,rr in outputs.items(): write_csv(name,rr)
    assert sum(r['annotations'] for r in inventory)==len(rows)
    assert len(motion)==sum(len(rr)-1 for rr in by_class.values())
    assert all(hashes[str(p.relative_to(ROOT))]==hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs)
    result=dict(integrity='PASS',images=len(images),annotation_files=len(annotations),annotations=len(rows),
        frames=list(images),image_dimensions=[W,H],allowed_classes=list(OBJECT_CLASSES),metadata=meta,
        identity='No separate instance ID. object_id is class; metadata explicitly has one physical instance per class.',
        schema={'frame':'integer','pose':'x/y/z','annotations':'list of object_id and integer source xyxy bbox','object_counts':'per-class counts'},
        assumptions={'fps':FPS,'correspondence':'class continuity backed by metadata; Helsinki only',
                     'percentiles':'numpy linear interpolation; descriptive, not inferential',
                     'camera_geometry':'integer rounded/clamped object centers; full box containment, static geometry not reachable policy',
                     'survival':'consecutive observed-valid prefix; visible GT only; end is right-censored',
                     'velocity':'two exact consecutive GT boxes; linear center and size; clip to source, invalid boxes fail'},
        environment={'python':platform.python_version(),'numpy':np.__version__,'opencv':cv2.__version__},
        input_sha256=hashes,rows={name:len(rr) for name,rr in outputs.items()},
        boundary_annotations=sum(r['boundary_touch'] for r in rows),temporal=temporal,
        representative_frames=representatives,
        global_stale=[r for r in stale_summary if r['class']=='ALL'],
        camera={str(l):dict(target_views=len(ss:=[r for r in centered if r['level']==l]),
            isolated=sum(r['other_full_count']==0 for r in ss),other_full_stats=stats([r['other_full_count'] for r in ss]),
            pair_fit=sum(r[f'fits_L{l}'] for r in pairs),pair_total=len(pairs)) for l in [1,2]})
    (OUT/'scene_audit.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item())+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['integrity','images','annotations','boundary_annotations','camera']},indent=2))


if __name__=='__main__':
    main()
