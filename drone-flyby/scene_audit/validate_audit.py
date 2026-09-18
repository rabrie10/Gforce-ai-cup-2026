"""Read-only source checks and independent center/size verification of saved IoUs."""
import csv
import hashlib
import json
import math
from pathlib import Path
from collections import Counter

P = Path(__file__).resolve().parent
ROOT = P.parent


def rows(name):
    with (P/name).open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def center_size(box):
    x,y,X,Y=box
    return [(x+X)/2,(y+Y)/2,X-x,Y-y]


def overlap(a,b):
    ax,ay,aw,ah=center_size(a); bx,by,bw,bh=center_size(b)
    if min(aw,ah,bw,bh)<=0: return 0.
    intersection=max(0,min(ax+aw/2,bx+bw/2)-max(ax-aw/2,bx-bw/2))*max(0,min(ay+ah/2,by+bh/2)-max(ay-ah/2,by-bh/2))
    return intersection/(aw*ah+bw*bh-intersection)


def main():
    assert overlap([0,0,2,2],[0,0,1,2]) == .5
    source={}
    for f in sorted((ROOT/'src/helsinki/annotations').glob('*.json')):
        doc=json.loads(f.read_text())
        for a in doc['annotations']: source[(a['object_id'],doc['frame'])]=a['bbox']
    annotation=rows('annotation_metrics.csv')
    assert len(annotation)==len(source)==259
    assert len({(r['class'],r['frame']) for r in annotation})==259
    for r in annotation:
        box=source[r['class'],int(r['frame'])]
        assert box==[int(r[k]) for k in ['x1','y1','x2','y2']]
        assert int(r['width'])==box[2]-box[0] and int(r['height'])==box[3]-box[1]
        for level,factor in [(0,4),(1,2),(2,1)]:
            assert float(r[f'L{level}_width'])==int(r['width'])/factor
            assert float(r[f'L{level}_height'])==int(r['height'])/factor
    predictions=rows('stale_box_iou.csv')
    keys={(r['class'],r['origin_frame'],r['target_frame'],r['method']) for r in predictions}
    assert len(keys)==len(predictions)
    expected=0
    for c,t in source:
        for C,T in source:
            if C==c and T>t: expected+=1+int((c,t-1) in source)
    assert len(predictions)==expected
    for r in predictions:
        c=r['class']; t=int(r['origin_frame']); T=int(r['target_frame']); h=T-t
        assert h==int(r['horizon'])
        box=source[c,t]
        if r['method']=='velocity':
            now=center_size(box); previous=center_size(source[c,t-1])
            cx,cy,w,hh=[n+h*(n-p) for n,p in zip(now,previous)]
            box=[min(3840,max(0,cx-w/2)),min(2160,max(0,cy-hh/2)),
                 min(3840,max(0,cx+w/2)),min(2160,max(0,cy+hh/2))]
        computed=overlap(box,source[c,T])
        assert math.isclose(computed,float(r['iou']),abs_tol=1e-12)
        assert (computed>=.5)==(r['valid_at_050']=='True')
    for r in rows('stale_summary.csv'):
        chosen=[v for v in predictions if (r['class']=='ALL' or v['class']==r['class']) and v['method']==r['method'] and v['horizon']==r['horizon'] and (r['cohort']=='all_available' or v['common_cohort']=='True')]
        assert len(chosen)==int(r['count'])
        assert sum(v['valid_at_050']=='True' for v in chosen)==int(r['valid_count'])
    for l in range(3):
        assert sum(int(r['count']) for r in rows('scale_buckets.csv') if r['class']=='ALL' and r['level']==str(l))==259
    assert len(rows('motion_metrics.csv'))==243
    assert all(r['target_fits']=='True' for r in rows('camera_centered.csv'))
    assert len(rows('camera_centered.csv'))==518
    counts=Counter(r['class'] for r in annotation)
    assert sum(counts.values())==259
    assert all(int(r['visible_frames'])==counts[r['class']] for r in rows('temporal_metrics.csv'))
    gallery_keys=[]
    for r in rows('gallery_index.csv'):
        assert (P/r['path']).is_file()
        gallery_keys.extend((r['class'],int(f)) for f in r['frames'].split(';'))
    assert len(gallery_keys)==len(set(gallery_keys))==259 and set(gallery_keys)==set(source)
    manifest=json.loads((P/'scene_audit.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in manifest['input_sha256'].items())
    result={'status':'PASS','annotation_rows':259,'motion_rows':243,'independently_checked_iou_rows':len(predictions),
            'unique_gallery_annotations':259,'authoritative_inputs_unchanged':True,
            'checks':['raw annotation reconciliation','positive geometry and level scale','future horizon completeness',
                      'independent center/size extrapolation and IoU','paired summary denominators',
                      'bucket totals','camera target containment','gallery coverage','input SHA-256']}
    (P/'validation_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
