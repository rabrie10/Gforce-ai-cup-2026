"""Independent scalar IRLS fit, transform/prediction reconstruction and IoU checks.

Primary numpy least-squares code is imported only for behavioral leakage and
synthetic support tests after saved results have been independently checked.
"""
from pathlib import Path
from statistics import median, mean
from collections import defaultdict
import csv
import hashlib
import json
import math
import sys

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PRIOR = OUT.parent / 'motion_oracle'
LIMITS = [3840,2160,3840,2160]
MODELS = ['CV','T_FIXED','T_RES_CV','S_FIXED','S_RES_CV','S_SCALE','S_RES_SCALE']


def read(name):
    with (OUT / (name+'.csv')).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=2e-7),(a,b)


def rect(b):
    return ((b[0]+b[2])/2,(b[1]+b[3])/2,b[2]-b[0],b[3]-b[1])


def score(a,b):
    x,y,w,h=rect(a); xx,yy,ww,hh=rect(b)
    if min(w,h)<=0: return 0.
    iw=max(0,min(x+w/2,xx+ww/2)-max(x-w/2,xx-ww/2))
    ih=max(0,min(y+h/2,yy+hh/2)-max(y-h/2,yy-hh/2))
    return iw*ih/(w*h+ww*hh-iw*ih)


def edge(b):
    return int(min(b[0],b[1],3840-b[2],2160-b[3])<=1)


def bucket(b):
    s=min(rect(b)[2:])/4
    return '<4' if s<4 else '4-<8' if s<8 else '8-<16' if s<16 else '16-32' if s<=32 else '>32'


def spread(x,weights):
    total=sum(weights)
    cx=sum(w*p[0] for w,p in zip(weights,x))/total
    cy=sum(w*p[1] for w,p in zip(weights,x))/total
    vx=sum(w*(p[0]-cx)**2 for w,p in zip(weights,x))/total
    vy=sum(w*(p[1]-cy)**2 for w,p in zip(weights,x))/total
    cross=sum(w*(p[0]-cx)*(p[1]-cy) for w,p in zip(weights,x))/total
    rad=math.sqrt((vx-vy)**2+4*cross**2)
    minor,maxor=max(0,(vx+vy-rad)/2),max(0,(vx+vy+rad)/2)
    return dict(rms_spread=math.sqrt(vx+vy),minor_spread=math.sqrt(minor),
                major_spread=math.sqrt(maxor),eigen_ratio=minor/maxor if maxor else 0.)


def scalar_fit(x,y):
    """Analytic weighted complex similarity, independent of numpy/SVD/lstsq."""
    weights=[1.]*len(x)
    def solve(w):
        total=sum(w)
        mx=[sum(ww*p[i] for ww,p in zip(w,x))/total for i in [0,1]]
        my=[sum(ww*p[i] for ww,p in zip(w,y))/total for i in [0,1]]
        xx=[(p[0]-mx[0],p[1]-mx[1]) for p in x]
        yy=[(p[0]-my[0],p[1]-my[1]) for p in y]
        denom=sum(ww*(p[0]**2+p[1]**2) for ww,p in zip(w,xx))
        a=sum(ww*(p[0]*q[0]+p[1]*q[1]) for ww,p,q in zip(w,xx,yy))/denom
        b=sum(ww*(p[0]*q[1]-p[1]*q[0]) for ww,p,q in zip(w,xx,yy))/denom
        tx,ty=my[0]-a*mx[0]+b*mx[1],my[1]-b*mx[0]-a*mx[1]
        errors=[math.hypot(a*p[0]-b*p[1]+tx-q[0],b*p[0]+a*p[1]+ty-q[1]) for p,q in zip(x,y)]
        return (a,b,tx,ty),errors
    for _ in range(20):
        transform,errors=solve(weights)
        weights=[min(1.,3./max(e,1e-15)) for e in errors]
    transform,errors=solve(weights)
    return transform,weights,errors


def apply(f,p):
    a,b,tx,ty=f
    return a*p[0]-b*p[1]+tx,b*p[0]+a*p[1]+ty


def correlate(x,y):
    mx,my=mean(x),mean(y)
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))


def main():
    manifest=json.loads((OUT/'preserved_inputs.json').read_text())
    for path,digest in manifest.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    boxes={}; frames=[]
    for path in sorted((ROOT/'drone-flyby/src/helsinki/annotations').glob('*.json')):
        doc=json.loads(path.read_text()); frames.append(doc['frame'])
        for ann in doc['annotations']:
            key=ann['object_id'],doc['frame']
            assert key not in boxes
            b=ann['bbox']; assert b[0]<b[2] and b[1]<b[3]
            assert all(0<=v<=lim for v,lim in zip(b,LIMITS))
            boxes[key]=b
    classes=sorted({c for c,_ in boxes})
    assert len(boxes)==259 and frames==list(range(25)) and len(classes)==16
    meta=json.loads((ROOT/'drone-flyby/src/helsinki/run_metadata.json').read_text())
    assert meta['object_totals']=={c:1 for c in classes}
    for c in classes:
        ff=sorted(k for cl,k in boxes if cl==c)
        assert ff==list(range(min(ff),max(ff)+1))
    saved=read('donor_transforms')
    assert len(saved)==384
    transforms={}; translations={}; donor_counts={}
    for row in saved:
        c,k=row['excluded_class'],int(row['end_frame'])
        assert row['update_id']==f'{c}:{k}' and int(row['start_frame'])==k-1
        names=sorted(cl for cl in classes if cl!=c and (cl,k-1) in boxes and (cl,k) in boxes)
        assert names==row['donors'].split('|') and c not in names
        assert len(names)==int(row['donor_count'])
        donor_counts[c,k]=len(names)
        x=[rect(boxes[cl,k-1])[:2] for cl in names]
        y=[rect(boxes[cl,k])[:2] for cl in names]
        g=tuple(median(q[i]-p[i] for p,q in zip(x,y)) for i in [0,1])
        translations[c,k]=g
        close(row['dx'],g[0]); close(row['dy'],g[1])
        assert int(row['boundary_donor_count'])==sum(edge(boxes[cl,k-1]) or edge(boxes[cl,k]) for cl in names)
        unweighted=spread(x,[1.]*len(x))
        assert len(x)>=3 and unweighted['rms_spread']>=100 and unweighted['minor_spread']>=25 and unweighted['eigen_ratio']>=.001
        for key,value in unweighted.items(): close(row[key],value)
        transform,weights,errors=scalar_fit(x,y)
        for key,value in zip(['a','b','tx','ty'],transform): close(row[key],value)
        for a,b in zip(weights,map(float,row['weights'].split('|'))): close(a,b)
        for a,b in zip(errors,map(float,row['donor_errors'].split('|'))): close(a,b)
        ws=spread(x,weights); effective=sum(weights)**2/sum(v*v for v in weights)
        assert effective>=3 and ws['rms_spread']>=100 and ws['minor_spread']>=25 and ws['eigen_ratio']>=.001
        close(row['weighted_rms_spread'],ws['rms_spread']); close(row['weighted_minor_spread'],ws['minor_spread'])
        close(row['weighted_eigen_ratio'],ws['eigen_ratio']); close(row['effective_donors'],effective)
        close(row['scale'],math.hypot(*transform[:2])); close(row['rotation_deg'],math.degrees(math.atan2(transform[1],transform[0])))
        close(row['fit_median'],median(errors)); close(row['fit_rms'],math.sqrt(mean(e*e for e in errors)))
        assert row['sufficient']=='1' and row['translation_sufficient']=='1' and not row['reason']
        assert (c,k) not in transforms
        transforms[c,k]=transform
    assert set(transforms)=={(c,k) for c in classes for k in range(1,25)}

    rows=read('prediction_results'); seen=set(); cohorts=defaultdict(dict)
    for r in rows:
        c,t,u,h,m=r['target_class'],int(r['origin_frame']),int(r['target_frame']),int(r['horizon']),r['model']
        key=c,t,h,m
        assert key not in seen; seen.add(key)
        assert u==t+h and h>0 and (c,t) in boxes and (c,u) in boxes
        close(r['seconds'],h/3)
        residual='_RES_' in m
        history_ok=(c,t-1) in boxes
        available=not ((m=='CV' or residual) and not history_ok)
        assert int(r['available'])==available and int(r['common_cohort'])==history_ok
        assert int(r['oracle_refreshed'])==(m!='CV')
        ff=[t-1,t] if (m=='CV' or residual) and history_ok else [t]
        assert r['target_feature_frames']=='|'.join(map(str,ff)) and max(ff)<=t
        ids=[] if not available or m=='CV' else ([t] if residual else [])+list(range(t+1,u+1))
        assert r['update_ids']=='|'.join(f'{c}:{k}' for k in ids)
        if ids: assert int(r['donor_count_min'])==min(donor_counts[c,k] for k in ids)
        else: assert not r['donor_count_min']
        assert r['origin_size_bin']==bucket(boxes[c,t]) and r['target_size_bin']==bucket(boxes[c,u])
        assert int(r['any_boundary'])==any(edge(boxes[c,k]) for k in [t-1,t,u] if (c,k) in boxes)
        cohorts[c,t,h][m]=r
        if not available:
            assert r['unavailable_reason']=='missing_target_history' and not r['iou'] and not r['raw_x1']
            continue
        cx,cy,w,ht=rect(boxes[c,t]); startw,starth=w,ht
        if m=='CV' or residual:
            px,py,pw,ph=rect(boxes[c,t-1])
        if m=='CV':
            cx,cy,w,ht=cx+h*(cx-px),cy+h*(cy-py),w+h*(w-pw),ht+h*(ht-ph)
        elif m.startswith('T_'):
            rx,ry=(cx-px-translations[c,t][0],cy-py-translations[c,t][1]) if residual else (0,0)
            cx+=sum(translations[c,k][0] for k in range(t+1,u+1))+h*rx
            cy+=sum(translations[c,k][1] for k in range(t+1,u+1))+h*ry
        else:
            pp=apply(transforms[c,t],(px,py)) if residual else (cx,cy)
            rx,ry=cx-pp[0],cy-pp[1]
            for k in range(t+1,u+1):
                cx,cy=apply(transforms[c,k],(cx,cy)); cx+=rx; cy+=ry
                if m.endswith('_SCALE'):
                    sc=math.hypot(*transforms[c,k][:2]); w*=sc; ht*=sc
        if m.endswith('_CV'): w,ht=startw+h*(startw-pw),starth+h*(starth-ph)
        raw=[cx-w/2,cy-ht/2,cx+w/2,cy+ht/2]
        pred=[min(lim,max(0,v)) for v,lim in zip(raw,LIMITS)]
        for i,coord in enumerate(['x1','y1','x2','y2']):
            close(r['raw_'+coord],raw[i]); close(r['pred_'+coord],pred[i])
        truth=boxes[c,u] # first use in reconstruction only after raw prediction
        value=score(pred,truth)
        assert 0<=value<=1; close(r['iou'],value); assert int(r['valid'])==(value>=.5)
        assert int(r['prediction_legal'])==(pred[2]>pred[0] and pred[3]>pred[1])
        tx,ty,tw,th=rect(truth); ce=math.hypot(cx-tx,cy-ty)
        close(r['center_error'],ce); close(r['center_error_short_side'],ce/min(tw,th))
        close(r['width_error'],w-tw); close(r['height_error'],ht-th)
    expected={(c,t,u-t,m) for c,t in boxes for u in range(t+1,25) if (c,u) in boxes for m in MODELS}
    assert seen==expected and len(rows)==17192
    assert all(set(g)==set(MODELS) for g in cohorts.values())
    assert sum(r['common_cohort']=='1' for r in rows)==15491
    for g in cohorts.values():
        for a,b in [('T_FIXED','S_FIXED'),('T_RES_CV','S_RES_CV')]:
            if g[a]['available']=='1' and g[b]['available']=='1':
                for lo,hi in [('x1','x2'),('y1','y2')]:
                    close(float(g[a]['raw_'+hi])-float(g[a]['raw_'+lo]),float(g[b]['raw_'+hi])-float(g[b]['raw_'+lo]))
        for a,b in [('S_FIXED','S_SCALE'),('S_RES_CV','S_RES_SCALE')]:
            if g[a]['available']=='1':
                for lo,hi in [('x1','x2'),('y1','y2')]:
                    close(float(g[a]['raw_'+hi])+float(g[a]['raw_'+lo]),float(g[b]['raw_'+hi])+float(g[b]['raw_'+lo]))

    checked_summaries=0
    for name,keys,common in [('horizon_summary',['model','horizon'],False),('common_cohort_summary',['model','horizon'],True),
        ('per_class_summary',['model','horizon','target_class'],True),('size_summary',['model','horizon','origin_size_bin'],True),
        ('target_size_summary',['model','horizon','target_size_bin'],True),('boundary_summary',['model','horizon','any_boundary'],True)]:
        groups=defaultdict(list)
        for r in rows:
            if not common or r['common_cohort']=='1': groups[tuple(r[k] for k in keys)].append(r)
        ss=read(name); assert len(ss)==len(groups)
        for r in ss:
            group=groups[tuple(r[k] for k in keys)]; aa=[v for v in group if v['available']=='1']
            assert int(r['eligible_count'])==len(group) and int(r['available_count'])==len(aa)
            valid=sum(int(v['valid']) for v in aa); assert int(r['valid_count'])==valid
            if aa:
                close(r['valid_fraction'],valid/len(aa)); close(r['mean_iou'],mean(float(v['iou']) for v in aa))
                close(r['median_iou'],median(float(v['iou']) for v in aa))
                close(r['median_center_error'],median(float(v['center_error']) for v in aa))
                close(r['median_normalized_center_error'],median(float(v['center_error_short_side']) for v in aa))
            checked_summaries+=1
    pairs=read('pairwise_common_cohorts')
    for r in pairs:
        a,b,h=r['model_a'],r['model_b'],int(r['horizon'])
        gg=[g for key,g in cohorts.items() if key[2]==h and g[a]['available']=='1' and g[b]['available']=='1']
        assert len(gg)==int(r['count'])
        assert int(r['a_valid'])==sum(int(g[a]['valid']) for g in gg)
        assert int(r['b_valid'])==sum(int(g[b]['valid']) for g in gg)
        assert int(r['a_only'])==sum(g[a]['valid']=='1' and g[b]['valid']=='0' for g in gg)
        assert int(r['b_only'])==sum(g[b]['valid']=='1' and g[a]['valid']=='0' for g in gg)
    loo=read('loo_spatial_generalization'); assert len(loo)==729
    for r in loo:
        c,k,m=r['target_class'],int(r['end_frame']),r['model']
        assert int(r['start_frame'])==k-1 and r['update_id']==f'{c}:{k}'
        assert r['available']=='1'
        pp=rect(boxes[c,k-1]); truth=boxes[c,k]; tt=rect(truth)
        predcenter=apply(transforms[c,k],pp[:2]) if m.startswith('S_') else (pp[0]+translations[c,k][0],pp[1]+translations[c,k][1])
        scale=math.hypot(*transforms[c,k][:2]) if m=='S_SCALE' else 1
        w,h=pp[2]*scale,pp[3]*scale
        pred=[predcenter[0]-w/2,predcenter[1]-h/2,predcenter[0]+w/2,predcenter[1]+h/2]
        pred=[min(lim,max(0,v)) for lim,v in zip(LIMITS,pred)]
        close(r['iou'],score(pred,truth)); close(r['center_error'],math.dist(predcenter,tt[:2]))
        close(r['error_dx'],predcenter[0]-tt[0]); close(r['error_dy'],predcenter[1]-tt[1])
    results=json.loads((OUT/'similarity_oracle_results.json').read_text())
    for r in results['spatial']:
        aa=[v for v in loo if v['model']==r['model'] and (r['cohort']=='all' or v['boundary']=='0')]
        assert r['count']==len(aa) and r['valid_count']==sum(int(v['valid']) for v in aa)
        for axis,component in [('x','dx'),('y','dy')]:
            xx=[float(v[axis]) for v in aa]; yy=[float(v['error_'+component]) for v in aa]
            close(r['corr_'+axis+'_error_'+component],correlate(xx,yy))
            groups=defaultdict(list)
            for i,v in enumerate(aa): groups[v['end_frame']].append(i)
            xd,yd=xx.copy(),yy.copy()
            for indices in groups.values():
                mx,my=mean(xx[i] for i in indices),mean(yy[i] for i in indices)
                for i in indices: xd[i]-=mx; yd[i]-=my
            close(r['within_transition_corr_'+axis+'_error_'+component],correlate(xd,yd))
    mapping={'M1_INDEPENDENT_CV':'CV','M3_ORACLE_REFRESHED_SHARED':'T_FIXED','M4_ORACLE_REFRESHED_SHARED_RESIDUAL':'T_RES_CV'}
    prior_checked=0
    with (PRIOR/'prediction_results.csv').open(newline='') as f:
        for old in csv.DictReader(f):
            if old['model'] not in mapping: continue
            r=cohorts[old['target_class'],int(old['origin_frame']),int(old['horizon'])][mapping[old['model']]]
            assert old['available']==r['available']
            if r['available']=='1':
                close(r['iou'],old['iou'])
                for coord in ['x1','y1','x2','y2']: close(r['raw_'+coord],old['raw_'+coord])
            prior_checked+=1
    for h,pair in {1:(227,227),2:(207,212),3:(179,197),4:(146,182),5:(104,168),6:(58,155),9:(0,117)}.items():
        rr=[r for r in rows if r['model']=='CV' and int(r['horizon'])==h and r['available']=='1']
        assert (sum(int(r['valid']) for r in rr),len(rr))==pair

    import run_experiment as primary
    np=primary.np
    source=primary.load_data()
    deletion_checks=0; feature_checks=0
    for c in classes:
        original=primary.build_updates(source,c)
        deleted=primary.build_updates({key:v for key,v in source.items() if key[0]!=c},c)
        poisoned=primary.build_updates({key:(v+np.array([1001.,-211.,1067.,-177.]) if key[0]==c else v) for key,v in source.items()},c)
        assert original==deleted==poisoned
        deletion_checks+=1
        for cl,t in sorted(source):
            if cl!=c or (c,t+1) not in source: continue
            h=max(k for name,k in source if name==c)-t
            # Supplying only permissible target features makes future access fail.
            history={k:source[c,k] for k in [t-1,t] if (c,k) in source}
            for m in MODELS:
                raw,_,_=primary.predict(m,t,h,history,deleted)
                r=cohorts[c,t,h][m]
                if raw is not None:
                    for coord,v in zip(['x1','y1','x2','y2'],raw): close(r['raw_'+coord],v)
            feature_checks+=1
    # Deterministic support and transform fixtures, never using Helsinki targets.
    x=np.array([[0.,0.],[1000.,0.],[0.,800.],[1000.,800.]])
    angle=.08; sc=1.07; a,b=sc*math.cos(angle),sc*math.sin(angle)
    y=x@np.array([[a,b],[-b,a]])+np.array([23.,-41.])
    fixture=primary.fit_similarity(x,y)
    assert fixture['sufficient']
    for key,value in [('a',a),('b',b),('tx',23),('ty',-41)]: close(fixture[key],value)
    cases=[(x[:2],y[:2],'too_few_donors'),(np.zeros((4,2)),np.zeros((4,2)),'degenerate_spatial_support'),
           (np.array([[0,0],[1000,0],[2000,0]]),np.array([[1,2],[1001,2],[2001,2]]),'degenerate_spatial_support'),
           (np.array([[0,0],[1000,1],[2000,0]]),np.array([[1,2],[1001,3],[2001,2]]),'degenerate_spatial_support')]
    for xx,yy,reason in cases:
        fit=primary.fit_similarity(xx,yy)
        assert not fit['sufficient'] and fit['reason']==reason and fit['a'] is None
    collapsed=primary.fit_similarity(x,np.ones_like(x)*7)
    assert not collapsed['sufficient'] and collapsed['reason']=='collapsed_transform'
    history={0:np.array([10.,20.,30.,40.]),1:np.array([11.,22.,32.,43.])}
    for m in MODELS[1:]:
        raw,_,reason=primary.predict(m,1,2,history,{})
        assert raw is None and reason=='insufficient_or_degenerate_donors'
    future_rejected=False
    try: primary.predict('CV',1,1,{**history,2:history[1]}, {})
    except AssertionError: future_rejected=True
    assert future_rejected
    for path,digest in manifest.items(): assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    assert len(list((OUT/'figures').glob('*.png')))==9
    report=dict(status='PASS',independent_transform_reconstructions=384,independent_prediction_rows=len(rows),
        common_cohort_keys=2213,matched_size_and_identical_center_ablation_checks='PASS',
        independently_checked_summary_rows=checked_summaries,pairwise_cohort_rows=len(pairs),
        held_out_transition_rows=len(loo),prior_control_rows=prior_checked,
        whole_target_deletion_and_poisoning_checks=deletion_checks,restricted_history_origin_checks=feature_checks,
        known_transform_and_degeneracy_fixtures='PASS',no_donor_fallback='PASS',future_history_rejected='PASS',
        preserved_input_files=len(manifest),figures=9)
    (OUT/'validation_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    supplementary()


def supplementary():
    """Verify appended descriptive slopes without re-running passed fit checks."""
    loo=read('loo_spatial_generalization')
    slopes=read('spatial_slopes')
    assert len(slopes)==8
    for r in slopes:
        aa=[v for v in loo if v['model']==r['model'] and (r['cohort']=='all' or v['boundary']=='0')]
        assert len(aa)==int(r['n'])
        xx=[float(v[r['axis']]) for v in aa]
        yy=[float(v['error_d'+r['axis']]) for v in aa]
        slope=(mean(x*y for x,y in zip(xx,yy))-mean(xx)*mean(yy))/(mean(x*x for x in xx)-mean(xx)**2)
        close(r['signed_error_slope_px_per_1000_source_px'],1000*slope)
        close(r['rms_signed_error'],math.sqrt(mean(y*y for y in yy)))
    checked=dict(status='PASS',descriptive_spatial_slope_rows=len(slopes))
    (OUT/'supplementary_validation_results.json').write_text(json.dumps(checked,indent=2)+'\n')
    print(json.dumps(checked,indent=2))


if __name__=='__main__':
    supplementary() if '--supplementary-only' in sys.argv else main()
