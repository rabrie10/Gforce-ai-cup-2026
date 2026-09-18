"""Independent scalar affine reconstruction, prediction, leakage and cohort checks."""
from pathlib import Path
from collections import defaultdict
from statistics import mean,median
import csv,hashlib,json,math,sys
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];SIM=OUT.parent/'motion_similarity_oracle';MOT=OUT.parent/'motion_oracle'
LIMITS=[3840,2160,3840,2160];MODELS=['CV','T_FIXED','T_RES_CV','S_FIXED','S_RES_CV','A_FIXED','A_RES_CV']
def read(n):
    with (OUT/(n+'.csv')).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def close(a,b,tol=3e-7):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=tol),(a,b)
def rect(b):return ((b[0]+b[2])/2,(b[1]+b[3])/2,b[2]-b[0],b[3]-b[1])
def iou(a,b):
    x,y,w,h=rect(a);xx,yy,ww,hh=rect(b)
    if min(w,h)<=0:return 0.
    iw=max(0,min(x+w/2,xx+ww/2)-max(x-w/2,xx-ww/2));ih=max(0,min(y+h/2,yy+hh/2)-max(y-h/2,yy-hh/2))
    return iw*ih/(w*h+ww*hh-iw*ih)
def apply(f,p):a,b,c,d,tx,ty=f;return a*p[0]+b*p[1]+tx,c*p[0]+d*p[1]+ty
def eig2(vx,cross,vy):
    rad=math.sqrt((vx-vy)**2+4*cross*cross);return ((vx+vy-rad)/2,(vx+vy+rad)/2)
def spread(x,w):
    z=sum(w);mx=sum(a*p[0] for a,p in zip(w,x))/z;my=sum(a*p[1] for a,p in zip(w,x))/z
    vx=sum(a*(p[0]-mx)**2 for a,p in zip(w,x))/z;vy=sum(a*(p[1]-my)**2 for a,p in zip(w,x))/z
    co=sum(a*(p[0]-mx)*(p[1]-my) for a,p in zip(w,x))/z;e0,e1=eig2(vx,co,vy)
    return math.sqrt(e0+e1),math.sqrt(max(0,e0)),math.sqrt(max(0,e1)),e0/e1,e1/e0
def scalar_fit(x,y):
    w=[1.]*len(x)
    def solve(ww):
        z=sum(ww);xm=[sum(a*p[i] for a,p in zip(ww,x))/z for i in [0,1]];ym=[sum(a*p[i] for a,p in zip(ww,y))/z for i in [0,1]]
        xx=[(p[0]-xm[0],p[1]-xm[1]) for p in x];yy=[(p[0]-ym[0],p[1]-ym[1]) for p in y]
        s00=sum(a*p[0]*p[0] for a,p in zip(ww,xx));s01=sum(a*p[0]*p[1] for a,p in zip(ww,xx));s11=sum(a*p[1]*p[1] for a,p in zip(ww,xx));det=s00*s11-s01*s01
        def coef(j):
            q0=sum(a*p[0]*q[j] for a,p,q in zip(ww,xx,yy));q1=sum(a*p[1]*q[j] for a,p,q in zip(ww,xx,yy))
            return ((q0*s11-q1*s01)/det,(q1*s00-q0*s01)/det)
        (a,b),(c,d)=coef(0),coef(1);tx=ym[0]-a*xm[0]-b*xm[1];ty=ym[1]-c*xm[0]-d*xm[1];f=(a,b,c,d,tx,ty)
        e=[math.dist(apply(f,p),q) for p,q in zip(x,y)];return f,e
    for _ in range(20):f,e=solve(w);w=[min(1.,3/max(v,1e-15)) for v in e]
    f,e=solve(w);return f,w,e
def corr(x,y):
    mx,my=mean(x),mean(y);return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
def main():
    manifest=json.loads((OUT/'preserved_inputs.json').read_text())
    for p,h in manifest.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    boxes={}
    for p in sorted((ROOT/'drone-flyby/src/helsinki/annotations').glob('*.json')):
        d=json.loads(p.read_text())
        for a in d['annotations']:boxes[a['object_id'],d['frame']]=a['bbox']
    classes=sorted({c for c,_ in boxes});assert len(boxes)==259 and len(classes)==16
    transforms={};saved=read('affine_transforms');assert len(saved)==384
    for r in saved:
        c,k=r['excluded_class'],int(r['end_frame']);names=sorted(x for x in classes if x!=c and (x,k-1) in boxes and (x,k) in boxes)
        assert r['donors'].split('|')==names and c not in names and int(r['donor_count'])==len(names)
        x=[rect(boxes[n,k-1])[:2] for n in names];y=[rect(boxes[n,k])[:2] for n in names];f,w,e=scalar_fit(x,y)
        for key,v in zip(['a','b','c','d','tx','ty'],f):close(r[key],v)
        for q,v in zip(r['weights'].split('|'),w):close(q,v)
        for q,v in zip(r['donor_errors'].split('|'),e):close(q,v)
        raw=spread(x,[1.]*len(x));weighted=spread(x,w)
        for key,v in zip(['rms_spread','minor_spread','major_spread','eigen_ratio','source_condition'],raw):close(r[key],v)
        for key,v in zip(['weighted_rms_spread','weighted_minor_spread','weighted_eigen_ratio','weighted_source_condition'],[weighted[0],weighted[1],weighted[3],weighted[4]]):close(r[key],v)
        eff=sum(w)**2/sum(v*v for v in w);close(r['effective_donors'],eff);close(r['fit_median'],median(e));close(r['fit_rms'],math.sqrt(mean(v*v for v in e)))
        a,b,cc,d,_,_=f;det=a*d-b*cc;close(r['determinant'],det)
        # Singular values from eigenvalues of A^T A.
        e0,e1=eig2(a*a+cc*cc,a*b+cc*d,b*b+d*d);s0,s1=math.sqrt(e0),math.sqrt(e1)
        close(r['singular_min'],s0);close(r['singular_max'],s1);close(r['anisotropy_ratio'],s1/s0);close(r['map_condition'],s1/s0)
        assert int(r['sufficient']) and det>0 and s0>=.5 and s1<=2 and s1/s0<=4 and raw[4]<=10 and weighted[4]<=10
        transforms[c,k]=f
    rows=read('prediction_results');groups=defaultdict(dict)
    for r in rows:groups[r['target_class'],int(r['origin_frame']),int(r['horizon'])][r['model']]=r
    assert len(rows)==17192 and len(groups)==2456 and all(set(g)==set(MODELS) for g in groups.values())
    assert sum(r['common_cohort']=='1' for r in rows)==15491
    prior={(r['target_class'],int(r['origin_frame']),int(r['horizon']),r['model']):r for r in csv.DictReader((SIM/'prediction_results.csv').open()) if r['model'] in MODELS[:5]}
    control_count=0
    for key,g in groups.items():
        c,t,h=key
        for m in MODELS[:5]:
            a,b=g[m],prior[c,t,h,m];assert a['available']==b['available'] and a['valid']==b['valid']
            if a['available']=='1':
                close(a['iou'],b['iou']);[close(a['raw_'+z],b['raw_'+z]) for z in ['x1','y1','x2','y2']]
            control_count+=1
        for m in ['A_FIXED','A_RES_CV']:
            r=g[m];history_ok=(c,t-1) in boxes;available=not(m=='A_RES_CV' and not history_ok)
            assert int(r['available'])==available and int(r['common_cohort'])==history_ok
            ff=[t-1,t] if m=='A_RES_CV' and history_ok else [t];assert r['target_feature_frames']=='|'.join(map(str,ff)) and max(ff)<=t
            if not available:assert r['unavailable_reason']=='missing_target_history';continue
            cx,cy,w,hgt=rect(boxes[c,t]);startw,starth=w,hgt
            if m=='A_RES_CV':
                px,py,pw,ph=rect(boxes[c,t-1]);qx,qy=apply(transforms[c,t],(px,py));rx,ry=cx-qx,cy-qy
            else:rx=ry=0
            for k in range(t+1,t+int(r['horizon'])+1):cx,cy=apply(transforms[c,k],(cx,cy));cx+=rx;cy+=ry
            if m=='A_RES_CV':w,hgt=startw+int(r['horizon'])*(startw-pw),starth+int(r['horizon'])*(starth-ph)
            raw=[cx-w/2,cy-hgt/2,cx+w/2,cy+hgt/2];pred=[min(l,max(0,v)) for v,l in zip(raw,LIMITS)]
            for z,v in zip(['x1','y1','x2','y2'],raw):close(r['raw_'+z],v)
            truth=boxes[c,int(r['target_frame'])];v=iou(pred,truth);close(r['iou'],v);assert int(r['valid'])==(v>=.5)
            tx,ty,tw,th=rect(truth);ce=math.hypot(cx-tx,cy-ty);close(r['center_error'],ce);close(r['center_error_short_side'],ce/min(tw,th))
    # All grouped summaries and pairwise cells.
    checks=0
    for n,keys,common in [('horizon_summary',['model','horizon'],0),('common_cohort_summary',['model','horizon'],1),('per_class_summary',['model','horizon','target_class'],1),('size_summary',['model','horizon','origin_size_bin'],1),('target_size_summary',['model','horizon','target_size_bin'],1),('boundary_summary',['model','horizon','any_boundary'],1)]:
        gs=defaultdict(list)
        for r in rows:
            if not common or r['common_cohort']=='1':gs[tuple(r[k] for k in keys)].append(r)
        ss=read(n);assert len(ss)==len(gs)
        for s in ss:
            aa=[r for r in gs[tuple(s[k] for k in keys)] if r['available']=='1'];success=sum(int(r['valid']) for r in aa)
            assert int(s['available_count'])==len(aa) and int(s['valid_count'])==success
            if aa:close(s['valid_fraction'],success/len(aa));close(s['median_iou'],median(float(r['iou']) for r in aa));close(s['median_center_error'],median(float(r['center_error']) for r in aa))
            checks+=1
    pair_rows=read('pairwise_common_cohorts')
    for r in pair_rows:
        a,b,h=r['model_a'],r['model_b'],int(r['horizon']);gg=[g for key,g in groups.items() if key[2]==h and g[a]['available']=='1' and g[b]['available']=='1']
        assert int(r['count'])==len(gg);assert int(r['a_only'])==sum(g[a]['valid']=='1' and g[b]['valid']=='0' for g in gg);assert int(r['b_only'])==sum(g[b]['valid']=='1' and g[a]['valid']=='0' for g in gg)
    loo=read('loo_spatial_generalization');assert len(loo)==729
    for r in [x for x in loo if x['model']=='A_FIXED']:
        c,k=r['target_class'],int(r['end_frame']);p=rect(boxes[c,k-1]);q=rect(boxes[c,k]);pred=apply(transforms[c,k],p[:2]);close(r['center_error'],math.dist(pred,q[:2]));close(r['error_dx'],pred[0]-q[0]);close(r['error_dy'],pred[1]-q[1])
    result=json.loads((OUT/'affine_oracle_results.json').read_text())
    for s in result['spatial']:
        rr=[r for r in loo if r['model']==s['model'] and (s['cohort']=='all' or r['boundary']=='0')];assert len(rr)==s['count']
        for axis,comp in [('x','dx'),('y','dy')]:
            x=[float(r[axis]) for r in rr];e=[float(r['error_'+comp]) for r in rr];close(s[f'corr_{axis}_error_{comp}'],corr(x,e))
    # Primary feature API: delete/poison target, deterministic repeat, reject future history.
    import run_experiment as primary, importlib.util
    spec=importlib.util.spec_from_file_location('verified_similarity_module',SIM/'run_experiment.py')
    verified_similarity=importlib.util.module_from_spec(spec);spec.loader.exec_module(verified_similarity)
    primary.sim=verified_similarity
    source=verified_similarity.load_data();leak=repeat=0
    for c in classes:
        base=primary.make_updates(source,c);again=primary.make_updates(source,c);deleted=primary.make_updates({k:v for k,v in source.items() if k[0]!=c},c)
        poisoned=primary.make_updates({k:(v+primary.np.array([911.,-307.,977.,-271.]) if k[0]==c else v) for k,v in source.items()},c)
        assert base==again==deleted==poisoned;leak+=1;repeat+=1
    future_rejected=False
    try:primary.predict('A_RES_CV',1,1,{0:primary.np.array([0.,0.,1.,1.]),1:primary.np.array([1.,1.,2.,2.]),2:primary.np.array([2.,2.,3.,3.])},{})
    except AssertionError:future_rejected=True
    assert future_rejected
    # Known affine plus all explicit failure paths.
    x=primary.np.array([[0.,0.],[1000.,0.],[0.,800.],[1000.,800.]])
    A=primary.np.array([[1.02,.01],[-.004,1.008]]);y=x@A.T+primary.np.array([23.,-41.]);f=primary.fit_affine(x,y);assert f['sufficient']
    for key,v in zip(['a','b','c','d'],A.ravel()):close(f[key],v)
    fixtures=[(x[:2],y[:2],'too_few_donors'),(primary.np.zeros((4,2)),primary.np.zeros((4,2)),'degenerate_spatial_support'),
      (primary.np.array([[0,0],[1000,0],[2000,0]]),primary.np.array([[1,2],[1001,2],[2001,2]]),'degenerate_spatial_support')]
    for xx,yy,reason in fixtures:q=primary.fit_affine(xx,yy);assert not q['sufficient'] and q['reason']==reason
    reflected=y.copy();reflected[:,0]*=-1;q=primary.fit_affine(x,reflected);assert not q['sufficient'] and q['reason']=='reflection_or_nonpositive_determinant'
    bad=x@primary.np.array([[3.,0.],[0.,1.]]).T;q=primary.fit_affine(x,bad);assert not q['sufficient'] and q['reason']=='implausible_or_unstable_singular_values'
    raw,_,reason=primary.predict('A_FIXED',1,2,{1:primary.np.array([0.,0.,1.,1.])},{});assert raw is None and reason=='insufficient_or_degenerate_donors'
    for p,h in manifest.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    figures=list((OUT/'figures').glob('*.png'));assert len(figures)==9 and all(p.stat().st_size>10000 for p in figures)
    report=dict(status='PASS',independent_affine_transforms=384,independent_prediction_rows=len(rows),prior_control_rows=control_count,
      common_cohort_keys=2213,summary_rows=checks,pairwise_rows=len(pair_rows),one_step_rows=len(loo),target_delete_poison_checks=leak,
      deterministic_repeat_class_checks=repeat,known_and_degenerate_fixtures='PASS',future_target_history_rejected='PASS',no_fallback='PASS',
      preserved_files=len(manifest),figures=len(figures))
    (OUT/'validation_results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
