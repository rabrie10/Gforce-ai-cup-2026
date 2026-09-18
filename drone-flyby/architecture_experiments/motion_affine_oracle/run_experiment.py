"""Helsinki GT-only strict-LOO affine experiment; no image access or inference."""
from pathlib import Path
from collections import defaultdict
import csv, hashlib, itertools, json, os, platform, subprocess, sys
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[2]
SIM=OUT.parent/'motion_similarity_oracle'; MOT=OUT.parent/'motion_oracle'
sys.path[:0]=[str(SIM),str(SIM/'.deps'),str(MOT/'.deps')]
import numpy as np
import run_experiment as sim

MODELS=['CV','T_FIXED','T_RES_CV','S_FIXED','S_RES_CV','A_FIXED','A_RES_CV']
LABELS={'CV':'Independent CV','T_FIXED':'Translation / fixed size','T_RES_CV':'Translation + residual',
        'S_FIXED':'Similarity / fixed size','S_RES_CV':'Similarity + residual',
        'A_FIXED':'Affine / fixed size','A_RES_CV':'Affine + residual'}
CFG=dict(min_donors=3,min_rms_spread=100.,min_minor_spread=25.,min_eigen_ratio=.001,
         huber_delta_px=3.,reweight_iterations=20,max_source_condition=10.,
         min_singular_value=.5,max_singular_value=2.,max_map_condition=4.)
LIMITS=np.array([3840.,2160.,3840.,2160.])

def write_json(n,d): (OUT/n).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def read_csv(p):
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_csv(n,rows):
    with (OUT/n).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def manifest():
    paths=[]
    for package in [MOT,SIM]:
        paths+=subprocess.check_output(['git','ls-files','--',package.relative_to(ROOT).as_posix()],cwd=ROOT,text=True).splitlines()
    paths += [p.relative_to(ROOT).as_posix() for p in sorted((sim.DATA/'annotations').glob('*.json'))]
    paths += [(sim.DATA/'run_metadata.json').relative_to(ROOT).as_posix()]
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}

def support(x,w):
    s=sim.spatial_support(x,w); eig=np.linalg.eigvalsh(np.cov(x.T,aweights=w,bias=True))
    return dict(**s,source_condition=float(eig[-1]/eig[0]) if eig[0]>0 else None)
def adequate(s):
    return s['rms_spread']>=100 and s['minor_spread']>=25 and s['eigen_ratio']>=.001 and s['source_condition']<=10
def fit_affine(x,y):
    """Weighted centered affine y=x@A.T+t with fixed Huber IRLS."""
    x=np.asarray(x,float).reshape(-1,2);y=np.asarray(y,float).reshape(-1,2);w=np.ones(len(x))
    base=dict(sufficient=0,reason='',rms_spread=0.,minor_spread=0.,major_spread=0.,eigen_ratio=0.,
        source_condition=None,weighted_rms_spread=None,weighted_minor_spread=None,weighted_eigen_ratio=None,
        weighted_source_condition=None,effective_donors=None,a=None,b=None,c=None,d=None,tx=None,ty=None,
        determinant=None,singular_min=None,singular_max=None,map_condition=None,anisotropy_ratio=None,
        rotation_deg=None,isotropic_scale=None,shear_magnitude=None,fit_median=None,fit_rms=None,fit_p90=None,
        weights='',donor_errors='')
    if len(x)<3:return dict(base,reason='too_few_donors')
    initial=support(x,w);base.update(initial)
    if not adequate(initial):return dict(base,reason='degenerate_spatial_support')
    def solve(ww):
        xm=np.average(x,axis=0,weights=ww);ym=np.average(y,axis=0,weights=ww)
        xc=x-xm;yc=y-ym; sw=np.sqrt(ww)
        # One shared weighted design; solve both destination coordinates together.
        coef=np.linalg.lstsq(xc*sw[:,None],yc*sw[:,None],rcond=None)[0]
        A=coef.T;t=ym-A@xm;err=np.linalg.norm(x@A.T+t-y,axis=1)
        return A,t,err
    for _ in range(20):
        A,t,err=solve(w);w=np.minimum(1.,3./np.maximum(err,1e-15))
    A,t,err=solve(w);weighted=support(x,w);eff=float(w.sum()**2/(w@w));sv=np.linalg.svd(A,compute_uv=False)
    det=float(np.linalg.det(A)); cond=float(sv[0]/sv[1]) if sv[1]>0 else None
    out=dict(base,weighted_rms_spread=weighted['rms_spread'],weighted_minor_spread=weighted['minor_spread'],
        weighted_eigen_ratio=weighted['eigen_ratio'],weighted_source_condition=weighted['source_condition'],
        effective_donors=eff,weights='|'.join(format(v,'.17g') for v in w),
        donor_errors='|'.join(format(v,'.17g') for v in err))
    if eff<3 or not adequate(weighted):return dict(out,reason='degenerate_weighted_support')
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(t)):return dict(out,reason='nonfinite_transform')
    if det<=0:return dict(out,reason='reflection_or_nonpositive_determinant')
    if sv[1]<.5 or sv[0]>2 or cond>4:return dict(out,reason='implausible_or_unstable_singular_values')
    # Polar rotation and symmetric stretch diagnose, without asserting camera physics.
    U,_,Vt=np.linalg.svd(A);R=U@Vt;P=Vt.T@np.diag(sv)@Vt
    return dict(out,sufficient=1,a=float(A[0,0]),b=float(A[0,1]),c=float(A[1,0]),d=float(A[1,1]),
        tx=float(t[0]),ty=float(t[1]),determinant=det,singular_min=float(sv[1]),singular_max=float(sv[0]),
        map_condition=cond,anisotropy_ratio=cond,rotation_deg=float(np.degrees(np.arctan2(R[1,0],R[0,0]))),
        isotropic_scale=float(np.sqrt(det)),shear_magnitude=float(abs(P[0,1])),
        fit_median=float(np.median(err)),fit_rms=float(np.sqrt(np.mean(err**2))),fit_p90=float(np.percentile(err,90)))

def make_updates(boxes,target):
    classes=sorted({c for c,_ in boxes if c!=target});out={}
    for k in range(1,25):
        donors=[c for c in classes if (c,k-1) in boxes and (c,k) in boxes]
        x=np.array([sim.center(boxes[c,k-1]) for c in donors]).reshape(-1,2)
        y=np.array([sim.center(boxes[c,k]) for c in donors]).reshape(-1,2)
        out[k]=dict(update_id=f'{target}:{k}',excluded_class=target,start_frame=k-1,end_frame=k,
            donors='|'.join(donors),donor_count=len(donors),boundary_donor_count=sum(sim.boundary(boxes[c,k-1]) or sim.boundary(boxes[c,k]) for c in donors),**fit_affine(x,y))
    return out
def apply(u,p):return np.array([u['a']*p[0]+u['b']*p[1]+u['tx'],u['c']*p[0]+u['d']*p[1]+u['ty']])
def predict(model,t,h,history,updates):
    assert max(history)<=t
    if model=='A_RES_CV' and t-1 not in history:return None,[],'missing_target_history'
    ids=([t] if model=='A_RES_CV' else [])+list(range(t+1,t+h+1))
    if any(k not in updates or not updates[k]['sufficient'] for k in ids):return None,ids,'insufficient_or_degenerate_donors'
    b=history[t];c=sim.center(b);s=sim.size(b)
    r=c-apply(updates[t],sim.center(history[t-1])) if model=='A_RES_CV' else np.zeros(2)
    for k in range(t+1,t+h+1):
        c=apply(updates[k],c)+r
        if not np.all(np.isfinite(c)):return None,ids,'nonfinite_prediction'
    if model=='A_RES_CV':s=s+h*(s-sim.size(history[t-1]))
    return sim.corners(c,s),ids,''

def control_rows():
    keep={'CV','T_FIXED','T_RES_CV','S_FIXED','S_RES_CV'}
    return [dict(r,oracle_refreshed=int(r['model']!='CV')) for r in read_csv(SIM/'prediction_results.csv') if r['model'] in keep]
def controls_gate(rows):
    prior={(r['target_class'],r['origin_frame'],r['horizon'],r['model']):r for r in read_csv(SIM/'prediction_results.csv')}
    assert len(rows)==12280
    for r in rows:
        p=prior[r['target_class'],r['origin_frame'],r['horizon'],r['model']]
        assert all(r[k]==p[k] for k in p if k!='oracle_refreshed')
        assert int(r['oracle_refreshed'])==int(p['oracle_refreshed'])
    result=dict(status='PASS',exact_rows=len(rows),cv_baseline=json.loads((MOT/'baseline_reproduction.json').read_text()),
        translation_validation=json.loads((MOT/'validation_results.json').read_text())['status'],
        similarity_validation=json.loads((SIM/'validation_results.json').read_text())['status'])
    write_json('control_reproduction.json',result);return result
def affine_rows(boxes,updates):
    out=[]
    for (c,t),b in sorted(boxes.items()):
        history={k:v for (cl,k),v in boxes.items() if cl==c and k<=t}
        for u in range(t+1,25):
            if (c,u) not in boxes:continue
            for model in ['A_FIXED','A_RES_CV']:
                raw,ids,reason=predict(model,t,u-t,history,updates[c]);truth=boxes[c,u];available=raw is not None
                pred=np.clip(raw,0,LIMITS) if available else None;ce=float(np.linalg.norm(sim.center(raw)-sim.center(truth))) if available else None
                r=dict(target_class=c,origin_frame=t,target_frame=u,horizon=u-t,seconds=(u-t)/3,model=model,oracle_refreshed=1,
                    available=int(available),unavailable_reason=reason,target_feature_frames='|'.join(map(str,[t-1,t] if model=='A_RES_CV' and t-1 in history else [t])),
                    update_ids='|'.join(f'{c}:{k}' for k in ids),donor_count_min=min((updates[c][k]['donor_count'] for k in ids),default=None),
                    origin_size_bin=sim.size_bin(b),target_size_bin=sim.size_bin(truth),
                    any_boundary=int(any(sim.boundary(boxes[c,k]) for k in [t-1,t,u] if (c,k) in boxes)),
                    origin_boundary=sim.boundary(b),target_boundary=sim.boundary(truth),origin_cx=float(sim.center(b)[0]),origin_cy=float(sim.center(b)[1]),
                    iou=sim.iou(pred,truth) if available else None,valid=int(sim.iou(pred,truth)>=.5) if available else None,
                    prediction_legal=int(np.all(sim.size(pred)>0)) if available else None,center_error=ce,
                    center_error_short_side=ce/min(sim.size(truth)) if available else None,
                    clipped_center_error=float(np.linalg.norm(sim.center(pred)-sim.center(truth))) if available else None,
                    width_error=float(sim.size(raw)[0]-sim.size(truth)[0]) if available else None,height_error=float(sim.size(raw)[1]-sim.size(truth)[1]) if available else None)
                for i,z in enumerate(['x1','y1','x2','y2']):r['raw_'+z]=float(raw[i]) if available else None;r['pred_'+z]=float(pred[i]) if available else None
                out.append(r)
    return out
def normalize_controls(rows):
    # Ensure schema/order exactly matches new rows; values remain numerically identical.
    fields=list(rows[0]);out=[]
    for old in control_rows():
        r={k:old.get(k,'') for k in fields};r['oracle_refreshed']=int(old['model']!='CV')
        for k in ['origin_frame','target_frame','horizon','available','valid','prediction_legal','origin_boundary','target_boundary','any_boundary']:
            if r.get(k,'')!='':r[k]=int(r[k])
        out.append(r)
    return out
def summary(rows,keys):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    out=[]
    for key,g in sorted(groups.items()):
        a=[r for r in g if int(r['available'])]
        out.append(dict(zip(keys,key),eligible_count=len(g),available_count=len(a),availability_fraction=len(a)/len(g),
            valid_count=sum(int(r['valid']) for r in a),valid_fraction=sum(int(r['valid']) for r in a)/len(a) if a else None,
            mean_iou=float(np.mean([float(r['iou']) for r in a])) if a else None,median_iou=float(np.median([float(r['iou']) for r in a])) if a else None,
            mean_center_error=float(np.mean([float(r['center_error']) for r in a])) if a else None,
            median_center_error=float(np.median([float(r['center_error']) for r in a])) if a else None,
            median_normalized_center_error=float(np.median([float(r['center_error_short_side']) for r in a])) if a else None))
    return out
def commonize(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['target_class'],r['origin_frame'],r['horizon']].append(r)
    for g in groups.values():
        v=int(len(g)==7 and all(int(r['available']) for r in g))
        for r in g:r['common_cohort']=v
    return rows
def loo(boxes,updates):
    rows=[]
    prior=read_csv(SIM/'loo_spatial_generalization.csv')
    for r in prior:
        if r['model'] in ['T_FIXED','S_FIXED']:rows.append(dict(r,unavailable_reason=''))
    for (c,k),truth in sorted(boxes.items()):
        if (c,k-1) not in boxes:continue
        raw,_,reason=predict('A_FIXED',k-1,1,{k-1:boxes[c,k-1]},updates[c]);available=raw is not None
        err=sim.center(raw)-sim.center(truth) if available else None
        rows.append(dict(target_class=c,start_frame=k-1,end_frame=k,model='A_FIXED',update_id=f'{c}:{k}',available=int(available),
            x=float(sim.center(boxes[c,k-1])[0]),y=float(sim.center(boxes[c,k-1])[1]),boundary=int(sim.boundary(boxes[c,k-1]) or sim.boundary(truth)),
            center_error=float(np.linalg.norm(err)) if available else None,error_dx=float(err[0]) if available else None,error_dy=float(err[1]) if available else None,
            iou=sim.iou(np.clip(raw,0,LIMITS),truth) if available else None,valid=int(sim.iou(np.clip(raw,0,LIMITS),truth)>=.5) if available else None,
            unavailable_reason=reason))
    summaries=[]
    bad={(r['target_class'],r['end_frame']) for r in rows if not int(r['available'])}
    for m in ['T_FIXED','S_FIXED','A_FIXED']:
        for cohort in ['all','nonboundary']:
            rr=[r for r in rows if r['model']==m and (cohort=='all' or not int(r['boundary'])) and (r['target_class'],r['end_frame']) not in bad]
            d=dict(model=m,cohort=cohort,count=len(rr),valid_count=sum(int(r['valid']) for r in rr),success_fraction=sum(int(r['valid']) for r in rr)/len(rr),
                mean_iou=float(np.mean([float(r['iou']) for r in rr])),median_iou=float(np.median([float(r['iou']) for r in rr])),
                mean_center_error=float(np.mean([float(r['center_error']) for r in rr])),median_center_error=float(np.median([float(r['center_error']) for r in rr])),
                p90_center_error=float(np.percentile([float(r['center_error']) for r in rr],90)))
            for axis,comp in [('x','dx'),('y','dy')]:
                x=np.array([float(r[axis]) for r in rr]);e=np.array([float(r['error_'+comp]) for r in rr]);d[f'corr_{axis}_error_{comp}']=float(np.corrcoef(x,e)[0,1])
                d[f'{axis}_slope_per_1000']=float(np.cov(x,e,bias=True)[0,1]/np.var(x)*1000);d[f'rms_{comp}']=float(np.sqrt(np.mean(e**2)))
                by=defaultdict(list)
                for i,r in enumerate(rr):by[r['end_frame']].append(i)
                xx=x.copy();ee=e.copy()
                for ids in by.values():xx[ids]-=x[ids].mean();ee[ids]-=e[ids].mean()
                d[f'within_transition_corr_{axis}']=float(np.corrcoef(xx,ee)[0,1])
            summaries.append(d)
    return rows,summaries
def pairs(rows):
    groups=defaultdict(dict)
    for r in rows:groups[r['target_class'],r['origin_frame'],r['horizon']][r['model']]=r
    out=[]
    for a,b in itertools.combinations(MODELS,2):
        for h in range(1,25):
            gg=[g for k,g in groups.items() if int(k[2])==h and int(g[a]['available']) and int(g[b]['available'])]
            if gg:out.append(dict(model_a=a,model_b=b,horizon=h,count=len(gg),both_success=sum(int(g[a]['valid']) and int(g[b]['valid']) for g in gg),
                a_only=sum(int(g[a]['valid']) and not int(g[b]['valid']) for g in gg),b_only=sum(int(g[b]['valid']) and not int(g[a]['valid']) for g in gg),
                both_fail=sum(not int(g[a]['valid']) and not int(g[b]['valid']) for g in gg)))
    return out
def plots(common,loo_rows,flat,classes):
    os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig');import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi':130,'axes.grid':True,'grid.alpha':.2});(OUT/'figures').mkdir(exist_ok=True)
    def finish(fig,n):fig.tight_layout();fig.savefig(OUT/'figures'/(n+'.png'));plt.close(fig)
    for metric,n,y in [('valid_fraction','success_horizon','IoU >=0.50 fraction'),('median_center_error','center_error_horizon','Median center error px')]:
        fig,ax=plt.subplots(figsize=(10,6))
        for m in ['CV','T_RES_CV','S_RES_CV','A_RES_CV']:
            rr=sorted([r for r in common if r['model']==m],key=lambda r:int(r['horizon']));ax.plot([r['horizon'] for r in rr],[r[metric] for r in rr],'.-',label=LABELS[m])
        ax.set(xlabel='Horizon (frames; 3 FPS)',ylabel=y);ax.legend();finish(fig,n)
    fig,ax=plt.subplots(figsize=(9,5))
    for m in ['S_FIXED','A_FIXED']:
        ax.ecdf([float(r['center_error']) for r in loo_rows if r['model']==m and int(r['available'])],label=LABELS[m])
    ax.set(xlabel='Held-out one-step center error px',ylabel='ECDF');ax.legend();finish(fig,'similarity_affine_error_distribution')
    for comp,axis in [('dx','x'),('dy','y')]:
        fig,ax=plt.subplots(figsize=(9,5))
        for m in ['T_FIXED','S_FIXED','A_FIXED']:
            rr=[r for r in loo_rows if r['model']==m and not int(r['boundary'])];ax.scatter([float(r[axis]) for r in rr],[float(r['error_'+comp]) for r in rr],s=10,alpha=.55,label=LABELS[m])
        ax.set(xlabel='Source '+axis,ylabel='Signed '+comp+' error px');ax.legend();finish(fig,'spatial_'+comp)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,key in zip(axes.flat,['a','b','c','d','singular_min','singular_max']):
        for k in range(1,25):
            v=[float(r[key]) for r in flat if int(r['end_frame'])==k and int(r['sufficient'])];ax.plot([k]*len(v),v,'.',alpha=.35)
        ax.set(title=key,xlabel='End frame')
    finish(fig,'affine_components_time')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key in zip(axes,['determinant','anisotropy_ratio','rotation_deg']):
        for k in range(1,25):
            v=[float(r[key]) for r in flat if int(r['end_frame'])==k and int(r['sufficient'])];ax.plot([k]*len(v),v,'.',alpha=.35)
        ax.set(title=key,xlabel='End frame')
    finish(fig,'affine_stability_time')
    fig,ax=plt.subplots(figsize=(10,6))
    fit={r['update_id']:float(r['fit_rms']) for r in flat if int(r['sufficient'])}
    for m in ['S_FIXED','A_FIXED']:
        rr=[r for r in loo_rows if r['model']==m]; source=read_csv(SIM/'donor_transforms.csv') if m=='S_FIXED' else flat
        fm={r['update_id']:float(r['fit_rms']) for r in source if str(r['sufficient']) in ['1','1.0']}
        ax.scatter([fm[r['update_id']] for r in rr],[float(r['center_error']) for r in rr],s=15,alpha=.6,label=LABELS[m])
    ax.set(xlabel='Donor fit RMS px',ylabel='Held-out target center error px');ax.legend();finish(fig,'donor_fit_vs_heldout')
    fig,ax=plt.subplots(figsize=(11,7));names=sorted({r['target_class'] for r in classes if int(r['horizon'])==6});mods=['CV','T_RES_CV','S_RES_CV','A_RES_CV']
    mat=[[next(float(r['valid_fraction']) for r in classes if r['target_class']==c and int(r['horizon'])==6 and r['model']==m) for m in mods] for c in names]
    im=ax.imshow(mat,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set_yticks(range(len(names)),names);ax.set_xticks(range(4),mods)
    for y,row in enumerate(mat):
        for x,v in enumerate(row):ax.text(x,y,f'{v:.0%}',ha='center',va='center',color='white' if v<.5 else 'black')
    fig.colorbar(im,ax=ax);finish(fig,'per_class_h6')

def main():
    before=manifest();write_json('preserved_inputs.json',before);boxes=sim.load_data();classes=sorted({c for c,_ in boxes})
    updates={c:make_updates(boxes,c) for c in classes};ctrl=control_rows();gate=controls_gate(ctrl);print('Controls PASS',gate['exact_rows'],flush=True)
    aff=affine_rows(boxes,updates);rows=commonize(normalize_controls(aff)+aff);common_rows=[r for r in rows if int(r['common_cohort'])]
    common=summary(common_rows,['model','horizon']);classes_out=summary(common_rows,['model','horizon','target_class']);loo_rows,spatial=loo(boxes,updates);flat=[r for d in updates.values() for r in d.values()]
    tables=[('prediction_results.csv',rows),('affine_transforms.csv',flat),('horizon_summary.csv',summary(rows,['model','horizon'])),('common_cohort_summary.csv',common),
      ('per_class_summary.csv',classes_out),('size_summary.csv',summary(common_rows,['model','horizon','origin_size_bin'])),('boundary_summary.csv',summary(common_rows,['model','horizon','any_boundary'])),
      ('target_size_summary.csv',summary(common_rows,['model','horizon','target_size_bin'])),('loo_spatial_generalization.csv',loo_rows),('pairwise_common_cohorts.csv',pairs(rows))]
    for n,t in tables:write_csv(n,t)
    used={i for r in rows if int(r['available']) for i in str(r['update_ids']).split('|') if i};uf=[r for r in flat if r['update_id'] in used]
    result=dict(scope='Helsinki-only GT oracle; future other-object GT only',config=CFG,models=LABELS,python=platform.python_version(),numpy='2.5.3',
      prediction_rows=len(rows),common_prediction_rows=len(common_rows),controls=gate,spatial=spatial,transform_count=len(flat),unavailable_transforms=sum(not r['sufficient'] for r in flat),
      used_transform_count=len(uf),used_transform_statistics={k:sim.stats([r[k] for r in uf if r[k] is not None]) for k in ['donor_count','effective_donors','fit_median','fit_rms','rms_spread','minor_spread','eigen_ratio','determinant','singular_min','singular_max','anisotropy_ratio','source_condition','rotation_deg','isotropic_scale','shear_magnitude']},common_cohort_summary=common)
    write_json('affine_oracle_results.json',result);plots(common,loo_rows,flat,classes_out)
    assert before==manifest();print('Saved',len(rows),'rows;',len(common_rows),'common; prior hashes unchanged',flush=True)
if __name__=='__main__':main()
