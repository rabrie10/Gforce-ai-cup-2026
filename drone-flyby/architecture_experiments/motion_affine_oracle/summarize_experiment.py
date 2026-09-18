"""Generate exhaustive human-readable tables and donor/held-out diagnostics."""
from pathlib import Path
from collections import defaultdict
from statistics import mean,median
import csv,json,math
OUT=Path(__file__).resolve().parent;SIM=OUT.parent/'motion_similarity_oracle'
def read(n,base=OUT):
    with (base/(n+'.csv')).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def mdtable(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def corr(x,y):
    mx,my=mean(x),mean(y);return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
def main():
    result=json.loads((OUT/'affine_oracle_results.json').read_text());common=result['common_cohort_summary'];models=list(result['models'])
    loo=read('loo_spatial_generalization');aff={r['update_id']:float(r['fit_rms']) for r in read('affine_transforms')};simi={r['update_id']:float(r['fit_rms']) for r in read('donor_transforms',SIM)}
    donor=[]
    for m,fit in [('S_FIXED',simi),('A_FIXED',aff)]:
        rr=[r for r in loo if r['model']==m];x=[fit[r['update_id']] for r in rr];y=[float(r['center_error']) for r in rr]
        donor.append(dict(model=m,n=len(rr),median_donor_fit_rms=median(x),median_heldout_center_error=median(y),correlation=corr(x,y)))
    with (OUT/'donor_fit_vs_heldout_summary.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(donor[0]));w.writeheader();w.writerows(donor)
    text=['# Helsinki affine measurement appendix','',
      'Generated from saved results. All T/S/A models are **ORACLE REFRESHED SHARED MOTION**; CV uses target history only.',
      'Success means IoU >=0.50. Common tables use the identical seven-model cohort. Observations are correlated; no significance claim.','']
    for m in models:
        rr=sorted([r for r in common if r['model']==m],key=lambda r:r['horizon'])
        text+=['## '+m+' — '+result['models'][m],'',mdtable(['h','N','Success','Success %','Mean IoU','Median IoU','Mean center','Median center','Median normalized center'],
          [[r['horizon'],r['eligible_count'],r['valid_count'],f"{100*r['valid_fraction']:.2f}",f"{r['mean_iou']:.6f}",f"{r['median_iou']:.6f}",f"{r['mean_center_error']:.3f}",f"{r['median_center_error']:.3f}",f"{r['median_normalized_center_error']:.4f}"] for r in rr]),'']
    text+=['## Strict LOO one-step spatial transfer','',mdtable(['Model','Cohort','N','Success','Success %','Mean IoU','Median IoU','Mean center','Median center','P90 center','corr x/dx','corr y/dy','within x','within y','x slope/1000','y slope/1000','RMS dx','RMS dy'],
      [[r['model'],r['cohort'],r['count'],r['valid_count'],f"{100*r['success_fraction']:.2f}",f"{r['mean_iou']:.6f}",f"{r['median_iou']:.6f}",f"{r['mean_center_error']:.3f}",f"{r['median_center_error']:.3f}",f"{r['p90_center_error']:.3f}",f"{r['corr_x_error_dx']:.6f}",f"{r['corr_y_error_dy']:.6f}",f"{r['within_transition_corr_x']:.6f}",f"{r['within_transition_corr_y']:.6f}",f"{r['x_slope_per_1000']:.4f}",f"{r['y_slope_per_1000']:.4f}",f"{r['rms_dx']:.4f}",f"{r['rms_dy']:.4f}"] for r in result['spatial']]),'']
    for filename,key in [('per_class_summary','target_class'),('size_summary','origin_size_bin'),('boundary_summary','any_boundary')]:
        rr=[r for r in read(filename) if r['horizon']=='6'];groups=sorted({r[key] for r in rr})
        text+=['## '+filename+' at h=6','',mdtable([key,'N']+models,[[g,next(r['eligible_count'] for r in rr if r[key]==g)]+[next(r['valid_count'] for r in rr if r[key]==g and r['model']==m) for m in models] for g in groups]),'']
    text+=['## Donor fit versus held-out target','',mdtable(['Model','N','Median donor RMS','Median held-out center error','Correlation'],[[r['model'],r['n'],f"{r['median_donor_fit_rms']:.4f}",f"{r['median_heldout_center_error']:.4f}",f"{r['correlation']:.4f}"] for r in donor]),'']
    (OUT/'MEASUREMENTS.md').write_text('\n'.join(text),encoding='utf-8');print(json.dumps(donor,indent=2))
if __name__=='__main__':main()
