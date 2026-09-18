"""Create a readable numerical appendix and descriptive spatial slopes from saved data."""
from pathlib import Path
from statistics import mean
import csv
import json

OUT=Path(__file__).resolve().parent


def read(name):
    with (OUT/(name+'.csv')).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    result=json.loads((OUT/'similarity_oracle_results.json').read_text())
    loo=read('loo_spatial_generalization')
    slopes=[]
    for model in ['T_FIXED','S_FIXED']:
        for cohort in ['all','nonboundary']:
            rr=[r for r in loo if r['model']==model and r['available']=='1' and (cohort=='all' or r['boundary']=='0')]
            for axis,error in [('x','dx'),('y','dy')]:
                xx=[float(r[axis]) for r in rr]; yy=[float(r['error_'+error]) for r in rr]
                mx,my=mean(xx),mean(yy)
                slope=sum((x-mx)*(y-my) for x,y in zip(xx,yy))/sum((x-mx)**2 for x in xx)
                slopes.append(dict(model=model,cohort=cohort,axis=axis,n=len(rr),
                    signed_error_slope_px_per_1000_source_px=slope*1000,
                    rms_signed_error=(mean(y*y for y in yy))**.5))
    with (OUT/'spatial_slopes.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(slopes[0])); w.writeheader(); w.writerows(slopes)
    text=['# Helsinki similarity measurement appendix','',
        'Generated from saved results. MEASURED, descriptive, correlated observations; no significance claims.',
        'Every T/S model uses **ORACLE REFRESHED SHARED MOTION**. CV uses target history only.',
        'Success means IoU >=0.50. Tables below use the identical seven-model common cohort.',
        'All eligible common cases are available. Full model-specific availability is in `horizon_summary.csv`.','']
    for model in result['models']:
        rr=sorted([r for r in result['common_cohort_summary'] if r['model']==model],key=lambda r:r['horizon'])
        text+=['## '+model+' — '+result['models'][model],'',table(
            ['h','Eligible N','Success N','Success %','Mean IoU','Median IoU','Mean center px','Median center px','Median center / short side','Mean abs W/H error'],
            [[r['horizon'],r['eligible_count'],r['valid_count'],f"{100*r['valid_fraction']:.2f}",
              f"{r['mean_iou']:.6f}",f"{r['median_iou']:.6f}",f"{r['mean_center_error']:.3f}",
              f"{r['median_center_error']:.3f}",f"{r['median_normalized_center_error']:.4f}",
              f"{r['mean_abs_width_error']:.3f} / {r['mean_abs_height_error']:.3f}"] for r in rr]),'']
    for filename,groupkey in [('per_class_summary','target_class'),('size_summary','origin_size_bin'),('boundary_summary','any_boundary')]:
        rows=read(filename)
        for h in [1,3,6]:
            rr=[r for r in rows if int(r['horizon'])==h]; groups=sorted({r[groupkey] for r in rr})
            models=list(result['models'])
            text+=['## '+filename+f' at h={h}','',table([groupkey,'N']+models,
                [[g,next(r['eligible_count'] for r in rr if r[groupkey]==g)]+
                 [next(r['valid_count'] for r in rr if r[groupkey]==g and r['model']==m) for m in models] for g in groups]),'']
    text+=['## Descriptive univariate spatial slopes','',
        'Signed predicted-minus-true center error versus source position. These are diagnostic slopes, not an affine motion fit.',
        '',table(['Model','Cohort','Axis','N','Error px per 1000 position px','RMS component error'],
                 [[r['model'],r['cohort'],r['axis'],r['n'],f"{r['signed_error_slope_px_per_1000_source_px']:.4f}",f"{r['rms_signed_error']:.4f}"] for r in slopes]),'']
    (OUT/'MEASUREMENTS.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps(slopes,indent=2))


if __name__=='__main__': main()
