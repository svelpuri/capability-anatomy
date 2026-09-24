"""Offline Phase 5 specificity analysis. Imports no model or inference packages.

Run with --artifacts artifacts --output NEW_DIRECTORY.
Outputs never overwrite an existing directory or change source evidence.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

MODELS = ('0.6', '1.7')
TARGETS = ('tool_selection', 'argument_binding', 'full_call')
SPECS = ('log_ratio', 'relative')
MATERIAL = 0.10


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    return {p.relative_to(root).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size}
            for p in sorted(root.rglob('*')) if p.is_file()}


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind='stable')
    out = np.empty(len(values))
    start = 0
    while start < len(values):
        end = start+1
        while end < len(values) and abs(values[order[end]]-values[order[start]]) <= 1e-12:
            end += 1
        out[order[start:end]] = (start+end-1)/2 + 1
        start = end
    return out


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    a, b = x-x.mean(), y-y.mean()
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denom) if denom > 1e-14 else None


def fit(x, y):
    """OLS along last axis; scalar/batched fits use centered covariates."""
    xbar = np.mean(x, axis=-1, keepdims=True)
    ybar = np.mean(y, axis=-1, keepdims=True)
    xc = x-xbar
    ss = np.sum(xc*xc, axis=-1, keepdims=True)
    if np.any(ss <= 0):
        raise ValueError('Degenerate general-importance predictor')
    slope = np.sum(xc*(y-ybar), axis=-1, keepdims=True)/ss
    fitted = ybar+slope*xc
    return (ybar-slope*xbar)[..., 0], slope[..., 0], fitted, y-fitted


def isotonic(x, y):
    """Equally weighted, nondecreasing PAVA; equal x values grouped first."""
    order = np.argsort(x, kind='stable')
    blocks = []
    for index in order:
        if blocks and abs(x[index]-x[blocks[-1]['indices'][-1]]) <= 1e-12:
            blocks[-1]['indices'].append(int(index))
            blocks[-1]['total'] += y[index]
        else:
            blocks.append({'indices': [int(index)], 'total': float(y[index])})
    result = []
    for block in blocks:
        result.append(block)
        while len(result)>1 and result[-2]['total']/len(result[-2]['indices']) > result[-1]['total']/len(result[-1]['indices']):
            b, a = result.pop(), result.pop()
            result.append({'indices': a['indices']+b['indices'], 'total': a['total']+b['total']})
    prediction = np.empty(len(x))
    for block in result:
        prediction[block['indices']] = block['total']/len(block['indices'])
    return prediction


def load_model(root):
    rows = [json.loads(line) for line in (root/'observations.jsonl').read_text().splitlines()]
    conditions = {}
    for row in rows:
        assert row['status']=='complete'
        key = (row['condition'],row['example_id'])
        values = {k: float(v['value']) for k,v in row['scores'].items()}
        if key in conditions:
            assert conditions[key]['scores']==values
            assert conditions[key]['output_sha256']==row['output_sha256']
        else:
            conditions[key] = {'scores': values, 'group_id': row['group_id'], 'output_sha256': row['output_sha256']}
    def metric(condition, metric):
        return {identifier: row['scores'][metric] for (c,identifier),row in conditions.items()
                if c==condition and metric in row['scores']}
    tool_ids = sorted(metric('baseline.beginning','tool_selection'))
    ppl_ids = sorted(metric('baseline.beginning','perplexity'))
    assert len(tool_ids)==20 and len(ppl_ids)==5
    for ids in [tool_ids,ppl_ids]:
        assert len({conditions[('baseline.beginning',i)]['group_id'] for i in ids})==len(ids)
    for k in [*TARGETS,'perplexity']:
        original = metric('baseline.beginning',k)
        for c in ['baseline.middle','baseline.end','control.no-op']:
            assert metric(c,k)==original
    def array(condition,k,ids):
        by_id=metric(condition,k)
        assert set(by_id)==set(ids)
        return [by_id[i] for i in ids]
    baseline = np.array([array('baseline.beginning',k,tool_ids) for k in TARGETS])
    target = np.array([[array(f'scan.transformer.block.{l:03}',k,tool_ids) for l in range(28)] for k in TARGETS])
    ppl_base = np.array(array('baseline.beginning','perplexity',ppl_ids))
    ppl = np.array([array(f'scan.transformer.block.{l:03}','perplexity',ppl_ids) for l in range(28)])
    assert np.all(ppl>0) and np.all(ppl_base>0)
    assert np.all((target==0)|(target==1)) and np.all((baseline==0)|(baseline==1))
    return tool_ids,ppl_ids,baseline,target,ppl_base,ppl


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path(__file__).resolve().parents[1] / 'artifacts')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--draws',type=int,default=10000)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    roots=[args.artifacts/f'qwen3-{m}b' for m in MODELS]
    before={m:inventory(root) for m,root in zip(MODELS,roots)}
    data=[load_model(root) for root in roots]
    assert data[0][:2]==data[1][:2], 'Shared resampling requires same item IDs'
    tool_ids,ppl_ids=data[0][:2]
    baseline=np.stack([d[2] for d in data])
    target=np.stack([d[3] for d in data])
    ppl_base=np.stack([d[4] for d in data])
    ppl=np.stack([d[5] for d in data])
    damage=baseline.mean(-1)[:,:,None]-target.mean(-1)
    ratio=ppl.mean(-1)/ppl_base.mean(-1)[:,None]
    predictors=np.stack([np.log(ratio),ratio-1],axis=1)
    fitted=np.empty((2,3,2,28));residual=fitted.copy()
    coefficients={};loo=residual.copy();leverage=np.empty((2,2,28))
    iso=np.empty_like(damage)
    controls=[]
    for mi in range(2):
        for ki in range(3):
            iso[mi,ki]=damage[mi,ki]-isotonic(predictors[mi,0],damage[mi,ki])
            for si in range(2):
                x,y=predictors[mi,si],damage[mi,ki]
                a,b,p,e=fit(x,y)
                # Independent least-squares check with scaled covariate.
                z=(x-x.mean())/x.std()
                ref=np.linalg.lstsq(np.column_stack([np.ones(28),z]),y,rcond=None)[0]
                assert np.allclose(p,ref[0]+ref[1]*z,rtol=1e-9,atol=1e-10)
                fitted[mi,ki,si]=p;residual[mi,ki,si]=e
                leverage[mi,si]=1/28+(x-x.mean())**2/np.sum((x-x.mean())**2)
                coefficients[(mi,ki,si)]=(float(a),float(b),float(1-np.sum(e**2)/np.sum((y-y.mean())**2)))
                for layer in range(28):
                    keep=np.arange(28)!=layer
                    aa,bb,_,_=fit(x[keep],y[keep])
                    loo[mi,ki,si,layer]=y[layer]-(aa+bb*x[layer])
    # Sanity controls do not modify evidence.
    assert np.allclose(fit(np.arange(5.),2+3*np.arange(5.))[3],0)
    assert np.allclose(isotonic(np.array([0.,1.,2.]),np.array([0.,2.,1.])),[0,1.5,1.5])
    assert np.allclose(ranks([0,0,1]),[1.5,1.5,3])
    # Validate independent damage reconstruction against both historical metrics files.
    for mi,root in enumerate(roots):
        stored=json.loads((root/'metrics.json').read_text())['damage_matrix']
        for layer in range(28):
            row=stored[f'transformer.block.{layer:03}']
            for ki,k in enumerate(TARGETS):
                assert math.isclose(row[k]['absolute_damage'],damage[mi,ki,layer],abs_tol=1e-12)
            assert math.isclose(row['perplexity']['absolute_damage'],predictors[mi,1,layer],rel_tol=1e-10,abs_tol=1e-10)
    rng=np.random.default_rng(20260916)
    boot=np.empty((args.draws,2,3,2,28))
    boot_coef=np.empty((args.draws,2,3,2,2))
    for start in range(0,args.draws,250):
        n=min(250,args.draws-start)
        ti=rng.integers(0,20,(n,20));gi=rng.integers(0,5,(n,5))
        for mi in range(2):
            gy=np.take(ppl[mi],gi,axis=-1).mean(-1).T/np.take(ppl_base[mi],gi).mean(-1)[:,None]
            for ki in range(3):
                y=np.take(baseline[mi,ki],ti).mean(-1)[:,None]-np.take(target[mi,ki],ti,axis=-1).mean(-1).T
                for si,x in enumerate([np.log(gy),gy-1]):
                    a,b,_,e=fit(x,y)
                    boot[start:start+n,mi,ki,si]=e
                    boot_coef[start:start+n,mi,ki,si,0]=a
                    boot_coef[start:start+n,mi,ki,si,1]=b
    assert np.isfinite(boot).all()
    se=boot.std(axis=0,ddof=1)
    assert np.all(se>1e-12), 'Degenerate residual standard error: report before inference'
    maximum=np.max(np.abs((boot-residual)/se),axis=(1,2,3,4))
    critical=float(np.quantile(maximum,.95))
    low,high=residual-critical*se,residual+critical*se
    percentile=np.quantile(boot,[.025,.975],axis=0)
    records=[];fits=[];items=[]
    for mi,m in enumerate(MODELS):
        for ki,k in enumerate(TARGETS):
            base=float(baseline[mi,ki].mean())
            for si,spec in enumerate(SPECS):
                a,b,r2=coefficients[(mi,ki,si)]
                fits.append({'model_b':m,'metric':k,'general_proxy':spec,'alpha_probability':a,'beta_probability_per_proxy_unit':b,
                    'r_squared':r2,'spearman_damage_vs_proxy':correlation(ranks(damage[mi,ki]),ranks(predictors[mi,si])),
                    'baseline_accuracy':base,'layers':28,'unique_tool_examples':20,'unique_perplexity_documents':5,
                    'max_leverage':float(leverage[mi,si].max()),
                    'predictions_outside_possible_damage_bounds':int(np.sum((fitted[mi,ki,si]<base-1)|(fitted[mi,ki,si]>base))),
                    'alpha_bootstrap_pointwise_low':float(np.quantile(boot_coef[:,mi,ki,si,0],.025)),
                    'alpha_bootstrap_pointwise_high':float(np.quantile(boot_coef[:,mi,ki,si,0],.975)),
                    'beta_bootstrap_pointwise_low':float(np.quantile(boot_coef[:,mi,ki,si,1],.025)),
                    'beta_bootstrap_pointwise_high':float(np.quantile(boot_coef[:,mi,ki,si,1],.975))})
                for layer in range(28):
                    index=(mi,ki,si,layer)
                    records.append({'model_b':m,'metric':k,'general_proxy':spec,'layer':layer,
                        'baseline_accuracy':base,'intervention_accuracy':float(target[mi,ki,layer].mean()),
                        'tool_damage_probability':float(damage[mi,ki,layer]),
                        'baseline_perplexity_arithmetic_mean':float(ppl_base[mi].mean()),
                        'intervention_perplexity_arithmetic_mean':float(ppl[mi,layer].mean()),
                        'perplexity_relative_damage':float(predictors[mi,1,layer]),'perplexity_log_ratio':float(predictors[mi,0,layer]),
                        'predicted_tool_damage_probability':float(fitted[index]),'residual_probability':float(residual[index]),
                        'residual_pointwise_95_low':float(percentile[(0,*index)]),'residual_pointwise_95_high':float(percentile[(1,*index)]),
                        'residual_simultaneous_95_low':float(low[index]),'residual_simultaneous_95_high':float(high[index]),
                        'residual_bootstrap_se':float(se[index]),'leverage':float(leverage[mi,si,layer]),
                        'leave_one_layer_out_residual':float(loo[index]),'isotonic_log_residual':float(iso[mi,ki,layer]),
                        'damage_rank_descending_average_ties':float(ranks(-damage[mi,ki])[layer]),
                        'residual_rank_descending_average_ties':float(ranks(-residual[mi,ki,si])[layer]),
                        'equal_damage_layers':int(np.sum(np.abs(damage[mi,ki]-damage[mi,ki,layer])<=1e-12)),
                        'intervention_complete_failure':bool(np.all(target[mi,ki,layer]==0)),
                        'baseline_at_ceiling':bool(base==1),
                        'point_residual_at_least_10pp':bool(residual[index]>=MATERIAL-1e-12),
                        'simultaneous_low_above_zero':bool(low[index]>0),
                        'simultaneous_low_above_10pp':bool(low[index]>MATERIAL),
                        'raw_log_and_loo_point_material':bool(np.all(residual[mi,ki,:,layer]>=MATERIAL-1e-12) and np.all(loo[mi,ki,:,layer]>=MATERIAL-1e-12)),
                        'isotonic_point_material':bool(iso[mi,ki,layer]>=MATERIAL-1e-12)})
            for layer in range(28):
                for ii,identifier in enumerate(tool_ids):
                    items.append({'model_b':m,'metric':k,'layer':layer,'example_id':identifier,
                        'baseline_score':float(baseline[mi,ki,ii]),'intervention_score':float(target[mi,ki,layer,ii]),
                        'paired_damage':float(baseline[mi,ki,ii]-target[mi,ki,layer,ii])})
    general=[]
    for mi,m in enumerate(MODELS):
        for layer in range(28):
            for ii,identifier in enumerate(ppl_ids):
                general.append({'model_b':m,'layer':layer,'example_id':identifier,
                                'baseline_perplexity':float(ppl_base[mi,ii]),'intervention_perplexity':float(ppl[mi,layer,ii])})
    comparisons=[]
    for ki,k in enumerate(TARGETS):
        for si,spec in enumerate(SPECS):
            a=set(np.flatnonzero(residual[0,ki,si]>=MATERIAL-1e-12).tolist())
            b=set(np.flatnonzero(residual[1,ki,si]>=MATERIAL-1e-12).tolist())
            comparisons.append({'metric':k,'general_proxy':spec,
                'residual_spearman_across_sizes':correlation(ranks(residual[0,ki,si]),ranks(residual[1,ki,si])),
                'material_layers_0_6':';'.join(map(str,sorted(a))),'material_layers_1_7':';'.join(map(str,sorted(b))),
                'material_layers_common':';'.join(map(str,sorted(a&b))),'material_set_jaccard':len(a&b)/len(a|b) if a|b else '',
                'scope':'descriptive same-family aligned block index; not independent replication'})
    write_csv(args.output/'layer_residuals.csv',records)
    for m in MODELS:
        write_csv(args.output/f'qwen_{m.replace(".","_")}b_layer_residuals.csv',[r for r in records if r['model_b']==m])
    write_csv(args.output/'regression_fits.csv',fits)
    write_csv(args.output/'paired_tool_observations.csv',items)
    write_csv(args.output/'perplexity_observations.csv',general)
    write_csv(args.output/'cross_size_descriptive.csv',comparisons)
    summary={'status':'exploratory_zero_inference','seed':20260916,'bootstrap_draws':args.draws,
        'simultaneous_family_residuals':336,'simultaneous_critical_value':critical,
        'simultaneous_method':'95% centered standardized bootstrap maximum absolute deviation, shared item resampling',
        'material_residual_probability':MATERIAL,'minimum_bootstrap_standard_error':float(se.min()),
        'zero_variance_residuals':int(np.sum(se<=1e-12)),
        'models':{},'cross_size':comparisons,'fits':fits,
        'validation_interventions_used':0,'abstention_reasoning_format_used':False,
        'source_before':before,'python_version':sys.version,'numpy_version':np.__version__}
    for mi,m in enumerate(MODELS):
        summary['models'][m]={}
        for k in TARGETS:
            rs=[r for r in records if r['model_b']==m and r['metric']==k and r['general_proxy']=='log_ratio']
            summary['models'][m][k]={
                'point_material_layers':[r['layer'] for r in rs if r['point_residual_at_least_10pp']],
                'simultaneous_positive_layers':[r['layer'] for r in rs if r['simultaneous_low_above_zero']],
                'simultaneous_material_layers':[r['layer'] for r in rs if r['simultaneous_low_above_10pp']],
                'raw_log_loo_material_layers':[r['layer'] for r in rs if r['raw_log_and_loo_point_material']],
                'isotonic_material_layers':[r['layer'] for r in rs if r['isotonic_point_material']],
                'top_residuals':sorted(rs,key=lambda r:r['residual_probability'],reverse=True)[:5]}
    robustness={}
    for mi,m in enumerate(MODELS):
        robustness[m]={}
        sets=[]
        for ki,k in enumerate(TARGETS):
            selected=[layer for layer in range(28)
                      if np.all(low[mi,ki,:,layer]>MATERIAL)
                      and np.all(loo[mi,ki,:,layer]>=MATERIAL-1e-12)
                      and iso[mi,ki,layer]>=MATERIAL-1e-12]
            robustness[m][k]=selected
            sets.append(set(selected))
        robustness[m]['all_three']=sorted(set.intersection(*sets))
    write_json(args.output/'robustness-summary.json',robustness)
    # Store the resampling result without inputs requiring any model package.
    np.savez_compressed(args.output/'residual_bootstrap.npz',residuals=boot,maximum_standardized_deviation=maximum)
    plot(args.output,records,predictors,damage,residual,low,high,iso,coefficients)
    after={m:inventory(root) for m,root in zip(MODELS,roots)}
    assert before==after, 'Source evidence changed'
    forbidden=[m for m in sys.modules if m.split('.')[0] in {'torch','transformers','mlx','huggingface_hub','capability_anatomy','qca_phase0'}]
    assert not forbidden,forbidden
    summary['source_unchanged']=True
    summary['model_libraries_loaded']=forbidden
    write_json(args.output/'results.json',summary)
    write_json(args.output/'verification.json',{'source_files_checked':sum(len(x) for x in before.values()),
        'source_unchanged':True,'model_libraries_loaded':forbidden,'ols_matches_independent_lstsq':True,
        'historical_damage_matches':True,'deterministic_repetitions_collapsed':True,
        'unique_tool_items':len(tool_ids),'unique_perplexity_documents':len(ppl_ids),
        'regression_control':'T=2+3G exact fit','rank_control':'ties average','isotonic_control':'[0,2,1] -> [0,1.5,1.5]'})
    print(json.dumps({'critical_value':critical,'models':{m:{k:{a:b for a,b in s.items() if a!='top_residuals'} for k,s in v.items()} for m,v in summary['models'].items()},'cross_size':comparisons},indent=2))


def plot(output,records,predictors,damage,residual,low,high,iso,coefs):
    os.environ.setdefault('MPLCONFIGDIR',str(output/'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'ca-specificity-20260916'})
    labels=('Tool selection','Argument binding','Full call')
    colors=('#2962a3','#bb4b26')
    def save(fig,name):
        fig.savefig(output/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(output/(name+'.svg'),bbox_inches='tight',metadata={'Date': None})
        plt.close(fig)
    for mi,m in enumerate(MODELS):
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        for ki,label in enumerate(labels):
            x,y=predictors[mi,0],damage[mi,ki]
            a,b,_=coefs[(mi,ki,0)]
            ax=axes[0,ki];ax.scatter(x,y,c=colors[mi],s=32,alpha=.85)
            xx=np.linspace(x.min(),x.max(),200);ax.plot(xx,a+b*xx,color='#222222',lw=1.4,label='OLS')
            order=np.argsort(x);ax.plot(x[order],(y-iso[mi,ki])[order],color='#8f8f8f',lw=1.1,ls='--',label='Monotone shape check')
            # The small model has a dense tied collapse cluster: do not overlay
            # several labels at effectively the same plotted coordinates.
            selected=set(np.argsort(residual[mi,ki,0])[-(1 if mi==0 else 4):].tolist())|{int(np.argmax(x))}
            for layer in sorted(selected):ax.annotate(str(layer),(x[layer],y[layer]),xytext=(4,5),textcoords='offset points',fontsize=8)
            ax.set(title=label,xlabel='log(PPL intervention / intact)',ylabel='Tool damage (rate units)')
            ax.axhline(0,color='#cccccc',lw=.7);ax.legend(fontsize=8,loc='best')
            ax=axes[1,ki];e=residual[mi,ki,0];lower=low[mi,ki,0];upper=high[mi,ki,0]
            ax.errorbar(np.arange(28),e,yerr=np.stack([e-lower,upper-e]),fmt='o',ms=3.5,c=colors[mi],ecolor='#a4b6c9',elinewidth=.8,capsize=1.5)
            material=lower>MATERIAL
            if material.any():ax.scatter(np.flatnonzero(material),e[material],marker='s',facecolors='none',edgecolors='#111111',s=75,zorder=4,label='Joint lower bound > 0.10')
            ax.axhline(0,color='#444444',lw=.8);ax.axhline(MATERIAL,color='#b2342d',ls='--',lw=1,label='Material residual = 0.10 (10 pp)')
            ax.set(xlabel='Block index',ylabel='OLS residual (rate units)',xticks=[0,4,8,12,16,20,24,27])
            ax.legend(fontsize=7,loc='best')
        fig.suptitle(f'Qwen3-{m}B: tool damage conditional on perplexity damage\nExploratory log-ratio fit; residual bands jointly cover 336 comparisons approximately',fontsize=13)
        save(fig,f'qwen_{m.replace(".","_")}b_specificity')
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for mi,m in enumerate(MODELS):
        for ki,label in enumerate(labels):
            ax=axes[mi,ki];x=residual[mi,ki,1];y=residual[mi,ki,0]
            ax.scatter(x,y,c=colors[mi],s=30)
            lim=[min(x.min(),y.min())-.05,max(x.max(),y.max())+.05]
            ax.plot(lim,lim,c='#999999',ls='--',lw=1)
            for layer in np.argsort(np.abs(x-y))[-3:]:
                offset=(-16,-14) if layer==0 else (-16,10) if layer==1 else (4,5)
                ax.annotate(str(layer),(x[layer],y[layer]),xytext=offset,textcoords='offset points',fontsize=8)
            ax.axvline(.1,c='#b2342d',ls=':',lw=1);ax.axhline(.1,c='#b2342d',ls=':',lw=1)
            ax.set(title=f'{m}B — {label}',xlabel='Raw-relative PPL OLS residual',ylabel='Log-ratio PPL OLS residual')
    fig.suptitle('Specification sensitivity: each dot is a layer; dotted lines mark 0.10 residual',fontsize=13)
    save(fig,'raw_vs_log_sensitivity')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for ki,label in enumerate(labels):
        ax=axes[ki]
        for mi,m in enumerate(MODELS):ax.plot(np.arange(28),residual[mi,ki,0],marker='o',ms=3,c=colors[mi],label=f'Qwen3-{m}B')
        ax.axhline(0,c='#999999',lw=.7);ax.axhline(.1,c='#b2342d',lw=1,ls=':')
        ax.set(title=label,xlabel='Aligned block index',ylabel='Log-ratio OLS residual',xticks=[0,4,8,12,16,20,24,27]);ax.legend(fontsize=8)
    fig.suptitle('Descriptive cross-size comparison — same family, shared examples; no independent replication claim',fontsize=12)
    save(fig,'cross_size_residuals')


if __name__=='__main__':
    main()
