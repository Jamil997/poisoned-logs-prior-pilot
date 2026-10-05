import json,csv,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
from experiment import *
from continuous_certificate import certify

config=json.loads((ROOT/'protocol.json').read_text());ws=json.loads((ROOT/'results/witnesses.json').read_text())
maxerr=0.;tie_resolutions=[];checked=0
for w in ws:
    env=setup(config,w['environment'],w['eta']);counts=np.array(w['clean_counts']);net=np.array(w['net']);n=int(counts.sum());k=int(n*w['epsilon'])
    assert np.abs(net).sum()<=k;dirty=poison(counts,net);gate=make_gate(counts,env,w['method'],config['alpha'],k)
    # Independent per-row construction (without grouped top-k implementation).
    group=np.repeat(np.repeat(np.arange(len(counts)),2),dirty.ravel());r=np.repeat(np.tile([0,1],len(counts)),dirty.ravel())
    cc=gate['c'][:,group];direct=(cc*r+gate['offset'][:,group]).mean(axis=1)-gate['rad']
    if gate['k']:
        d=cc*r-np.minimum(0,cc);direct-=np.partition(d,n-gate['k'],axis=1)[:,-gate['k']:].sum(axis=1)/n
    err=float(np.max(np.abs(direct-np.array(w['scores']))));maxerr=max(maxerr,err);assert err<1e-8
    assert choose(direct)==w['selected'];assert abs(value(w['selected'],env['truth'])-w['improvement'])<1e-12
    if w['unresolved_lower_targets']:
        # In these two rows only target 5 is tied. Candidate 4 wins every tie.
        assert w['environment']=='diverse_matched_gain' and w['eta']==.5 and w['epsilon']==.005 and w['seed']==2038
        assert w['method'] in ['centered_dr_b','robust_centered_dr_b']
        diff=gate['c'][5]-gate['c'][4];off=gate['offset'][5]-gate['offset'][4]
        clean_numerator=int(round(diff@counts[:,1]+off@counts.sum(axis=1)))
        gains=diff[group]*(1-r) if False else None
        # Original rows, to independently maximize pairwise score difference.
        cg=np.repeat(np.repeat(np.arange(len(counts)),2),counts.ravel());cr=np.repeat(np.tile([0,1],len(counts)),counts.ravel())
        impacts=np.maximum(0,diff[cg])-diff[cg]*cr
        max_increase=int(round(np.sort(impacts)[-k:].sum()))
        assert clean_numerator==-500 and max_increase==500
        assert abs(gate['rad'][5]-gate['rad'][4])<1e-14
        if gate['k']:
            # Both top-k corrections remain exactly 2*k/n for any k edits.
            for j in [4,5]:
                c=gate['c'][j];available=np.where(c>0,counts[:,1],np.where(c<0,counts[:,0],0)).sum()
                assert available-k>=k and set(np.abs(c)).issubset({0.,2.})
        tie_resolutions.append(dict(environment=w['environment'],eta=w['eta'],epsilon=w['epsilon'],seed=w['seed'],method=w['method'],
                                    unreachable_target=5,dominating_candidate=4,max_pairwise_score_difference=0.,reason='Candidate 5 cannot strictly beat lower-index candidate 4; exact integer numerator bound -500+500=0.'))
    checked+=1

# Verify the real-valued envelope on random fractional attacks in the primary cell.
rng=np.random.default_rng(815);env=setup(config,'diverse_matched_gain',.05);fractional_checks=0
for seed in range(2000,2100):
    counts=log_counts(50000,env,seed,.05,'diverse_matched_gain');gate=make_gate(counts,env,'robust_centered_dr_b',.05,250)
    bound,lower,upper=certify(counts,gate,env['truth'],250)
    group=np.repeat(np.repeat(np.arange(len(counts)),2),counts.ravel());r=np.repeat(np.tile([0.,1.],len(counts)),counts.ravel())
    cc=gate['c'][:,group];base=gate['offset'][:,group]
    for _ in range(5):
        rr=r.copy();ix=rng.choice(50000,250,replace=False);rr[ix]=rng.random(250)
        d=cc*rr-np.minimum(0,cc)
        scores=(cc*rr+base).mean(axis=1)-np.partition(d,50000-250,axis=1)[:,-250:].sum(axis=1)/50000-gate['rad']
        assert np.all(scores>=lower-1e-10) and np.all(scores<=upper+1e-10)
        assert value(choose(scores),env['truth'])>=bound-1e-12;fractional_checks+=1

primary=[r for r in ws if r['environment']=='diverse_matched_gain' and r['eta']==.05 and r['epsilon']==.005]
supp=json.loads((ROOT/'eslb_results/witnesses.json').read_text());primary += [r for r in supp if r['epsilon']==.005]
by_method={}
for r in primary:by_method.setdefault(r['method'],{})[r['seed']]=r
cert={r['seed']:r['guaranteed_true_improvement'] for r in json.loads((ROOT/'results/continuous_certificate.json').read_text())}
comparisons=[]
for method,rs in by_method.items():
    if method=='robust_centered_dr_b':continue
    for axis in ['binary_worst_value','continuous_certificate_lower_bound']:
        diff=np.array([(by_method['robust_centered_dr_b'][s]['improvement'] if axis=='binary_worst_value' else cert[s])-rs[s]['improvement'] for s in sorted(rs)])
        radius=float(student_t.ppf(.975,len(diff)-1)*diff.std(ddof=1)/np.sqrt(len(diff)))
        comparisons.append(dict(comparator=method,axis=axis,paired_mean_gain=float(diff.mean()),ci_low=float(diff.mean()-radius),ci_high=float(diff.mean()+radius)))
res=dict(passed=True,row_reconstructions=checked,max_row_error=maxerr,reported_ambiguous_ties=len(tie_resolutions),
         tie_resolutions=tie_resolutions,remaining_ambiguous_ties=0,random_fractional_checks=fractional_checks,
         paired_primary_comparisons=comparisons,interval_scope='Pointwise paired Student-t intervals. Not simultaneous across baselines. A bootstrap/Holm sensitivity check is in analysis.json.')
(ROOT/'results/audit.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
