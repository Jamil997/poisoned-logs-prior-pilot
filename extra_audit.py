import json
import numpy as np
from experiment import *
config=json.loads((ROOT/'protocol.json').read_text());records=[];max_error=0.
# Published implementation supplement, using saved reward-independent coefficients.
gates={g['seed']:g for g in json.loads((ROOT/'eslb_results/gates.json').read_text())}
for w in json.loads((ROOT/'eslb_results/witnesses.json').read_text()):
    saved=gates[w['seed']];counts=np.array(saved['counts']);n=int(counts.sum());k=int(n*w['epsilon']) if w['method'].startswith('robust') else 0
    gate=dict(c=np.array(saved['c']),offset=np.array(saved['offset']),rad=np.zeros(6),n=n,k=k,name=w['method'])
    records.append((w,counts,gate,int(n*w['epsilon'])))
for w in json.loads((ROOT/'frontier_results/witnesses.json').read_text()):
    env=setup(config,w['environment'],w['eta']);counts=log_counts(config['n'],env,w['seed'],w['eta'],w['environment']);ka=int(config['n']*w['epsilon']);kd=int(w['multiplier']*ka)
    gate=make_gate(counts,env,'robust_centered_dr_b',config['alpha'],kd);records.append((w,counts,gate,ka))
for w,counts,gate,ka in records:
    net=np.array(w['net']);assert np.abs(net).sum()<=ka;dirty=poison(counts,net);n=gate['n'];k=gate['k']
    groups=np.repeat(np.repeat(np.arange(len(counts)),2),dirty.ravel());r=np.repeat(np.tile([0,1],len(counts)),dirty.ravel());c=gate['c'][:,groups]
    s=(c*r+gate['offset'][:,groups]).mean(axis=1)-gate['rad']
    if k:
        d=c*r-np.minimum(0,c);s-=np.partition(d,n-k,axis=1)[:,-k:].sum(axis=1)/n
    err=float(np.max(np.abs(s-w['scores'])));max_error=max(max_error,err);assert err<1e-9
    assert choose(s)==w['selected'];assert w['unresolved_lower_targets']==0
out=dict(passed=True,additional_row_reconstructions=len(records),max_error=max_error,unresolved=0)
(ROOT/'extra_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
