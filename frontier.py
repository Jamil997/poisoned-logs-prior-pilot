import json,time
import numpy as np
from experiment import *

config=json.loads((ROOT/'protocol.json').read_text());p=json.loads((ROOT/'frontier_protocol.json').read_text());out=ROOT/'frontier_results';out.mkdir(exist_ok=True)
rows=[];ws=[];start=time.perf_counter()
for name in p['environments']:
    env=setup(config,name,p['eta'])
    for seed in range(p['seeds']['start'],p['seeds']['start']+p['seeds']['count']):
        counts=log_counts(config['n'],env,seed,p['eta'],name)
        for eps in p['epsilons']:
            ka=int(config['n']*eps)
            for scale in p['budget_multipliers']:
                kd=int(scale*ka);gate=make_gate(counts,env,'robust_centered_dr_b',config['alpha'],kd)
                a=worst_attack(counts,gate,ka,env['truth']);delta=value(a['selected'],env['truth'])
                row=dict(environment=name,eta=p['eta'],epsilon=eps,seed=seed,method=f'budget_x{scale:g}',multiplier=scale,
                         selected=a['selected'],improvement=delta,harm=int(delta<-1e-12),edits=int(np.abs(a['net']).sum()),
                         calls=a['calls'],max_gap=a['max_gap'],unresolved_lower_targets=a['unresolved_lower_targets'])
                rows.append(row);ws.append(dict(**row,net=a['net'].tolist(),scores=a['scores'].tolist(),targets=a['targets']))
        if (seed-p['seeds']['start']+1)%10==0:
            write_csv(out/'runs.csv',rows);print(f'frontier {name} seed={seed} elapsed={time.perf_counter()-start:.1f}s',flush=True)
ss=summarize(rows);write_csv(out/'runs.csv',rows);write_csv(out/'summary.csv',ss);(out/'summary.json').write_text(json.dumps(ss,indent=2));(out/'witnesses.json').write_text(json.dumps(ws,separators=(',',':')))
agg=[]
for scale in p['budget_multipliers']:
    rr=[r for r in rows if r['multiplier']==scale];cells=[r for r in ss if r['method']==f'budget_x{scale:g}']
    agg.append(dict(multiplier=scale,max_cell_harm=max(r['harm_rate'] for r in cells),mean_improvement=float(np.mean([r['improvement'] for r in rr])),
                    harmful=sum(r['harm'] for r in rr),outcomes=len(rr),unresolved=sum(r['unresolved_lower_targets']>0 for r in rr)))
for a in agg:
    a['empirically_nondominated']=not any(b['max_cell_harm']<=a['max_cell_harm'] and b['mean_improvement']>=a['mean_improvement'] and (b['max_cell_harm']<a['max_cell_harm'] or b['mean_improvement']>a['mean_improvement']) for b in agg)
(out/'frontier.json').write_text(json.dumps(agg,indent=2));(out/'metadata.json').write_text(json.dumps(dict(runtime_seconds=time.perf_counter()-start,outcomes=len(rows)),indent=2));print(json.dumps(agg,indent=2))
