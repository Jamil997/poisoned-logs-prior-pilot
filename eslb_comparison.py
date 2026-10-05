"""Published ESLB implementation + an affine reward-attack adapter.

The upstream class definitions are loaded unchanged. See vendor/.../LICENSE.
Only unrelated dataset/model imports are avoided; no estimator formula edited.
"""
import ast,importlib.util,logging,math,enum,abc,subprocess
import json,time,hashlib
from pathlib import Path
import numpy as np
from experiment import ROOT,setup,log_counts,worst_attack,value,write_csv,summarize,corrected_scores

VENDOR=ROOT/'vendor/offpolicy_selection_eslb'
def load_eslb():
    source=(VENDOR/'estimators.py').read_text();tree=ast.parse(source)
    nodes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in ['Estimator','ESLBBiasType','ESLB']]
    assert len(nodes)==3
    spec=importlib.util.spec_from_file_location('upstream_eslb_utils',VENDOR/'utils.py');utils=importlib.util.module_from_spec(spec);spec.loader.exec_module(utils)
    ns=dict(np=np,abc=abc,enum=enum,math=math,logging=logging,utils=utils)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(VENDOR/'estimators.py'),'exec'),ns)
    return ns['ESLB'],ns['ESLBBiasType']

ESLB,Bias=load_eslb()
def build_gate(counts,env,alpha,mc_seed,iterations,batch):
    H=len(counts);nh=counts.sum(axis=1);n=int(nh.sum());M=len(env['pi'])
    # Random row order independent of reward labels (counts are sufficient for
    # our gates but the official sequential variance proxy uses row order).
    group=np.repeat(np.arange(H),nh);rng=np.random.default_rng(mc_seed);rng.shuffle(group)
    x=group//2;actions=group%2;b_probs=env['mu'][x]
    probs=np.vstack([np.zeros(env['X']),env['pi']]);parts=[]
    for j,pi1 in enumerate(probs):
        t_probs=np.stack([1-pi1[x],pi1[x]],axis=1)
        model=ESLB(delta=alpha/(M+1),n_iterations=iterations,n_batch_size=batch,bias_type=Bias.Bernstein)
        np.random.seed(mc_seed+100*j)
        result=model(t_probs,b_probs,actions,np.zeros(n))
        w=np.stack([1-pi1,pi1],axis=1).ravel()/env['mu'].ravel()
        denom=float(w@nh);coefficient=result['mult_bias']*w/denom
        constant=-result['mult_bias']*(result['concentration']+1/np.sqrt(n))-result['concentration_of_contexts']
        parts.append(dict(coefficient=coefficient,constant=constant,mult_bias=result['mult_bias'],
                          concentration=result['concentration'],context_term=result['concentration_of_contexts']))
    base=parts[0];c=np.array([n*(p['coefficient']-base['coefficient']) for p in parts[1:]])
    offset=np.array([np.repeat(p['constant']+base['constant']+base['mult_bias']-1,H) for p in parts[1:]])
    gate=dict(c=c,offset=offset,rad=np.zeros(M),k=0,n=n,name='sn_eslb')
    return gate,parts,group

def validate_linearization(config,secondary):
    env=setup(config,secondary['environment'],secondary['eta']);counts=log_counts(config['n'],env,2000,.05,secondary['environment'])
    gate,parts,group=build_gate(counts,env,.05,902000,secondary['mc_iterations'],secondary['mc_batch_size'])
    n=gate['n'];x=group//2;a=group%2;rng=np.random.default_rng(17);rewards=rng.random(n);rewards[rng.random(n)<.5]=0
    scores=gate['c'][:,group]@rewards/n+gate['offset'][:,group].mean(axis=1)
    raw=[]
    for j,pi1 in enumerate(np.vstack([np.zeros(env['X']),env['pi']])):
        t=np.stack([1-pi1[x],pi1[x]],axis=1);model=ESLB(delta=.05/7,n_iterations=secondary['mc_iterations'],n_batch_size=secondary['mc_batch_size'],bias_type=Bias.Bernstein)
        np.random.seed(902000+100*j)
        rr=model(t,env['mu'][x],a,1-rewards if j==0 else rewards)
        raw.append(rr['lower_bound'])
    clipped=np.array(raw[1:])+raw[0]-1
    from experiment import choose
    assert choose(scores)==choose(clipped)
    # All positive scores must agree; raw unclipped scores can be more negative.
    assert np.allclose(scores[scores>0],clipped[scores>0],atol=1e-12)
    # Direct raw formula with complement baseline, checked for every policy.
    base=parts[0];direct=np.array([p['coefficient'][group]@rewards+p['constant']+base['coefficient'][group]@(1-rewards)+base['constant']-1 for p in parts[1:]])
    assert np.allclose(scores,direct,atol=1e-12)
    return float(np.max(np.abs(scores-direct)))

def run():
    config=json.loads((ROOT/'protocol.json').read_text());secondary=json.loads((ROOT/'eslb_protocol.json').read_text())
    out=ROOT/'eslb_results';out.mkdir(exist_ok=True);start=time.perf_counter()
    error=validate_linearization(config,secondary);rows=[];witness=[];gate_cache=[]
    env=setup(config,secondary['environment'],secondary['eta'])
    for seed in range(secondary['seeds']['start'],secondary['seeds']['start']+secondary['seeds']['count']):
        counts=log_counts(config['n'],env,seed,secondary['eta'],secondary['environment'])
        gate,parts,group=build_gate(counts,env,config['alpha'],900000+seed,secondary['mc_iterations'],secondary['mc_batch_size'])
        gate_cache.append(dict(seed=seed,c=gate['c'].tolist(),offset=gate['offset'].tolist(),counts=counts.tolist()))
        for eps in secondary['epsilons']:
            k=int(config['n']*eps)
            for method in secondary['methods']:
                actual=dict(gate,name=method,k=k if method.startswith('robust') else 0)
                attack=worst_attack(counts,actual,k,env['truth']);delta=value(attack['selected'],env['truth'])
                row=dict(environment=secondary['environment'],eta=secondary['eta'],epsilon=eps,seed=seed,method=method,
                         selected=attack['selected'],improvement=delta,harm=int(delta<-1e-12),edits=int(np.abs(attack['net']).sum()),
                         calls=attack['calls'],attack_seconds=attack['seconds'],max_gap=attack['max_gap'],unresolved_lower_targets=attack['unresolved_lower_targets'])
                rows.append(row);witness.append(dict(**row,net=attack['net'].tolist(),scores=attack['scores'].tolist(),targets=attack['targets']))
        if (seed-secondary['seeds']['start']+1)%10==0:
            write_csv(out/'runs.csv',rows);print(f'ESLB seed={seed} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    write_csv(out/'runs.csv',rows);ss=summarize(rows);write_csv(out/'summary.csv',ss);(out/'summary.json').write_text(json.dumps(ss,indent=2))
    (out/'gates.json').write_text(json.dumps(gate_cache,separators=(',',':')));(out/'witnesses.json').write_text(json.dumps(witness,separators=(',',':')))
    metadata=dict(runtime_seconds=time.perf_counter()-start,upstream_commit=(VENDOR/'UPSTREAM_COMMIT.txt').read_text().strip(),
                  upstream_estimators_sha256=hashlib.sha256((VENDOR/'estimators.py').read_bytes()).hexdigest(),
                  linearization_max_error=error,unresolved_outcomes=sum(r['unresolved_lower_targets']>0 for r in rows),
                  outcomes=len(rows),protocol_sha256=hashlib.sha256((ROOT/'eslb_protocol.json').read_bytes()).hexdigest())
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata,indent=2))

if __name__=='__main__':run()
