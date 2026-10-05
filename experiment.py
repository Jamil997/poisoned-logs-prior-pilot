"""Second, fixed-protocol pilot. No reward oracle is provided to defenders.

All gates are affine in rewards before an optional exact edit correction.
The attack searches the complete binary-flip space using group counts.
Centered DR is DR with the fixed (deliberately misspecified) qhat=0.5.
"""
from pathlib import Path
import argparse,csv,hashlib,json,platform,time,warnings
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.stats import beta as beta_dist,t as student_t

ROOT=Path(__file__).resolve().parent
TOL=1e-9; OPT_TOL=1e-6
warnings.filterwarnings('ignore',message='Unrecognized options detected:.*mip_abs_gap.*',category=RuntimeWarning)

def setup(config,name,eta):
    env=config['environments'][name]; px=np.array(env['context_probabilities']);X=len(px)
    mu1=np.array([eta if v=='eta' else v for v in env['logger_action1']]);mu=np.stack([1-mu1,mu1],axis=1)
    q=np.stack([env['action0_means'],env['action1_means']],axis=1)
    if 'candidate_betas' in env:
        pi=.5+np.array(env['candidate_betas'])[:,None]*np.array([1.,-1.])
    else:pi=np.array(env['candidate_action1_probabilities'],dtype=float)
    truth=(pi*(q[:,1]-q[:,0])*px).sum(axis=1)
    joint=(px[:,None]*mu)[:,:,None]*np.stack([1-q,q],axis=-1)
    return dict(px=px,mu=mu,q=q,pi=pi,truth=truth,joint=joint.reshape(-1,2),X=X)

def log_counts(n,env,seed,eta,name):
    rng=np.random.default_rng(np.random.SeedSequence([seed,round(eta*10000),1 if name=='diverse_matched_gain' else 2]))
    return rng.multinomial(n,env['joint'].ravel()).reshape(-1,2)

def make_gate(counts,env,method,alpha,k):
    n=int(counts.sum()); nh=counts.sum(axis=1);X=env['X']; pi=env['pi'];M=len(pi)
    delta=np.stack([-pi,pi],axis=2);c=(delta/env['mu']).reshape(M,-1)
    offset=np.zeros_like(c);t=np.log(M/alpha)
    robust=method.startswith('robust_')
    if method.endswith('ips_h'):
        B=np.maximum(0,c.max(axis=1))-np.minimum(0,c.min(axis=1))
        rad=B*np.sqrt(t/(2*n))
    elif method.endswith('centered_dr_b'):
        offset=-.5*c
        V=(.25*env['mu']*(c.reshape(M,X,2)**2)).sum(axis=2).max(axis=1)
        B=np.abs(c).max(axis=1)
        a=B*t/(3*n)
        # Positive root of n*r^2 >= 2*t*(V+B*r/3).
        rad=a+np.sqrt(a*a+2*V*t/n)
    elif method.endswith('tabular_h'):
        nxa=nh.reshape(X,2); nx=nxa.sum(axis=1)
        c=(delta*nx[None,:,None]/np.maximum(nxa,1)[None,:,:]).reshape(M,-1)
        t=np.log(2*M/alpha)
        reward_radius=np.sqrt(t*((c*c)@nh)/(2*n*n))
        context_radius=2*pi.max(axis=1)*np.sqrt(t/(2*n))
        rad=reward_radius+context_radius
        invalid=np.any((pi>0)&np.any(nxa==0,axis=1)[None,:],axis=1)
        c[invalid]=0;rad[invalid]=1e6
    elif method=='retain':
        c*=0;rad=np.ones(M)*1e6
    else:raise ValueError(method)
    return dict(c=c,offset=offset,rad=rad,k=k if robust else 0,n=n,name=method)

def corrected_scores(counts,gate):
    c=gate['c'];n=gate['n'];nh=counts.sum(axis=1)
    scores=(c@counts[:,1]+gate['offset']@nh)/n-gate['rad']
    if gate['k']:
        for j,cj in enumerate(c):
            available=np.where(cj>=0,counts[:,1],counts[:,0]);left=gate['k'];damage=0.
            for h in np.argsort(-np.abs(cj),kind='stable'):
                take=min(left,int(available[h]));damage+=take*abs(cj[h]);left-=take
            scores[j]-=damage/n
    return scores

def choose(scores):
    j=int(np.argmax(scores));return j if scores[j]>TOL else -1

def value(j,truth):return 0. if j==-1 else float(truth[j])

def poison(counts,net):
    net=np.asarray(net,dtype=int);result=counts.copy()
    assert np.all(net<=counts[:,0]) and np.all(-net<=counts[:,1])
    result[:,1]+=net;result[:,0]-=net;assert np.all(result>=0)
    return result

def branches(counts,gate):
    c=gate['c'];n=gate['n'];nh=counts.sum(axis=1);out=[]
    for j,cj in enumerate(c):
        base=cj@counts[:,1]+gate['offset'][j]@nh-n*gate['rad'][j]
        if not gate['k']:out.append([(float(base),cj)]);continue
        avail=np.where(cj>=0,counts[:,1],counts[:,0]); bb=[]
        for lam in np.unique(np.r_[0.,np.abs(cj)]):
            exc=np.maximum(np.abs(cj)-lam,0)
            bb.append((float(base-gate['k']*lam-avail@exc),cj-np.sign(cj)*exc))
        out.append(bb)
    return out

def target_attack(counts,gate,ka,target,branch_cache=None):
    H=len(counts);n=gate['n'];bb=branch_cache if branch_cache is not None else branches(counts,gate)
    target_branches=[(0.,np.zeros(H))] if target==-1 else bb[target]
    best=None;upper=-np.inf;calls=0;maxgap=0.
    for intercept,slopes in target_branches:
        rows=[np.r_[np.zeros(H),np.ones(H),0.]];rhs=[ka]
        for h in range(H):
            for direction in [-1.,1.]:
                row=np.zeros(2*H+1);row[h]=direction;row[H+h]=-1;rows.append(row);rhs.append(0.)
        if target!=-1:rows.append(np.r_[-slopes,np.zeros(H),1.]);rhs.append(intercept)
        for other,bb2 in enumerate(bb):
            if other==target:continue
            for intercept2,slopes2 in bb2:
                rows.append(np.r_[slopes2-slopes,np.zeros(H),1.]);rhs.append(intercept-intercept2)
        result=milp(c=np.r_[np.zeros(2*H),-1.],integrality=np.r_[np.ones(H),np.zeros(H+1)],
                    bounds=Bounds(np.r_[-counts[:,1],np.zeros(H),-np.inf],np.r_[counts[:,0],np.repeat(ka,H),np.inf]),
                    constraints=LinearConstraint(np.array(rows),-np.inf,np.array(rhs)),
                    options={'mip_rel_gap':0.,'mip_abs_gap':n*OPT_TOL,'time_limit':10.})
        calls+=1
        if result.x is None:raise RuntimeError(f'No attack witness: {result.message}')
        net=np.round(result.x[:H]).astype(int);assert np.abs(net).sum()<=ka
        scores=corrected_scores(poison(counts,net),gate);sel=choose(scores)
        competitors=np.delete(scores,target) if target!=-1 else scores
        actual_margin=(-np.max(scores)) if target==-1 else scores[target]-max(0.,np.max(competitors))
        ub=-float(result.mip_dual_bound)/n;gap=abs(float(result.fun-result.mip_dual_bound))/n
        upper=max(upper,ub);maxgap=max(maxgap,gap)
        if best is None or actual_margin>best['margin']:
            best=dict(net=net,scores=scores,selected=sel,margin=float(actual_margin),target=target)
        # A valid feasible outcome proves reachability; optimizing its margin
        # further is unnecessary for the outer minimum-true-value objective.
        if sel==target:
            return dict(**best,calls=calls,max_gap=maxgap,upper=upper,reachable=True,status='witness')
        if result.status!=0:
            raise RuntimeError(f'Unresolved attack target {target}: {result.message}; gap={gap}; margin={actual_margin}')
    return dict(**best,calls=calls,max_gap=maxgap,upper=upper,reachable=False,status='bounded')

def worst_attack(counts,gate,ka,truth):
    start=time.perf_counter();clean=corrected_scores(counts,gate);sel=choose(clean)
    if not ka or gate['name']=='retain':
        return dict(net=np.zeros(len(counts),dtype=int),scores=clean,selected=sel,calls=0,seconds=time.perf_counter()-start,
                    max_gap=0.,unresolved_lower_targets=0,targets=[])
    bb=branches(counts,gate);candidates=sorted(range(-1,len(truth)),key=lambda j:(value(j,truth),j))
    calls=0;maxgap=0.;unresolved=0;targets=[]
    best=dict(net=np.zeros(len(counts),dtype=int),scores=clean,selected=sel)
    for target in candidates:
        if value(target,truth)>value(best['selected'],truth)+1e-12:break
        attack=target_attack(counts,gate,ka,target,bb);calls+=attack['calls'];maxgap=max(maxgap,attack['max_gap'])
        targets.append(dict(target=int(target),margin=attack['margin'],upper=attack['upper'],reachable=attack['reachable']))
        if value(attack['selected'],truth)<value(best['selected'],truth)+1e-12:best=attack
        if value(attack['selected'],truth)<=value(target,truth)+1e-12:
            break
        if attack['upper']>=-1e-8:unresolved+=1
    return dict(net=best['net'],scores=best['scores'],selected=best['selected'],calls=calls,
                seconds=time.perf_counter()-start,max_gap=maxgap,unresolved_lower_targets=unresolved,targets=targets)

def write_csv(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(rows):
    cells={}
    for r in rows:cells.setdefault((r['environment'],r['eta'],r['epsilon'],r['method']),[]).append(r)
    out=[]
    for (name,eta,eps,method),rr in cells.items():
        vals=np.array([r['improvement'] for r in rr]);R=len(rr);h=sum(r['harm'] for r in rr)
        rad=student_t.ppf(.975,R-1)*vals.std(ddof=1)/np.sqrt(R)
        out.append(dict(environment=name,eta=eta,epsilon=eps,method=method,R=R,harm_count=h,harm_rate=h/R,
                        harm_lower=0. if h==0 else float(beta_dist.ppf(.025,h,R-h+1)),
                        harm_upper=1. if h==R else float(beta_dist.ppf(.975,h+1,R-h)),
                        harm_upper_onesided=1. if h==R else float(beta_dist.ppf(.95,h+1,R-h)),
                        mean_improvement=float(vals.mean()),improvement_low=float(vals.mean()-rad),improvement_high=float(vals.mean()+rad),
                        fallback_rate=float(np.mean([r['selected']==-1 for r in rr])),
                        unresolved=sum(r['unresolved_lower_targets']>0 for r in rr)))
    return out

def run(config_path,out,smoke=False):
    config=json.loads(config_path.read_text());out.mkdir(parents=True,exist_ok=True)
    (out/'frozen_protocol.json').write_bytes(config_path.read_bytes());rows=[];witness=[];start=time.perf_counter()
    seeds=range(1,3) if smoke else range(config['seeds']['start'],config['seeds']['start']+config['seeds']['count'])
    for name in config['environments']:
        for eta in config['coverage']:
            env=setup(config,name,eta)
            for seed in seeds:
                counts=log_counts(config['n'],env,seed,eta,name)
                for eps in config['corruption']:
                    k=int(np.floor(config['n']*eps))
                    for method in config['methods']:
                        gate=make_gate(counts,env,method,config['alpha'],k)
                        a=worst_attack(counts,gate,k,env['truth']);delta=value(a['selected'],env['truth'])
                        if gate['k']:
                            clean_gate=dict(gate,k=0)
                            assert np.all(a['scores']<=corrected_scores(counts,clean_gate)+1e-8)
                        r=dict(environment=name,eta=eta,epsilon=eps,seed=seed,method=method,selected=a['selected'],improvement=delta,
                               harm=int(delta<-1e-12),edits=int(np.abs(a['net']).sum()),calls=a['calls'],attack_seconds=a['seconds'],
                               max_gap=a['max_gap'],unresolved_lower_targets=a['unresolved_lower_targets'])
                        rows.append(r);witness.append(dict(**r,clean_counts=counts.tolist(),net=a['net'].tolist(),scores=a['scores'].tolist(),targets=a['targets']))
                if smoke or (seed-config['seeds']['start']+1)%10==0:
                    write_csv(out/'runs.csv',rows)
                    print(f'{name} eta={eta} seed={seed} outcomes={len(rows)} runtime={time.perf_counter()-start:.1f}s',flush=True)
    summary=summarize(rows);write_csv(out/'runs.csv',rows);write_csv(out/'summary.csv',summary)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'witnesses.json').write_text(json.dumps(witness,separators=(',',':')))
    metadata=dict(runtime_seconds=time.perf_counter()-start,observations=len(rows),milp_calls=sum(r['calls'] for r in rows),
                  unresolved_outcomes=sum(r['unresolved_lower_targets']>0 for r in rows),max_subproblem_gap=max(r['max_gap'] for r in rows),
                  protocol_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  python=platform.python_version(),numpy=np.__version__,smoke=smoke)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'protocol.json');p.add_argument('--out',type=Path,default=ROOT/'results');p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.config,a.out,a.smoke)
