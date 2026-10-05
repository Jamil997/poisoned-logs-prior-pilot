"""Deterministic per-log lower bound on true value under EVERY reward attack.

With equal attack and defence caps k, triangle inequality gives
  min_attacks score_j >= clean_nominal_score_j - C_{2k,j}(clean).
Also score_j(attacked) <= clean_nominal_score_j, since the original log is
feasible for the correction. These hold for arbitrary replacements in [0,1],
not just flips. The evaluator uses true values solely to certify the possible
selected outcomes; the defender never receives true reward means.
"""
import json,csv
from pathlib import Path
import numpy as np
from experiment import ROOT,setup,log_counts,make_gate,corrected_scores,write_csv

def certify(counts,gate,truth,k):
    upper=corrected_scores(counts,dict(gate,k=0))
    lower=corrected_scores(counts,dict(gate,k=min(2*k,gate['n'])))
    bound=float(min(0.,truth.min()))
    for v in sorted(set([0.]+truth.tolist())):
        inferior=upper[truth<v-1e-12]
        inferior_upper=float(np.max(inferior)) if len(inferior) else -np.inf
        if v<=0:
            # Baseline is always available, so strictly negative scores cannot win.
            good_lower=max(0.,float(np.max(lower[truth>=v-1e-12])))
        else:
            inferior_upper=max(0.,inferior_upper)
            good_lower=float(np.max(lower[truth>=v-1e-12]))
        if good_lower>inferior_upper+1e-9:bound=float(v)
    return bound,lower,upper

if __name__=='__main__':
    config=json.loads((ROOT/'protocol.json').read_text());env=setup(config,'diverse_matched_gain',.05);rows=[]
    for seed in range(2000,2100):
        counts=log_counts(50000,env,seed,.05,'diverse_matched_gain');k=250
        gate=make_gate(counts,env,'robust_centered_dr_b',.05,k)
        bound,lower,upper=certify(counts,gate,env['truth'],k)
        rows.append(dict(seed=seed,guaranteed_true_improvement=bound,lower_scores=lower.tolist(),upper_scores=upper.tolist()))
    (ROOT/'results/continuous_certificate.json').write_text(json.dumps(rows,indent=2))
    from collections import Counter
    print(Counter(round(r['guaranteed_true_improvement']*100,6) for r in rows))
    print('mean guaranteed improvement pp',100*np.mean([r['guaranteed_true_improvement'] for r in rows]))
