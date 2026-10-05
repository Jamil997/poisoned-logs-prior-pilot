import itertools,json,unittest
from pathlib import Path
import numpy as np
from experiment import *
CONFIG=json.loads((ROOT/'protocol.json').read_text())

class Tests(unittest.TestCase):
    def test_population_means(self):
        for name in CONFIG['environments']:
            env=setup(CONFIG,name,.05);counts=np.round(env['joint']*10000000).astype(int)
            for method in CONFIG['methods'][1:]:
                gate=make_gate(counts,env,method,.05,0)
                means=(gate['c']*env['joint'][:,1]).sum(axis=1)+(gate['offset']*env['joint'].sum(axis=1)).sum(axis=1)
                np.testing.assert_allclose(means,env['truth'],atol=1e-12)
            self.assertAlmostEqual(env['truth'].max(),.04)

    def test_tiny_attack_enumeration(self):
        rng=np.random.default_rng(290)
        for name in CONFIG['environments']:
            env=setup(CONFIG,name,.2)
            # Use smaller radii solely to exercise nontrivial selection boundaries.
            for rep in range(2):
                counts=rng.integers(1,4,size=env['joint'].shape);n=counts.sum();k=2
                poss=[]
                for net in itertools.product(range(-2,3),repeat=len(counts)):
                    if sum(abs(v) for v in net)>k:continue
                    z=np.array(net)
                    if np.all(z<=counts[:,0]) and np.all(-z<=counts[:,1]):poss.append(z)
                for method in ['ips_h','robust_ips_h','centered_dr_b','robust_centered_dr_b','tabular_h','robust_tabular_h']:
                    gate=make_gate(counts,env,method,.05,k);gate['rad']*=.03
                    bb=branches(counts,gate);brute=10.
                    for net in poss:
                        scores=corrected_scores(poison(counts,net),gate)
                        derived=np.array([max(a+b@net for a,b in bbs)/n for bbs in bb])
                        np.testing.assert_allclose(derived,scores,atol=1e-12)
                        brute=min(brute,value(choose(scores),env['truth']))
                    attack=worst_attack(counts,gate,k,env['truth'])
                    self.assertAlmostEqual(value(attack['selected'],env['truth']),brute,places=10)

    def test_bernstein_variance_cap(self):
        rng=np.random.default_rng(2)
        for name in CONFIG['environments']:
            env=setup(CONFIG,name,.05);counts=log_counts(50000,env,8,.05,name)
            gate=make_gate(counts,env,'centered_dr_b',.05,0);c=gate['c'].reshape(len(env['pi']),env['X'],2)
            cap=(.25*env['mu']*c*c).sum(axis=2).max(axis=1)
            for _ in range(20):
                px=rng.dirichlet(np.ones(env['X']));q=rng.random((env['X'],2))
                mean=(px* (env['mu']*c*(q-.5)).sum(axis=2)).sum(axis=1)
                moment=(px*(.25*env['mu']*c*c).sum(axis=2)).sum(axis=1)
                self.assertTrue(np.all(moment-mean*mean<=cap+1e-12))

if __name__=='__main__':unittest.main(verbosity=2)
