# Poisoned Logs: Prior Pilot Experiments

**Status: prior pilot snapshot, 5 October 2026.** This repository preserves the
completed synthetic pilot used to assess the feasibility of the project
“Can Poisoned Logs Make a Bad Policy Look Safe?” It is separate from the final
or subsequently updated project. It contains experimental code, protocols and
recorded results; presentation and document files are intentionally excluded.

The snapshot contains the stronger follow-up pilot, including the original
simulation setting as a control. It does not claim a new algorithm, global
Pareto optimality, a real-data evaluation, or a finished research contribution.

## Research question

Can an offline deployment check resist manipulated rewards **and still approve
useful policies**? Simply retaining the current policy prevents harm but gains
nothing. The experiment therefore measures safety and useful improvement together.

A contextual-bandit log records a context, the action taken, its observed reward,
and the probability with which the logging policy chose that action. Only the
chosen action's reward is observed. A poisoned log has some of those reward
entries replaced. A bad policy is a candidate whose true expected reward is below
the current policy's value, even if its estimate on the poisoned log looks good.

All logs here are generated locally by a simulator. True values are available
for evaluation, so a successful-looking deployment can be checked against ground
truth rather than against another possibly corrupted estimate.

## Attacker and defender

| Component | Assumptions and objective |
|---|---|
| Defender | Chooses among six fixed candidate policies, or retains the current policy. Knows the randomized logging probabilities and a declared corruption cap. |
| Attacker | Sees the whole log, candidates and defender code. Also knows true policy values in this deliberately strong synthetic stress test. |
| Allowed edits | Replace at most `k` rewards; rewards must remain in `[0,1]`. Contexts, actions, propensities, policies and sample size cannot change. |
| Attack objective | Minimize the true value of the policy eventually selected, including forcing fallback when possible. |
| Main cap assumption | The defender's declared cap equals the attacker's actual budget. A separate sweep deliberately violates or overstates it. |

The evaluation log is independent of the fixed candidate policies. Defenders
receive neither oracle reward means nor additional clean reward data. Clean
records are i.i.d., and every action has positive logging probability.

## Methods

The main comparison implements seven deployment gates:

1. Retain the current policy.
2. IPS with a Hoeffding lower confidence bound.
3. IPS/Hoeffding with an exact reward-edit correction.
4. Centered doubly robust (DR) estimation with a Bernstein bound.
5. Centered DR/Bernstein with the same reward-edit correction.
6. A tabular estimator with conditional Hoeffding and context uncertainty.
7. The corrected tabular estimator.

A supplementary comparison adds the published **SN-ESLB** implementation and
its corrected version, adapted to comparing candidates against the current policy.

### Protected centered DR rule

For candidate `j`, let `pi_j` denote its action probabilities, `pi_0` those of
the current policy, and `mu` those of the logger. Define

```text
c_ij = (pi_j(a_i | x_i) - pi_0(a_i | x_i)) / mu(a_i | x_i)
Z_ij = c_ij * (r_i - 0.5)
```

This is DR with the fixed, deliberately simple reward predictor `qhat = 0.5`;
it is not a learned or oracle reward model. The mean of `Z_ij` estimates true
policy improvement. For `M` candidates and confidence level `alpha`:

```text
Vbar_j = max_x sum_a mu(a | x) * c_j(x,a)^2 / 4
B_j    = max_(x,a) |c_j(x,a)|
t      = log(M / alpha)
b_j    = B_j * t / (3*n)
s_j    = b_j + sqrt(b_j^2 + 2*Vbar_j*t/n)
```

`s_j` follows from Bernstein's inequality. Its variance bound depends on the
fixed policies and logger, so poisoned rewards cannot artificially shrink it.
For each observed reward, its maximum downward edit impact is

```text
d_ij = c_ij * observed_reward_i - min(0, c_ij)
C_kj = sum of the k largest d_ij / n
L_j  = mean(observed Z_ij) - C_kj - s_j
```

Choose the candidate with the largest strictly positive `L_j`; otherwise retain
the current policy. Ties prefer the lowest candidate index. With the stated
assumptions and a sufficient edit cap, the clean-data confidence bound and a
union bound give harmful-deployment probability at most `alpha`. This is a
conditional guarantee, not a claim of zero risk.

## Experimental design

The [main protocol](protocol.json) was fixed before running this follow-up's
outcomes. This is a local frozen protocol, not an externally registered study.
The later supplementary protocols explicitly state that they followed the main
results.

| Experiment | Settings | Recorded method outcomes |
|---|---|---:|
| Main comparison | Two environments; logging coverage `0.5, 0.2, 0.05`; edit fractions `0, 0.005, 0.01, 0.025, 0.05`; seven methods; 100 seeds per cell | 21,000 |
| SN-ESLB supplement | Diverse environment; coverage `0.05`; four edit fractions; two adapters; 100 seeds | 800 |
| Fresh-seed cap sweep | Two environments; coverage `0.05`; four edit fractions; six declared-cap multipliers; 100 fresh seeds | 4,800 |

Every log has **50,000 records**; `alpha = 0.05`. Main and supplementary seeds
are `2000–2099`; the cap sweep uses `3000–3099`. Logs are paired across methods
and edit budgets. These outcomes are **not 26,600 independent datasets**.

The original two-context, one-parameter policy family remains a control. The
added environment has four equally likely contexts, action-0 reward mean `0.5`,
and action-1 means `[0.62, 0.48, 0.54, 0.46]`. Six policies use action 1 on
different context subsets. Their true improvements are
`[3, -0.5, 1, -1, 4, 2.5]` percentage points. Both environments have maximum
candidate improvement **+4 percentage points**. The added setting deliberately
tests policy diversity; all original-control and unsuccessful results are retained.

### Metrics fixed before the main follow-up

- **Harm:** probability of selecting a policy with negative true improvement.
- **Utility:** mean true improvement after each method's own adaptive attack,
  counting fallback as zero.
- Secondary measures: fallback rate, numerical attack gaps and runtime.

The primary condition is the diverse environment, **5% rare-action coverage and
0.5% editable rewards**. Success requires a one-sided 95% harm upper bound below
5%, a 95% lower utility bound above +1 percentage point, and more than +1 point
paired improvement over corrected IPS. These criteria passed.

## Recorded results

### Primary comparison

All nine gates had **0/100 observed harmful selections** in the primary condition.
Zero observed harm does not establish zero population risk: the pointwise
one-sided 95% upper bound is **2.95%**.

| Gate | Mean true improvement (percentage points) |
|---|---:|
| **Corrected centered DR / Bernstein** | **3.43** |
| Nominal IPS / Hoeffding | 2.55 |
| Nominal centered DR / Bernstein | 2.50 |
| Nominal tabular / Hoeffding | 2.375 |
| Corrected IPS / Hoeffding | 0.27 |
| Corrected tabular / Hoeffding | 0.20 |
| SN-ESLB advantage adapter | 0.00 |
| Corrected SN-ESLB adapter | 0.00 |
| Retain current policy | 0.00 |

Corrected centered DR approved a useful policy on 98/100 primary logs. Its
pointwise 95% Student-t utility interval is **[3.29, 3.57]** points. Paired gains
were **3.16 [2.94, 3.36]** points over corrected IPS and **0.88 [0.58, 1.20]** over
nominal IPS, using percentile-bootstrap intervals. Paired one-sided utility
tests remained significant after Holm correction across eight comparisons.
Equality of the methods' population harm rates is not proved.

Sources: [main results](results/summary.csv), [primary analysis](results/analysis.json),
[supplementary results](eslb_results/summary.csv).

### Pareto comparison on fresh seeds

A tested configuration is empirically dominated when another has no higher
measured harm and no lower measured utility, with a strict improvement on at
least one axis. Here “Pareto” concerns safety and utility, not the 80/20 rule.

The exploratory [cap-sweep protocol](frontier_protocol.json) measures maximum
observed harm across eight stress settings and uniformly averaged utility
across those settings. The attacker always keeps its full actual edit budget.

| Declared cap / actual budget | Worst-setting harm | Mean improvement (points) |
|---:|---:|---:|
| 0 | 100% | -0.16125 |
| 0.25 | 100% | -0.138125 |
| 0.50 | 100% | 0.20625 |
| 0.75 | 64% | 0.31000 |
| **1.00** | **0%** | **0.42750** |
| 1.50 | 0% | 0.32500 |

The correct cap was the **only empirically non-dominated configuration among
these six**. Its utility interval, bootstrapping paired seed clusters, is
**[0.4100, 0.44375]** points. A Bonferroni one-sided bound gives a simultaneous
95% upper bound of **4.95%** on the maximum risk across its eight zero-harm cells.

This is a finite exploratory result, not global Pareto optimality. Multipliers
below one intentionally lose the sufficient-cap guarantee. The aggregate
0.4275-point result is different from the 3.43-point primary-condition result.

Sources: [all cap-sweep outcomes](frontier_results/runs.csv),
[frontier](frontier_results/frontier.json), [uncertainty](frontier_results/analysis.json).

### Stronger audit for arbitrary real-valued edits

The optimization searches binary reward flips. A separate deterministic audit
bounds **every** allowed replacement in `[0,1]` for the protected DR rule. For
a clean nominal score `U_j` and matching attacker/defender budget `k`:

```text
U_j - C_(2k,j)(clean log) <= attacked corrected score_j <= U_j
```

The upper bound holds because the original clean log is a feasible correction;
the lower bound follows by composing two `k`-edit neighborhoods. Comparing
candidate envelopes and evaluator-only true values certifies at least:

- **+3 points on 97 primary logs**;
- **+2.5 points on one log**;
- **baseline-or-better on two logs**.

The mean per-log lower bound is **2.935 points**. This is distinct from the
measured worst-binary-attack mean of 3.43. All 100 primary logs rule out harm
under every permitted real-valued attack. These are conditional certificates
on the realized simulated logs; the defender does not receive oracle labels.

The proposed restricted differential-evolution search in
[continuous_protocol.json](continuous_protocol.json) was superseded by this
stronger certificate and **was not run**. Five hundred random fractional attacks
were additionally checked as implementation tests.

## Limitations and negative results

- At **1% edits**, in the primary environment and coverage, protected centered
  DR falls back on **all 100 logs**.
- At **2.5% edits**, nominal centered DR makes harmful selections on all 100
  logs there, while the protected version again falls back.
- The original control remains difficult. A conservative rule cannot recover
  arbitrary information destroyed by corrupted rewards.
- Centered DR uses `qhat = 0.5`; learned DR and shrinkage baselines are not yet
  evaluated. Established mathematical ingredients are not a novelty claim.
- Unknown or poisoned propensities, dependent records, policies fitted on the
  evaluation log and underestimated corruption caps are outside the guarantee.
- Stronger policy libraries and independent benchmark datasets remain future work.

## Attack implementation and validation

The binary attack represents net reward changes by context/action-group counts,
a lossless reduction for these gates. It tests targets in ascending true value,
including fallback, using mixed-integer winning-margin optimization. Top-k
corrections are represented by finitely many affine branches. Realized selections
and tie-breaking are checked; subproblem tolerance is `1e-6` reward units.

[Tests](test_experiment.py) compare population expectations and variance bounds,
verify the correction identity, and compare tiny attacks with exhaustive search.
The [main audit](results/audit.json) reconstructs 21,000 outcomes from full row
vectors (maximum score error below `2.5e-13`). Two zero-margin flags are resolved
analytically using the tie rule; the original flags remain in run metadata for
transparency. The [additional audit](extra_audit.json) validates the other 5,600
outcomes, also within `2.5e-13`, with no unresolved cases.

### Published baseline boundary

The SN-ESLB supplement uses unmodified `ESLB` class definitions and the sampling
helper from the [authors' repository](https://github.com/google-deepmind/offpolicy_selection_eslb),
commit `3c867e75ca22fd6eb01ac21e41fdc77f47b06ca8`. The vendored subset includes
the upstream Apache-2.0 [license](vendor/offpolicy_selection_eslb/LICENSE).
The AST loader avoids unrelated dataset and model imports without changing the
estimator formulas.

The adapter uses general bounded-reward Bernstein bias control, candidate lower
bounds and a complement-reward baseline bound, with `alpha/(M+1)` per bound.
Monte Carlo randomness is frozen independently of rewards: 20 batches of five
draws. Formula checks agree within `2.1e-13`. This numerical approximation and
conservative baseline contrast are not the paper's full benchmark or a proof
that SN-ESLB is universally inferior.

## Reproduce

Tested with Python 3.12 and the versions pinned in [requirements.txt](requirements.txt).
No GPU or language model is needed. Recorded wall times on an Apple M5 Pro with
24 GiB RAM were **283.9 s** for the main run, **135.0 s** for the SN-ESLB supplement,
and **103.9 s** for the cap sweep, excluding installation, audits and plotting.
Runtime depends on hardware and solver behavior.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python test_experiment.py
```

Recorded results are included. To run a small, separate smoke experiment:

```bash
python experiment.py --smoke --out smoke_results
```

To regenerate the full experiment and analyses, run in this order. These commands
overwrite the corresponding recorded outputs, so use a copy if preserving the
snapshot is important:

```bash
python experiment.py
python eslb_comparison.py
python continuous_certificate.py
python audit.py
python analyze.py
python frontier.py
python extra_audit.py
python make_figures.py
```

The final command generates local PNG/SVG plots in ignored `figures/` and writes
the cap-sweep uncertainty analysis. It does not generate PDFs. Saved records use
reward units; multiply improvement by 100 to obtain percentage points.

| Files | Purpose |
|---|---|
| `experiment.py`, `protocol.json` | Simulator, gates, adaptive binary attacks, main sweep |
| `eslb_comparison.py`, `eslb_protocol.json` | Published implementation adapter and supplement |
| `continuous_certificate.py`, `continuous_protocol.json` | All-real-valued certificate and documented change from planned search |
| `audit.py`, `extra_audit.py`, `test_experiment.py` | Independent validation and tiny-instance checks |
| `analyze.py`, `make_figures.py` | Paired comparisons, uncertainty and optional local figures |
| `frontier.py`, `frontier_protocol.json` | Fresh-seed declared-cap sweep |
| `results/`, `eslb_results/`, `frontier_results/` | CSV/JSON outcomes, summaries, metadata and attack witnesses |
| `MANIFEST.sha256.json` | SHA-256 hashes of the publication snapshot |

## Related work

- Lobo et al. (UAI 2022), [Data Poisoning Attacks on Off-Policy Evaluation Methods](https://proceedings.mlr.press/v180/lobo22a.html).
- Thomas et al. (AAAI 2015), [High Confidence Off-Policy Evaluation](https://ojs.aaai.org/index.php/AAAI/article/view/9541).
- Dudik et al. (ICML 2011), [Doubly Robust Policy Evaluation and Learning](https://icml.cc/2011/papers/554_icmlpaper.pdf).
- Kuzborskij et al. (AISTATS 2021), [Confident Off-Policy Evaluation and Selection through Self-Normalized Importance Weighting](https://proceedings.mlr.press/v130/kuzborskij21a.html).
- Su et al. (ICML 2020), [Doubly Robust Off-Policy Evaluation with Shrinkage](https://proceedings.mlr.press/v119/su20a.html).
- Zhang et al. (AISTATS 2022), [Corruption-Robust Offline Reinforcement Learning](https://proceedings.mlr.press/v151/zhang22c.html).

## Ethics

The attacks edit local synthetic logs only. No private records or deployed
systems are attacked, and this pilot reports no external vulnerability requiring
disclosure. The repository is a preliminary research artifact, not a deployment
safety assessment for a real system.
