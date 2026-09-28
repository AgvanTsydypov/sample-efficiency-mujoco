# Sample efficiency of PPO and SAC on MuJoCo locomotion

SAC reaches a return of 3000 on Hopper-v5 in 250k environment steps, in all five
seeds. No PPO seed reached that level within the full 1M step budget. PPO is
faster than SAC below a return of about 1500, and its spread across seeds is
3.6 times wider.

A proposed explanation for SAC's slower start, the 10,000 step replay warmup, was
tested with a third experiment arm and rejected: removing the warmup made SAC
slower at every threshold, not faster.

![Learning curves for PPO, SAC and the learning_starts ablation on Hopper-v5](figures/hopper-v5_comparison.png)

## Question

How many environment steps does each algorithm need to reach a given level of
performance on a continuous control locomotion task?

This is deliberately not "which algorithm is better". Final performance under an
unlimited budget is a different question from step cost, and step cost is the one
that matters when the environment is slow or when the policy will eventually run
on physical hardware.

## Setup

| | |
|---|---|
| Environment | Hopper-v5 (Gymnasium, MuJoCo) |
| Arms | PPO, SAC, SAC with `learning_starts=1000` |
| Budget | 1,000,000 environment steps per run |
| Seeds | 5 per arm, 15 runs in total |
| Evaluation | every 10k steps, 10 episodes, deterministic policy, separate env |
| Hardware | Apple M3 Pro, CPU only, 6 concurrent runs |

Evaluation runs in its own environment instance seeded differently from training,
and uses the deterministic policy rather than the sampling one. Training returns
are not used as a performance measure anywhere in this repo: during training the
policy is exploring, so its return is both depressed and noisy.

For PPO the observations and rewards are normalized with `VecNormalize`. SAC is
not normalized, because reward scaling interferes with its automatic entropy
temperature tuning. The normalization statistics are frozen during evaluation and
reward normalization is disabled there, so all arms are scored on the raw reward
scale.

## Hyperparameters

PPO and SAC use published tuned configurations from
[rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo), not values
chosen here. This matters more than it might look. SAC works reasonably well out
of the box on MuJoCo, while PPO with library defaults does not, and a comparison
between a tuned algorithm and an untuned one measures tuning effort rather than
the algorithms.

The third arm changes exactly one value, `learning_starts` from 10000 to 1000, and
is a deliberate ablation rather than a tuning step. It writes to its own results
folder; `run_experiment.py` refuses an overridden hyperparameter that is not given
its own label, so an ablation can never be averaged into a baseline by accident.
Every run records its full configuration and any overrides in
`results/<env>/<arm>/seed_<n>/config.json`.

## Results

### Final performance

Average of the last 10 evaluation points of each seed, then aggregated across
seeds. Taking the single best evaluation point instead would reward lucky
evaluations and inflate every number.

| Arm | Seeds | Mean | 95% CI | Min | Max |
|---|---|---|---|---|---|
| PPO | 5 | 2152.6 | +/- 1007.4 | 1247.4 | 2900.1 |
| SAC | 5 | 3228.5 | +/- 280.0 | 2968.5 | 3524.1 |
| SAC, `learning_starts=1000` | 5 | 3247.5 | +/- 350.8 | 2832.9 | 3510.2 |

Every SAC seed scored above every PPO seed: the weakest SAC run (2968.5) beat the
strongest PPO run (2900.1). Complete separation of two groups of five has an exact
one sided Mann-Whitney p of 0.0040, which is also the smallest value this design
can produce, so the test is saturated rather than merely significant.

The two SAC arms are indistinguishable at the end. Their means differ by 19 points
against confidence intervals of 280 and 351, and their seed ranges overlap almost
completely.

### Sample efficiency

Environment steps at which the smoothed evaluation curve first reaches each
threshold, median across all seeds. Smoothing is a 5 point rolling mean applied
per seed before aggregation; without it a single lucky evaluation would count as
having reached the threshold. Seeds that never reached a threshold are censored at
the budget rather than dropped, so a count like (3/5) raises the reported cost
instead of hiding it.

| Threshold | PPO | SAC | SAC, `ls=1000` |
|---|---|---|---|
| 1000 | 60k | 100k | 130k |
| 2000 | 320k (3/5) | 170k | 190k |
| 2500 | 430k (3/5) | 190k | 340k |
| 3000 | not reached by any seed | 250k | 440k |

Against PPO, SAC is 1.9x cheaper to a return of 2000, 2.3x cheaper to 2500, and at
least 4x cheaper to 3000. Below 1500 the ordering is the other way round.

### Variance across seeds

| Arm | 95% CI half-width | Min to max spread |
|---|---|---|
| PPO | 1007.4 | 2.3x |
| SAC | 280.0 | 1.2x |
| SAC, `learning_starts=1000` | 350.8 | 1.2x |

This is a result in its own right, not a nuisance. Two of five PPO runs never
reached a return of 2000 at all, and one stayed inside a 1200 to 1400 band for the
entire million steps while another finished near 2900. Same code, same
hyperparameters, different seed. Both SAC arms land much closer together, although
their mid training curves are not smooth either: individual seeds climb above 3000
and then collapse toward 1000 before recovering, which is characteristic of Hopper,
where a policy can fall into a gait that terminates early.

For a setting where every run is expensive, an algorithm that fails in two runs out
of five is costly in a way that a mean does not express.

## Ablation: is the replay warmup the reason SAC starts slowly?

SAC spends its first 10,000 steps collecting random actions without a single
gradient update, while PPO improves from its first batch. That suggested a
hypothesis for PPO's early lead, with a clear prediction: reducing
`learning_starts` should move SAC's early curve left and leave its plateau alone.

The prediction was written down before the run, together with what would count as
a refutation. It was refuted.

| Threshold | SAC | SAC, `ls=1000` | Predicted | Observed |
|---|---|---|---|---|
| 1000 | 100k | 130k | faster | 1.3x slower |
| 2000 | 170k | 190k | faster | 1.1x slower |
| 2500 | 190k | 340k | same or faster | 1.8x slower |
| 3000 | 250k | 440k | same or faster | 1.8x slower |

The plateau half of the prediction held: 3247.5 against 3228.5, well inside the
confidence intervals. The speed half failed in the opposite direction at every
threshold. Removing the warmup does not buy an earlier start, it costs one, and it
slows the middle of training considerably.

A plausible reading is that updates against a buffer holding a thousand
transitions fit a narrow slice of experience and produce a worse critic, which
later training has to undo. That is presumably why the parameter exists. This repo
does not test that second explanation and does not claim it.

What the ablation does establish is negative and useful: the warmup is not the
reason PPO reaches low returns sooner. PPO gets to 1000 in 60k steps, and SAC
without the warmup still needs 130k. Whatever produces PPO's early advantage lies
elsewhere, and candidates this study has not examined include SAC's entropy
temperature starting high and its deterministic evaluation policy therefore
lagging its exploring one, and PPO's observation normalization giving it
well-scaled inputs from the first update.

## Three seeds were not enough, twice

The first sweep used 3 seeds. Extending to 5 changed the answer twice, in two
different ways:

**The size of an effect.** PPO's final performance fell from 2736.9 to 2152.6, a
27% difference, and the failure mode only appeared in the new seeds. None of the
first three PPO runs were bad ones.

**The direction of an effect.** On 3 seeds the ablation looked faster to a return
of 1000 than baseline SAC, 130k against 160k, which was weak support for the
hypothesis. On 5 seeds the baseline's median dropped to 100k while the ablation's
stayed at 130k, and the sign flipped. The conclusion reversed on data alone, with
no change to the code.

Both are the concern raised in Henderson et al. (2018), reproduced by accident on
a very small study. Five seeds is not a fix, it is the smallest number that made
these two problems visible.

## Limitations

- 5 seeds per arm. Enough to saturate the rank test between PPO and SAC, not
  enough to estimate the size of any gap with precision. The PPO confidence
  interval spans almost half its own mean.
- One environment. Hopper is among the noisier locomotion tasks, and nothing here
  shows the result transfers to Walker2d or HalfCheetah.
- One budget. The comparison is specific to 1M steps. PPO is usually run for 5M to
  10M steps on these tasks, and its plateau here may be a budget artifact.
- One ablation value. `learning_starts` was tested at 1000 against 10000. A sweep
  over the parameter would say more, and is not done here.
- Thresholds are absolute and chosen by hand. A threshold based metric depends on
  where the thresholds sit; the four used here span the range the arms cover.
- No hyperparameter search. Published values were used deliberately, but they were
  tuned by others and on earlier environment versions.
- Aggregation uses the mean and a t interval. For a small number of runs,
  Agarwal et al. (2021) argue for the interquartile mean with bootstrap intervals
  instead. That is the next thing to change here.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

SEEDS=5 ./run_all.sh
SEEDS=5 ALGOS=sac LABEL=sac_ls1000 RUN_ARGS="--learning-starts 1000" ./run_all.sh
python aggregate_results.py --env Hopper-v5 --algos ppo sac sac_ls1000
```

Raising `SEEDS` extends an existing sweep rather than repeating it: runs that
already have evaluation logs are skipped, and `FORCE=1` redoes them.

`run_all.sh` sets one BLAS thread per process and runs as many processes
concurrently as the machine has performance cores. The networks here are two
256 unit layers and the MuJoCo step is single threaded, so parallelism belongs
across runs rather than inside them. Fifteen runs at 1M steps took under three
hours in total on the hardware above.

Exact package versions used for the results above are in `requirements-lock.txt`.

## Repository layout

```
run_experiment.py      one (arm, seed) run
run_all.sh             the sweep, with bounded concurrency and skip-if-present
aggregate_results.py   curves, tables, statistics
results/               evaluation logs and configs, one folder per run
figures/               generated plots
```

Model checkpoints are not tracked. Evaluation logs (`evaluations.npz`) and run
configs (`config.json`) are, since they are small and they are the actual evidence
behind every number above.

## References

- Schulman et al., Proximal Policy Optimization Algorithms, 2017. https://arxiv.org/abs/1707.06347
- Haarnoja et al., Soft Actor-Critic, 2018. https://arxiv.org/abs/1801.01290
- Raffin et al., Stable-Baselines3, JMLR 2021. https://jmlr.org/papers/v22/20-1364.html
- Towers et al., Gymnasium, 2024. https://arxiv.org/abs/2407.17032
- Todorov et al., MuJoCo: A physics engine for model-based control, IROS 2012.
- Agarwal et al., Deep Reinforcement Learning at the Edge of the Statistical Precipice, NeurIPS 2021. https://arxiv.org/abs/2108.13264
- Henderson et al., Deep Reinforcement Learning that Matters, AAAI 2018. https://arxiv.org/abs/1709.06560
