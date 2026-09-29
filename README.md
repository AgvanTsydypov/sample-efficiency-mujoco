# Sample efficiency of PPO and SAC on MuJoCo locomotion

SAC beats PPO on both environments tested, but how much cheaper it is depends
heavily on the task: reaching a return of 2000 costs SAC 3 times fewer
environment steps than PPO on Hopper-v5 and 1.2 times fewer on Walker2d-v5.

A crossover found on Hopper, where PPO appeared to reach low returns first, did
not survive: it shrank to two evaluation points once a configuration error was
fixed, and the ordering is reversed on Walker2d.

![PPO, SAC and the learning_starts ablation on Hopper-v5](figures/hopper-v5_comparison.png)

![PPO and SAC on Walker2d-v5](figures/walker2d-v5_comparison.png)

## Question

How many environment steps does each algorithm need to reach a given level of
performance, and does the answer hold across tasks?

This is deliberately not "which algorithm is better". Final performance under an
unlimited budget is a different question from step cost, and step cost is the one
that matters when the environment is slow or when the policy will eventually run
on physical hardware.

## Setup

| | |
|---|---|
| Environments | Hopper-v5 and Walker2d-v5 (Gymnasium, MuJoCo) |
| Arms | PPO, SAC, plus SAC with `learning_starts=1000` on Hopper |
| Budget | 1,000,000 environment steps per run |
| Seeds | 5 per arm per environment, 25 runs in total |
| Evaluation | every 10k steps, 10 episodes, deterministic policy, separate env |
| Hardware | Apple M3 Pro, CPU only, 6 concurrent runs |

Evaluation runs in its own environment instance seeded differently from training,
and uses the deterministic policy rather than the sampling one. Training returns
are not used as a performance measure anywhere in this repo: during training the
policy is exploring, so its return is both depressed and noisy.

For PPO the observations and rewards are normalized with `VecNormalize`, which is
what the published configuration specifies. SAC is not normalized, because reward
scaling interferes with its automatic entropy temperature tuning. The
normalization statistics are frozen during evaluation and reward normalization is
disabled there, so all arms are scored on the raw reward scale.

## Hyperparameters

Values are transcribed from
[rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo),
`hyperparams/ppo.yml` and `hyperparams/sac.yml`, not chosen here. SAC works
reasonably well out of the box on MuJoCo while PPO with library defaults does not,
so a comparison between a tuned algorithm and an untuned one would measure tuning
effort rather than the algorithms.

The zoo tunes PPO per environment, so `run_experiment.py` holds a table keyed by
environment and raises rather than reusing another task's values. SAC's entry for
MuJoCo locomotion is one shared anchor, SB3 defaults with `learning_starts` raised
to 10000, identical across these tasks. The zoo publishes v4 entries and they are
applied here to v5, which changed the environments only slightly.

Every run records its full configuration, and any deliberate override, in
`results/<env>/<arm>/seed_<n>/config.json`.

## Results

### Final performance

Average of the last 10 evaluation points of each seed, then aggregated across
seeds. Taking the single best evaluation point instead would reward lucky
evaluations and inflate every number.

**Hopper-v5**

| Arm | Mean | 95% CI | Min | Max |
|---|---|---|---|---|
| PPO | 2252.4 | +/- 549.4 | 1777.2 | 2811.3 |
| SAC | 3228.5 | +/- 280.0 | 2968.5 | 3524.1 |
| SAC, `learning_starts=1000` | 3247.5 | +/- 350.8 | 2832.9 | 3510.2 |

**Walker2d-v5**

| Arm | Mean | 95% CI | Min | Max |
|---|---|---|---|---|
| PPO | 3217.9 | +/- 1059.0 | 2488.8 | 4272.9 |
| SAC | 4327.1 | +/- 254.9 | 4070.2 | 4600.6 |

On Hopper the two groups are completely separated: the weakest SAC run (2968.5)
beat the strongest PPO run (2811.3). That gives an exact one sided Mann-Whitney
p of 0.0040, the smallest value a 5 against 5 design can produce, so the test is
saturated rather than merely significant.

On Walker2d the groups interleave. SAC still leads, at p = 0.0159, but PPO's best
seed finished above two of the five SAC seeds. The same conclusion holds on both
tasks with visibly different confidence.

### Sample efficiency

Environment steps at which the smoothed evaluation curve first reaches each
threshold, median across all seeds. Smoothing is a 5 point rolling mean applied
per seed before aggregation; without it a single lucky evaluation would count as
having reached the threshold. Seeds that never reached a threshold are censored at
the budget rather than dropped, so a count like (3/5) raises the reported cost
instead of hiding it.

**Hopper-v5**

| Threshold | PPO | SAC | SAC, `ls=1000` | SAC vs PPO |
|---|---|---|---|---|
| 1000 | 80k | 100k | 130k | PPO by 1.3x |
| 2000 | 510k | 170k | 190k | SAC by 3.0x |
| 2500 | 970k (3/5) | 190k | 340k | SAC by 5.1x |
| 3000 | not reached by any seed | 250k | 440k | SAC by more than 4x |

**Walker2d-v5**

| Threshold | PPO | SAC | SAC vs PPO |
|---|---|---|---|
| 1000 | 270k | 220k | SAC by 1.2x |
| 2000 | 370k | 300k | SAC by 1.2x |
| 2500 | 470k | 380k | SAC by 1.2x |
| 3000 | 530k (3/5) | 400k | SAC by 1.3x |

The direction is the same on both tasks above the lowest threshold. The magnitude
is not: a factor of 3 to 5 on Hopper against a flat factor of 1.2 on Walker2d.
Anyone quoting a single number for "how much more sample efficient SAC is" is
quoting a property of their benchmark.

### What replicated across environments

| Finding | Hopper | Walker2d | Verdict |
|---|---|---|---|
| SAC ends above PPO | complete separation, p = 0.0040 | interleaved, p = 0.0159 | replicated, weaker |
| SAC cheaper above low thresholds | 3.0x to 5.1x | 1.2x | replicated, magnitude varies 4x |
| PPO more variable across seeds | CI 2.0x wider than SAC | CI 4.2x wider | replicated |
| PPO reaches low returns first | 80k against 100k | 270k against 220k, reversed | **did not replicate** |

The last row is the reason the second environment was worth an hour and a half of
compute. On Hopper alone, PPO reaching a return of 1000 sooner looked like a real
phenomenon worth explaining, and an ablation was built around explaining it. On
Walker2d the ordering is the other way round, and on Hopper the remaining gap is
80k against 100k, two evaluation points apart and not separable from noise. There
is no early PPO advantage to explain.

## Ablation: is the replay warmup the reason SAC starts slowly?

SAC spends its first 10,000 steps collecting random actions without a single
gradient update, while PPO improves from its first batch. That suggested a
hypothesis for PPO's apparent early lead on Hopper, with a clear prediction:
reducing `learning_starts` should move SAC's early curve left and leave its
plateau alone. The prediction and what would count as a refutation were written
down before the run.

| Threshold | SAC | SAC, `ls=1000` | Predicted | Observed |
|---|---|---|---|---|
| 1000 | 100k | 130k | faster | 1.3x slower |
| 2000 | 170k | 190k | faster | 1.1x slower |
| 2500 | 190k | 340k | same or faster | 1.8x slower |
| 3000 | 250k | 440k | same or faster | 1.8x slower |

The plateau half held: 3247.5 against 3228.5, well inside the confidence
intervals. The speed half failed in the opposite direction at every threshold.
Removing the warmup does not buy an earlier start, it costs one.

A plausible reading is that updates against a buffer holding a thousand
transitions fit a narrow slice of experience and produce a worse critic, which
later training has to undo. That is presumably why the parameter exists. This repo
does not test that second explanation and does not claim it.

The Walker2d results then removed the ground the hypothesis stood on: the early
advantage it was built to explain does not exist outside Hopper. The ablation is
kept here because a hypothesis that was stated, tested and refuted is part of the
record, and because its result stands on its own: `learning_starts` is not free to
reduce.

## Small samples changed the conclusion twice

**Three seeds flipped the sign of an effect.** On 3 seeds the ablation looked
faster to a return of 1000 than baseline SAC, 130k against 160k, which read as
weak support for the hypothesis. On 5 seeds the baseline's median dropped to 100k
while the ablation's stayed at 130k, and the ordering reversed. No code changed,
only the amount of data.

**One environment produced a finding that does not exist.** The PPO early
advantage was the most interesting thing in the single environment version of this
study. It is absent on the second environment.

Both are the concern raised in Henderson et al. (2018). Five seeds and two
environments is not a fix, it is the smallest setup that made these two problems
visible.

## Correction

An earlier version of this study used a PPO configuration with two values that did
not match the zoo: `vf_coef` was 0.19 against a published 0.835671, and `ent_coef`
was 0.0 against a published 0.00229519. PPO therefore ran with its entropy bonus
switched off and its value loss weighted about four times too lightly, which
undercut the "published values on both sides" claim the study rests on.

The PPO arm on Hopper was rerun with the corrected values. The effect was
substantial: the confidence interval narrowed from +/- 1007.4 to +/- 549.4, the
worst seed improved from 1247.4 to 1777.2, and a claim in the earlier write-up
that two of five PPO runs never reached a return of 2000 turned out to be an
artifact of the error. With the correct configuration all five do. Corrected PPO
is also slower, reaching 2000 at 510k steps against 320k before, so the SAC
advantage on Hopper is larger than first reported, not smaller.

The error was caught by checking the transcribed numbers against the source file
rather than by anything in the results looking wrong. That is the uncomfortable
part and the reason it is recorded here.

## Limitations

- 5 seeds per arm. Enough to saturate the rank test on Hopper, not enough to
  estimate the size of any gap with precision. The PPO confidence interval on
  Walker2d spans a third of its own mean.
- Two environments. Better than one, and enough to show that a headline finding
  can fail to replicate, but not a benchmark.
- One budget. The comparison is specific to 1M steps. PPO is usually run for 5M to
  10M steps on these tasks, and on Walker2d it is still climbing at the end.
- The ablation ran on Hopper only, at a single value, 1000 against 10000.
- Thresholds are absolute and chosen by hand. A threshold based metric depends on
  where the thresholds sit; the four used here span the range the arms cover.
- The zoo publishes v4 hyperparameters and they are used on v5 here.
- Aggregation uses the mean and a t interval. For a small number of runs,
  Agarwal et al. (2021) argue for the interquartile mean with bootstrap intervals
  instead. That is the next thing to change here.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

SEEDS=5 ./run_all.sh
SEEDS=5 ENV=Walker2d-v5 ./run_all.sh
SEEDS=5 ALGOS=sac LABEL=sac_ls1000 RUN_ARGS="--learning-starts 1000" ./run_all.sh

python aggregate_results.py --env Hopper-v5 --algos ppo sac sac_ls1000
python aggregate_results.py --env Walker2d-v5
```

Raising `SEEDS` extends an existing sweep rather than repeating it: runs that
already have evaluation logs are skipped, and `FORCE=1` redoes them.

`run_all.sh` sets one BLAS thread per process and runs as many processes
concurrently as the machine has performance cores. The networks here are two
256 unit layers and the MuJoCo step is single threaded, so parallelism belongs
across runs rather than inside them. Twenty five runs at 1M steps took about four
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
- Raffin, RL Baselines3 Zoo, 2020. https://github.com/DLR-RM/rl-baselines3-zoo
- Towers et al., Gymnasium, 2024. https://arxiv.org/abs/2407.17032
- Todorov et al., MuJoCo: A physics engine for model-based control, IROS 2012.
- Agarwal et al., Deep Reinforcement Learning at the Edge of the Statistical Precipice, NeurIPS 2021. https://arxiv.org/abs/2108.13264
- Henderson et al., Deep Reinforcement Learning that Matters, AAAI 2018. https://arxiv.org/abs/1709.06560
