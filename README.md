# Sample efficiency of PPO and SAC on MuJoCo locomotion

SAC reaches a return of 3000 on Hopper-v5 in 360k environment steps. No PPO seed
reached that level within the full 1M step budget. Below a return of about 1500
the ordering is reversed and PPO is ahead.

**Status:** preliminary. Results below come from 3 seeds per algorithm, which is
the minimum that allows any statistical statement at all. A 5 seed rerun is in
progress.

![Learning curves for PPO and SAC on Hopper-v5](figures/hopper-v5_comparison.png)

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
| Algorithms | PPO, SAC (Stable-Baselines3 2.4) |
| Budget | 1,000,000 environment steps per run |
| Seeds | 3 per algorithm |
| Evaluation | every 10k steps, 10 episodes, deterministic policy, separate env |
| Hardware | Apple M3 Pro, CPU only, 6 runs in parallel, 59 min wall clock |

Evaluation runs in its own environment instance seeded differently from training,
and uses the deterministic policy rather than the sampling one. Training returns
are not used as a performance measure anywhere in this repo: during training the
policy is exploring, so its return is both depressed and noisy.

For PPO the observations and rewards are normalized with `VecNormalize`. SAC is
not normalized, because reward scaling interferes with its automatic entropy
temperature tuning. The normalization statistics are frozen during evaluation and
reward normalization is disabled there, so both algorithms are scored on the raw
reward scale.

## Hyperparameters

Both algorithms use published tuned configurations from
[rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo), not values
chosen here. This matters more than it might look. SAC works reasonably well out
of the box on MuJoCo, while PPO with library defaults does not, and a comparison
between a tuned algorithm and an untuned one measures tuning effort rather than
the algorithms. The exact values are in `run_experiment.py` and are also written
to `results/<env>/<algo>/seed_<n>/config.json` for every run.

## Results

### Final performance

Average of the last 10 evaluation points of each seed, then aggregated across
seeds. Taking the single best evaluation point instead would reward lucky
evaluations and inflate both numbers.

| Algorithm | Seeds | Mean | 95% CI | Min | Max |
|---|---|---|---|---|---|
| PPO | 3 | 2736.9 | +/- 471.3 | 2528.7 | 2900.1 |
| SAC | 3 | 3083.6 | +/- 322.5 | 2968.5 | 3224.3 |

The confidence intervals overlap, so a t interval alone would call this no
detectable difference. The per seed values tell a different story: every SAC seed
scored above every PPO seed, since the lowest SAC run (2968.5) beat the highest
PPO run (2900.1). Complete separation of two groups of three has an exact
one sided Mann-Whitney p of 0.0500, which is the smallest value this design can
produce. The result is therefore at the boundary of what 3 seeds can show, and it
is reported as such rather than as an established difference. With 5 seeds per
group the floor drops to 0.004.

### Sample efficiency

Environment steps at which the smoothed evaluation curve first reaches each
threshold, median across seeds. Smoothing is a 5 point rolling mean applied per
seed before aggregation. Without it a single lucky evaluation would count as
having reached the threshold, which understates the real step cost.

| Threshold | PPO | SAC | Faster |
|---|---|---|---|
| 1000 | 60k | 160k | PPO, 2.7x |
| 2000 | 270k | 220k | SAC, 1.2x |
| 2500 | 400k | 310k | SAC, 1.3x |
| 3000 | not reached in 1M | 360k | SAC, at least 2.8x |

## Interpretation

**The ordering depends on the performance level.** PPO reaches low returns much
sooner, the two cross somewhere around 200k steps, and from there SAC pulls ahead
and keeps going while PPO flattens out.

**The early PPO advantage has a mechanical explanation.** SAC spends its first
10,000 steps filling the replay buffer with random actions and does not update at
all during them (`learning_starts=10000`), and its early updates are fitted to
data from a near random policy, so the critic is not useful yet. PPO improves from
its first batch. The benefit of off-policy replay only appears once the buffer
holds enough varied experience to be worth reusing. This explanation is a
hypothesis, not something this experiment tested; a run with a smaller
`learning_starts` would move the crossover point if it is correct.

**PPO plateaus and SAC does not.** Within this budget PPO settles near 2700 and
stays there, while SAC is still improving at 1M steps. Whether PPO would reach
3000 given more steps is not answered here.

**SAC is much less stable mid training.** The spread across SAC seeds between
roughly 300k and 700k steps is very wide, with individual runs climbing and then
collapsing. This is known behaviour for Hopper, where a policy can fall into a
gait that terminates early. It is visible in the per seed lines in the figure and
is one reason the best checkpoint by evaluation return is saved separately from
the final one.

## Limitations

- 3 seeds per algorithm. Enough to detect complete separation, not enough to
  estimate the size of the gap with any precision.
- One environment. Hopper is known to be among the noisier locomotion tasks, and
  nothing here shows the result transfers to Walker2d or HalfCheetah.
- One budget. The comparison is specific to 1M steps. PPO is usually run for 5M
  to 10M steps on these tasks, and its plateau here may be a budget artifact.
- Thresholds are absolute and chosen by hand. A threshold based metric depends on
  where the thresholds sit; the four used here span the range the two algorithms
  actually cover.
- No hyperparameter search. Published values were used deliberately, but they were
  tuned by others on earlier environment versions.
- Aggregation uses the mean and a t interval. For a small number of runs,
  Agarwal et al. (2021) argue for the interquartile mean with bootstrap intervals
  instead. That is the next thing to change here.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

./run_all.sh                    # 3 seeds, Hopper-v5, 1M steps each
SEEDS=5 ./run_all.sh            # 5 seeds
ENV=Walker2d-v5 ./run_all.sh    # another environment
```

`run_all.sh` sets one BLAS thread per process and runs as many processes
concurrently as the machine has performance cores. The networks here are two
256 unit layers and the MuJoCo step is single threaded, so parallelism belongs
across runs rather than inside them.

Aggregation and plotting run automatically at the end of the sweep, or on their
own:

```bash
python aggregate_results.py --env Hopper-v5
python aggregate_results.py --env Hopper-v5 --thresholds 500 1000 2000 3000
```

Exact package versions used for the results above are in `requirements-lock.txt`.

## Repository layout

```
run_experiment.py      one (algorithm, seed) run
run_all.sh             the sweep, with bounded concurrency
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
