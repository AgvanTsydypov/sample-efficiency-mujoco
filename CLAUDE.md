# Project context for Claude Code

## What this repository is

A small empirical study comparing the sample efficiency of PPO and SAC on MuJoCo
locomotion. It is a portfolio project for a PhD application in robot learning, so
methodological correctness matters more here than getting a favourable result.

Research question: how many environment steps does each algorithm need to reach a
given level of performance on Hopper-v5. Not "which algorithm is better".

## Who you are working with

MSc AI graduate (QMUL), strong in deep learning, NLP and computer vision. New to
reinforcement learning specifically, so RL concepts are worth explaining when they
come up, but PyTorch, Python and general ML need no explanation. Preparing a PhD
application in robot learning in Europe.

Explanations should be practical and direct. No filler.

## Current status

The 3 seed sweep is done. A 5 seed rerun may be in progress or complete; check
`results/Hopper-v5/*/seed_*/` for how many seed folders exist before quoting any
numbers, and rerun `aggregate_results.py` rather than trusting numbers written in
README.md or here.

Results from the 3 seed run:

| Algorithm | Final return (mean) | 95% CI | Min | Max |
|---|---|---|---|---|
| PPO | 2736.9 | +/- 471.3 | 2528.7 | 2900.1 |
| SAC | 3083.6 | +/- 322.5 | 2968.5 | 3224.3 |

Steps to first reach a threshold, median across seeds:

| Threshold | PPO | SAC |
|---|---|---|
| 1000 | 60k | 160k |
| 2000 | 270k | 220k |
| 2500 | 400k | 310k |
| 3000 | not reached in 1M | 360k |

Two findings so far:

1. The ordering crosses over around 200k steps. PPO reaches low returns sooner,
   SAC leads from there on and keeps improving while PPO plateaus near 2700.
2. Every SAC seed beat every PPO seed on final return. Complete separation of two
   groups of 3 gives an exact one sided Mann-Whitney p of 0.0500, the floor for
   that design. The result is at the boundary of what 3 seeds can show.

Working hypothesis for the early PPO advantage: SAC spends its first 10,000 steps
filling the replay buffer without any updates (`learning_starts=10000`), and its
early updates fit data from a near random policy. This has not been tested. The
test would be a SAC run with a smaller `learning_starts`; if the hypothesis holds,
the crossover moves left.

## Files

```
run_experiment.py      one (algorithm, seed) run, writes to results/<env>/<algo>/seed_<n>/
run_all.sh             the sweep, bounded concurrency, calls aggregate at the end
aggregate_results.py   learning curves, final performance table, threshold table, rank test
train_cartpole.py      early learning script, PPO on CartPole, kept for reference
train_mujoco.py        early learning script, SAC on MuJoCo, kept for reference
results/               evaluation logs and configs, one folder per run
figures/               generated plots
```

## Rules that must not be broken

These are methodological, not stylistic. Breaking any of them invalidates results.

- **Performance is measured from `evaluations.npz`, never from training returns.**
  During training the policy samples actions for exploration, so its return is
  both depressed and noisy. Evaluation uses the deterministic policy on a
  separately seeded environment.
- **Hyperparameters come from rl-baselines3-zoo, not from hand tuning here.**
  Tuning one algorithm and not the other would measure tuning effort instead of
  the algorithms. If a hyperparameter has to change, change it for a documented
  reason and say so in README.md.
- **PPO gets `VecNormalize`, SAC does not.** PPO needs observation and reward
  normalization on MuJoCo. Normalizing SAC's rewards interferes with its automatic
  entropy temperature tuning. Evaluation freezes the statistics and disables reward
  normalization so both are scored on the raw reward scale.
- **Smoothing is applied per seed, before aggregating across seeds.** Threshold
  crossings are computed on the smoothed curve, otherwise a single lucky
  evaluation counts as having reached the threshold.
- **The shaded band is the min-max range across seeds, not a t interval.** With 3
  seeds a normal approximation interval extends past returns that are physically
  impossible. Per seed lines are drawn as well.
- **Final performance averages the last window of evaluation points, never the
  single best one.** The window scales with run length.
- **Never report a single seed result as a property of an algorithm.**

## Conventions

- All code and code comments in English, without exception.
- No em dashes in any generated text, code, or documentation. Use commas, colons
  or separate sentences.
- Model checkpoints (`*.zip`, `vecnormalize.pkl`) stay out of git. Evaluation logs
  (`evaluations.npz`) and run configs (`config.json`) stay in git: they are small
  and they are the evidence behind every number reported.
- Chart colors use the validated palette in `aggregate_results.py`
  (PPO `#2a78d6`, PPO blue; SAC `#eb6834`, orange). This pair is colorblind safe.
  Do not switch to red and blue.

## Hardware and running

Apple M3 Pro, 12 cores (6 performance, 6 efficiency), 36 GB RAM, CPU only.

`run_all.sh` sets one BLAS thread per process and runs as many processes as there
are performance cores. This is deliberate: the networks are two 256 unit layers
and the MuJoCo step is single threaded, so multithreading inside a run costs more
in synchronization than it saves. Parallelism belongs across runs.

Do not suggest MPS or GPU for these MLP policies. Kernel launch overhead exceeds
the compute at this batch and layer size. SB3 selects CPU by default here, which
is correct.

A 6 run sweep at 1M steps takes about 59 minutes wall clock.

```bash
./run_all.sh                    # 3 seeds, Hopper-v5, 1M steps
SEEDS=5 ./run_all.sh            # 5 seeds
ENV=Walker2d-v5 ./run_all.sh    # another environment
python aggregate_results.py --env Hopper-v5
```

## Open work, roughly in priority order

1. Finish the 5 seed rerun and update README.md. Four places change: the status
   line, the seed count in the setup table, the final performance table, and the
   p value discussion.
2. Test the `learning_starts` hypothesis with a SAC run at `learning_starts=1000`.
3. Add a second environment, Walker2d-v5, to show the result is not specific to
   Hopper.
4. Replace mean and t interval aggregation with the interquartile mean and
   bootstrap intervals, following Agarwal et al. 2021 (`rliable`).
5. Record a short video of the best SAC policy for the README. Host it outside
   git.
6. Optional: robustness to dynamics shift. Evaluate the best policies under
   modified torso mass, friction and actuator gain, and plot return degradation
   against the size of the shift. This is the sim-to-real angle and the most
   research relevant extension.

## What not to do

- Do not add a hyperparameter search. It is out of scope and it would undermine
  the "published values for both sides" argument.
- Do not delete or rewrite `results/`. Those logs are the record of runs that
  took hours.
- Do not soften the limitations section in README.md. Naming the weaknesses is
  the point of it.
