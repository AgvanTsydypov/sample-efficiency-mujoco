"""
Robustness of trained policies to a shift in the simulator's dynamics.

Reloads the saved checkpoints, changes one physical parameter of the MuJoCo
model at a time, and measures how much return survives. Nothing is retrained:
the question is whether a policy fitted to one set of dynamics still works when
that set is wrong, which is the situation every sim-to-real transfer is in.

Install:
    pip install "stable-baselines3[extra]>=2.4.0" "gymnasium[mujoco]" matplotlib

Examples:
    python robustness.py --env Hopper-v5
    python robustness.py --env Walker2d-v5 --episodes 20
    python robustness.py --env Hopper-v5 --reuse        # replot, no evaluation
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ALGO_CLASSES = {"ppo": PPO, "sac": SAC}

# One physical property per panel, each a plausible way a real robot differs
# from the model it was trained in.
PERTURBATIONS = {
    "mass": "link mass and inertia",
    "friction": "ground friction",
    "gear": "actuator strength",
}

DEFAULT_FACTORS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]
COLORS = {"ppo": CATEGORICAL[0], "sac": CATEGORICAL[1]}
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

N_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0


def iqm(values, axis=-1):
    """Mean of the middle 50% of the runs, as used everywhere else in this repo."""
    x = np.sort(np.asarray(values, dtype=float), axis=axis)
    n = x.shape[axis]
    k = int(np.floor(n * 0.25))
    if n - 2 * k > 0:
        x = np.take(x, np.arange(k, n - k), axis=axis)
    return x.mean(axis=axis)


def bootstrap_iqm_ci(values, level=0.95, n_resamples=N_RESAMPLES):
    """Percentile bootstrap interval for the IQM across seeds."""
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    stats = iqm(x[idx], axis=1)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(stats, [tail, 100 - tail])
    return float(lo), float(hi)


def perturbed_env(env_id, kind, factor):
    """A fresh environment with one property scaled away from its nominal value.

    The model is edited after construction, so every call starts from the
    pristine values and the factor is never applied twice. Mass and inertia
    scale together: changing one without the other would describe a body whose
    density and shape disagree, which is not a physical object.
    """
    env = gym.make(env_id)
    model = env.unwrapped.model

    if kind == "mass":
        model.body_mass[:] = model.body_mass * factor
        model.body_inertia[:] = model.body_inertia * factor
    elif kind == "friction":
        # Column 0 is sliding friction; torsional and rolling are left alone.
        model.geom_friction[:, 0] = model.geom_friction[:, 0] * factor
    elif kind == "gear":
        # Actuator gear converts control signal to joint torque, so scaling it
        # is the simplest stand-in for a weaker or stronger motor.
        model.actuator_gear[:, 0] = model.actuator_gear[:, 0] * factor
    else:
        raise SystemExit(f"Unknown perturbation '{kind}'")

    return env


def load_policy(results_dir, env_id, arm, seed, which="best"):
    """A saved checkpoint for one run, plus PPO's saved observation statistics.

    "best" is the checkpoint the evaluation callback kept, which is the policy
    anyone would actually deploy, so it is the default. "final" is the policy at
    the end of training and matches the final-performance figures reported
    elsewhere. They are different policies and score differently.
    """
    run_dir = os.path.join(results_dir, env_id, arm, f"seed_{seed}")

    order = ("best_model", "final_model") if which == "best" else ("final_model",)
    checkpoint = None
    for name in order:
        if os.path.exists(os.path.join(run_dir, f"{name}.zip")):
            checkpoint = os.path.join(run_dir, name)
            break
    if checkpoint is None:
        raise SystemExit(f"No {which} checkpoint in {run_dir}")

    model = ALGO_CLASSES[arm.split("_")[0]].load(checkpoint)

    # Without this, PPO sees observations on a scale it never trained on and
    # the whole sweep measures a broken policy rather than a shifted world.
    normalizer = None
    stats = os.path.join(run_dir, "vecnormalize.pkl")
    if os.path.exists(stats):
        normalizer = VecNormalize.load(stats, DummyVecEnv([lambda: gym.make(env_id)]))
        normalizer.training = False
        normalizer.norm_reward = False

    return model, normalizer


def evaluate(model, normalizer, env, episodes, seed):
    """Mean return over episodes with the deterministic policy."""
    returns = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done, total = False, 0.0
        while not done:
            model_obs = normalizer.normalize_obs(obs) if normalizer else obs
            action, _ = model.predict(model_obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns))


def seeds_available(results_dir, env_id, arm):
    base = os.path.join(results_dir, env_id, arm)
    if not os.path.isdir(base):
        raise SystemExit(f"No results for arm '{arm}' under {base}")
    return sorted(int(d.split("_")[1]) for d in os.listdir(base)
                  if d.startswith("seed_"))


def run_sweep(env_id, arms, factors, episodes, results_dir, which="best"):
    """Return scores[arm][kind] with shape (n_factors, n_seeds)."""
    scores = {}
    for arm in arms:
        seeds = seeds_available(results_dir, env_id, arm)
        print(f"\n{arm.upper()}: seeds {seeds}")
        scores[arm] = {}

        policies = {s: load_policy(results_dir, env_id, arm, s, which) for s in seeds}

        for kind in PERTURBATIONS:
            grid = np.zeros((len(factors), len(seeds)))
            for i, factor in enumerate(factors):
                for j, seed in enumerate(seeds):
                    model, normalizer = policies[seed]
                    env = perturbed_env(env_id, kind, factor)
                    # Offset the evaluation seed so these are not the episodes
                    # the checkpoint was selected on.
                    grid[i, j] = evaluate(model, normalizer, env, episodes,
                                          seed=seed + 50_000)
                    env.close()
                print(f"  {kind:<9} x{factor:<5} "
                      f"IQM {iqm(grid[i]):>7.1f}   per seed "
                      f"{np.array2string(grid[i], precision=0, floatmode='fixed')}")
            scores[arm][kind] = grid
    return scores


def sanity_check(scores, factors):
    """At factor 1.0 the model is untouched, so all three panels must be equal.

    This is not three independent estimates: the policy is deterministic, the
    environment and the episode seeds are the same, and MuJoCo is deterministic,
    so identical numbers are what correct code produces. Any spread at all means
    a perturbation is leaking into the nominal setting.

    These numbers are also not the final-performance figures reported elsewhere.
    Those average each seed's last evaluation points, while this evaluates the
    best checkpoint, which is the one you would actually deploy. The best
    checkpoint usually scores higher, and on a noisy task it can score lower,
    because both are estimates from a finite number of episodes.
    """
    if 1.0 not in factors:
        return
    row = factors.index(1.0)
    print("\nNOMINAL CHECK (factor 1.0: all three panels must be identical)")
    for arm, kinds in scores.items():
        values = [iqm(grid[row]) for grid in kinds.values()]
        spread = max(values) - min(values)
        print(f"  {arm.upper():<11} " +
              "  ".join(f"{k}={v:.0f}" for k, v in zip(kinds, values)) +
              f"   spread {spread:.0f}")
        if spread > 1e-6:
            print("    Nonzero spread. A perturbation is being applied at factor "
                  "1.0; fix that before reading anything else here.")


def normalized(grid, factors):
    """Return as a fraction of the same seed's score at factor 1.0."""
    row = factors.index(1.0)
    baseline = grid[row]
    return grid / np.where(baseline == 0, np.nan, baseline)


def plot(scores, factors, env_id, out_path):
    kinds = list(PERTURBATIONS)
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.2 * len(kinds), 4.2),
                             sharey=True)
    fig.patch.set_facecolor("#ffffff")

    for ax, kind in zip(axes, kinds):
        ax.set_facecolor("#ffffff")
        for arm, kinds_data in scores.items():
            color = COLORS.get(arm, INK)
            frac = normalized(kinds_data[kind], factors)

            center = np.array([iqm(frac[i]) for i in range(len(factors))])
            band = np.array([bootstrap_iqm_ci(frac[i]) for i in range(len(factors))])

            ax.plot(factors, center, color=color, linewidth=2, marker="o",
                    markersize=5, label=arm.upper())
            ax.fill_between(factors, band[:, 0], band[:, 1],
                            color=color, alpha=0.13, linewidth=0)

        # The nominal setting, where every curve passes through 1.0 by definition.
        ax.axvline(1.0, color=INK_MUTED, linewidth=1, linestyle=":", zorder=0)
        ax.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle=":", zorder=0)

        ax.set_xscale("log")
        ax.set_xticks(factors)
        ax.set_xticklabels([f"{f:g}x" for f in factors])
        ax.minorticks_off()
        ax.set_xlabel(PERTURBATIONS[kind], color="#52514e", fontsize=10)
        ax.set_title(kind, color=INK, fontsize=11, loc="left", pad=10)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors=INK_MUTED, labelsize=9)

    axes[0].set_ylabel("return kept, relative to nominal dynamics",
                       color="#52514e", fontsize=10)

    # Figure-level legend, so it cannot land on top of a curve in any panel.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=len(labels),
               loc="upper right", bbox_to_anchor=(0.995, 0.99))
    fig.suptitle(f"{env_id}: robustness to a shift in one physical parameter, "
                 f"IQM across seeds, band = 95% bootstrap",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=200, facecolor="#ffffff")
    print(f"\nPlot saved to {out_path}")


def print_table(scores, factors):
    """Relative and absolute returns, because they can disagree.

    Relative retention says how brittle a policy is next to its own nominal
    self. Absolute return says whether it still does the job. An arm that starts
    higher can keep a smaller fraction and still be the better policy under the
    shift, so reporting only the fraction would be misleading.
    """
    print("\nRETURN KEPT AT EACH SHIFT (IQM across seeds, 1.00 = nominal)")
    for kind in PERTURBATIONS:
        header = f"{kind:<11}" + "".join(f"{f:g}x".rjust(9) for f in factors)
        print("\n" + header)
        print("-" * len(header))
        for arm, kinds_data in scores.items():
            frac = normalized(kinds_data[kind], factors)
            cells = "".join(f"{iqm(frac[i]):>9.2f}" for i in range(len(factors)))
            print(f"{arm.upper():<11}{cells}")

    print("\n\nABSOLUTE RETURN AT EACH SHIFT (IQM across seeds)")
    for kind in PERTURBATIONS:
        header = f"{kind:<11}" + "".join(f"{f:g}x".rjust(9) for f in factors)
        print("\n" + header)
        print("-" * len(header))
        for arm, kinds_data in scores.items():
            grid = kinds_data[kind]
            cells = "".join(f"{iqm(grid[i]):>9.0f}" for i in range(len(factors)))
            print(f"{arm.upper():<11}{cells}")

    if len(scores) == 2:
        count_wins(scores, factors)


def count_wins(scores, factors):
    """How often each arm holds up better, off the nominal setting.

    A descriptive tally, not a test. With five seeds and returns that collapse
    rather than decay, single cells are noisy; the tally is only worth reading
    when it leans heavily one way.
    """
    (arm_a, data_a), (arm_b, data_b) = scores.items()
    rows = [i for i, f in enumerate(factors) if f != 1.0]

    print("\n\nWHICH ARM HOLDS UP BETTER, OFF NOMINAL (descriptive tally)")
    print(f"{'':<11}{'by fraction kept':>20}{'by absolute return':>22}")
    totals = {arm_a: [0, 0], arm_b: [0, 0]}

    for kind in PERTURBATIONS:
        frac_a = normalized(data_a[kind], factors)
        frac_b = normalized(data_b[kind], factors)
        rel = sum(iqm(frac_a[i]) > iqm(frac_b[i]) for i in rows)
        absolute_wins = sum(iqm(data_a[kind][i]) > iqm(data_b[kind][i]) for i in rows)
        totals[arm_a][0] += rel
        totals[arm_a][1] += absolute_wins
        totals[arm_b][0] += len(rows) - rel
        totals[arm_b][1] += len(rows) - absolute_wins
        print(f"{kind:<11}{f'{arm_a.upper()} {rel}/{len(rows)}':>20}"
              f"{f'{arm_a.upper()} {absolute_wins}/{len(rows)}':>22}")

    n = len(rows) * len(PERTURBATIONS)
    print("-" * 53)
    for arm, (rel, absolute) in totals.items():
        print(f"{arm.upper():<11}{f'{rel}/{n}':>20}{f'{absolute}/{n}':>22}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Hopper-v5")
    parser.add_argument("--arms", nargs="+", default=["ppo", "sac"])
    parser.add_argument("--factors", nargs="+", type=float, default=DEFAULT_FACTORS)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="robustness")
    parser.add_argument("--checkpoint", choices=["best", "final"], default="best",
                        help="best is the deployed policy and the default; final "
                             "matches the final-performance figures in README.md")
    parser.add_argument("--reuse", action="store_true",
                        help="replot from the saved sweep instead of rerunning it")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    store = os.path.join(args.out, f"{args.env.lower()}_robustness.npz")
    factors = list(args.factors)

    if args.reuse and os.path.exists(store):
        blob = np.load(store)
        factors = list(blob["factors"])
        scores = {arm: {kind: blob[f"{arm}__{kind}"] for kind in PERTURBATIONS}
                  for arm in args.arms}
        print(f"Reusing {store}")
    else:
        print(f"{args.env}: {len(args.arms)} arms x {len(PERTURBATIONS)} parameters "
              f"x {len(factors)} factors x {args.episodes} episodes per seed")
        scores = run_sweep(args.env, args.arms, factors, args.episodes,
                           args.results, args.checkpoint)

        flat = {f"{arm}__{kind}": grid
                for arm, kinds in scores.items() for kind, grid in kinds.items()}
        np.savez(store, factors=np.array(factors), **flat)
        with open(os.path.join(args.out, f"{args.env.lower()}_config.json"), "w") as f:
            json.dump({"env": args.env, "arms": args.arms, "factors": factors,
                       "checkpoint": args.checkpoint,
                       "episodes_per_seed": args.episodes,
                       "perturbations": PERTURBATIONS}, f, indent=2)
        print(f"\nSaved {store}")

    sanity_check(scores, factors)
    print_table(scores, factors)
    plot(scores, factors, args.env,
         os.path.join(args.out, f"{args.env.lower()}_robustness.png"))


if __name__ == "__main__":
    main()
