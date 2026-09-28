"""
Aggregate multi-seed results into one learning curve plot and a summary table.

Run after run_experiment.py has finished all runs:
    python aggregate_results.py --env Hopper-v5
"""

import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

# Two-sided 95% t critical values by degrees of freedom (n_seeds - 1).
# Hardcoded to avoid a scipy dependency.
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

COLORS = {"ppo": "#1f77b4", "sac": "#d62728"}


def load_algo_results(results_dir, env_id, algo):
    """Return (timesteps, curves) where curves has shape (n_seeds, n_evals)."""
    pattern = os.path.join(results_dir, env_id, algo, "seed_*", "evaluations.npz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None, None

    timesteps, curves = None, []
    for path in paths:
        data = np.load(path)
        # results has shape (n_evals, n_eval_episodes); average over episodes
        # to get one score per evaluation point.
        curves.append(data["results"].mean(axis=1))
        if timesteps is None:
            timesteps = data["timesteps"]

    # Runs can differ by one eval point if they stopped at slightly different
    # steps, so truncate everything to the shortest.
    n = min(len(c) for c in curves)
    return timesteps[:n], np.stack([c[:n] for c in curves])


def confidence_band(curves):
    """Mean across seeds and half-width of the 95% t confidence interval."""
    n_seeds = curves.shape[0]
    mean = curves.mean(axis=0)
    if n_seeds < 2:
        return mean, np.zeros_like(mean)
    # ddof=1 is the sample standard deviation, correct when estimating from
    # a small number of seeds.
    sem = curves.std(axis=0, ddof=1) / np.sqrt(n_seeds)
    t = T_CRIT_95.get(n_seeds - 1, 1.96)
    return mean, t * sem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Hopper-v5")
    parser.add_argument("--results", default="results")
    parser.add_argument("--algos", nargs="+", default=["ppo", "sac"])
    args = parser.parse_args()

    plt.figure(figsize=(8, 4.5))
    summary_rows = []

    for algo in args.algos:
        timesteps, curves = load_algo_results(args.results, args.env, algo)
        if curves is None:
            print(f"No results found for {algo}, skipping")
            continue

        mean, half_width = confidence_band(curves)
        color = COLORS.get(algo)

        plt.plot(timesteps, mean, label=f"{algo.upper()} (n={curves.shape[0]})",
                 color=color, linewidth=2)
        plt.fill_between(timesteps, mean - half_width, mean + half_width,
                         color=color, alpha=0.2)

        # Final performance: average the last 10 evaluation points per seed,
        # then aggregate across seeds. Taking the single best point instead
        # would reward lucky evaluations and overstate the result.
        final_per_seed = curves[:, -10:].mean(axis=1)
        n_seeds = len(final_per_seed)
        sem = final_per_seed.std(ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else 0.0
        ci = T_CRIT_95.get(n_seeds - 1, 1.96) * sem

        summary_rows.append((
            algo.upper(), n_seeds, final_per_seed.mean(), ci,
            final_per_seed.min(), final_per_seed.max(),
        ))

    plt.xlabel("environment steps")
    plt.ylabel("evaluation return (deterministic, 10 episodes)")
    plt.title(f"{args.env}: mean across seeds, shaded = 95% CI")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = f"{args.env.lower()}_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}\n")

    header = f"{'algo':<6}{'seeds':>7}{'final':>12}{'95% CI':>12}{'min':>10}{'max':>10}"
    print(header)
    print("-" * len(header))
    for algo, n, mean, ci, lo, hi in summary_rows:
        print(f"{algo:<6}{n:>7}{mean:>12.1f}{ci:>12.1f}{lo:>10.1f}{hi:>10.1f}")

    print("\nNote: with 3 seeds the confidence intervals are wide. Treat large "
          "overlaps as 'no detectable difference', not as evidence of equality.")


if __name__ == "__main__":
    main()
