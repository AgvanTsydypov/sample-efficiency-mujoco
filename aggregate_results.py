"""
Aggregate multi-seed results: learning curves, final performance, and the
sample-efficiency metric (steps to reach a performance threshold).

Run after the sweep has finished:
    python aggregate_results.py --env Hopper-v5
    python aggregate_results.py --env Hopper-v5 --thresholds 1000 2000 2500 3000
"""

import argparse
import glob
import math
import os
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt

# Two-sided 95% t critical values by degrees of freedom (n_seeds - 1).
# Hardcoded to avoid a scipy dependency.
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

# Validated categorical palette, colorblind-safe on a white surface: worst
# all-pairs separation is dE 9.2 under deuteranopia, 24.0 under normal vision.
# Slot order is the safety mechanism, so take slots in order, never at random.
# Aqua sits below 3:1 contrast on white, which the direct end-of-curve labels
# cover. Named arms keep a fixed color so a figure does not repaint when an
# ablation arm is added or dropped.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]
COLORS = {"ppo": CATEGORICAL[0], "sac": CATEGORICAL[1]}
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

SMOOTH_WINDOW = 5      # evaluation points; 5 x eval_freq 10k = 50k steps
N_RESAMPLES = 10_000   # bootstrap resamples for the final-performance interval
N_RESAMPLES_CURVE = 2_000   # fewer per evaluation point, since there are many
BOOTSTRAP_SEED = 0     # fixed, so the reported intervals are reproducible


def color_for(algo, index):
    """Fixed color for the baselines, next free slot for any other arm."""
    if algo in COLORS:
        return COLORS[algo]
    used = set(COLORS.values())
    free = [c for c in CATEGORICAL if c not in used]
    return free[index % len(free)] if free else INK


def iqm(values, axis=-1):
    """Interquartile mean: the mean of the middle 50% of the runs.

    Trims 25% from each tail. With 5 runs that drops the best and the worst and
    averages the remaining three, so one collapsed or one lucky seed cannot drag
    the estimate the way it drags a mean. This is the aggregate recommended by
    Agarwal et al. (2021) for the small run counts typical of deep RL.
    """
    x = np.sort(np.asarray(values, dtype=float), axis=axis)
    n = x.shape[axis]
    k = int(np.floor(n * 0.25))
    if n - 2 * k > 0:
        x = np.take(x, np.arange(k, n - k), axis=axis)
    return x.mean(axis=axis)


def bootstrap_iqm_ci(values, level=0.95, n_resamples=N_RESAMPLES):
    """Percentile bootstrap interval for the IQM over runs.

    Resamples the runs with replacement and takes percentiles of the resulting
    IQMs. It assumes no distribution, and unlike a t interval it cannot extend
    past returns that actually occurred: an early version of this script drew a
    normal-approximation band reaching below zero, a return the environment
    cannot produce.
    """
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    stats = iqm(x[idx], axis=1)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(stats, [tail, 100 - tail])
    return float(lo), float(hi)


def iqm_curve(curves, level=0.95, n_resamples=N_RESAMPLES_CURVE):
    """IQM across seeds at every evaluation point, with a bootstrap band."""
    n_seeds = curves.shape[0]
    center = iqm(curves, axis=0)
    if n_seeds < 2:
        return center, center, center
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n_seeds, size=(n_resamples, n_seeds))
    # curves[idx] is (n_resamples, n_seeds, n_evals); aggregate over seeds.
    stats = iqm(curves[idx], axis=1)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(stats, [tail, 100 - tail], axis=0)
    return center, lo, hi


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


def smooth(curves, window):
    """Centered rolling mean along the evaluation axis, applied per seed.

    Each raw evaluation point is only 10 episodes of a high-variance policy, so
    the unsmoothed curve shows sampling noise rather than learning progress.
    Smoothing happens per seed, before any aggregation across seeds.
    """
    if window <= 1 or curves.shape[1] < window:
        return curves, slice(None)
    kernel = np.ones(window) / window
    out = np.stack([np.convolve(c, kernel, mode="valid") for c in curves])
    # 'valid' drops (window - 1) points; keep the matching timestep slice.
    trim = slice(window - 1, None)
    return out, trim


def steps_to_threshold(timesteps, curves, threshold):
    """First timestep at which each seed's smoothed curve reaches threshold.

    Returns an array with np.nan for seeds that never got there. Using the
    smoothed curve matters: on a raw curve a single lucky evaluation would
    count as 'reached', which systematically understates the step cost.
    """
    out = []
    for curve in curves:
        reached = np.flatnonzero(curve >= threshold)
        out.append(timesteps[reached[0]] if reached.size else np.nan)
    return np.array(out, dtype=float)


def censored_median(steps):
    """Median steps-to-threshold across all seeds, with censoring.

    Seeds that never reached the threshold are treated as +inf rather than
    dropped. Dropping them and taking the median of the rest would make an
    algorithm that succeeds in 3 runs out of 5 look as cheap as one that
    succeeds in 5 out of 5, which is the wrong comparison entirely.
    """
    n = len(steps)
    filled = np.sort(np.where(np.isnan(steps), np.inf, steps))
    if n % 2 == 1:
        median = filled[n // 2]
    else:
        median = 0.5 * (filled[n // 2 - 1] + filled[n // 2])
    return median, int(np.sum(~np.isnan(steps)))


def mannwhitney_exact(a, b):
    """One-sided exact Mann-Whitney test for H1: values in b exceed those in a.

    With 3 seeds per group the t interval is far too crude to be informative,
    but a rank test still is: complete separation of two groups of 3 has exact
    p = 1/C(6,3) = 0.05, the smallest p this design can produce. Enumerating
    every split is exact and needs no scipy, but only stays cheap for small n.
    """
    n_a, n_b = len(a), len(b)
    if n_a + n_b > 16:
        return None, None

    combined = np.concatenate([a, b])

    def u_stat(xa, xb):
        return sum((y > x) + 0.5 * (y == x) for x in xa for y in xb)

    observed = u_stat(a, b)
    idx = set(range(len(combined)))
    at_least_as_extreme = total = 0
    for pick in combinations(sorted(idx), n_a):
        xa = combined[list(pick)]
        xb = combined[sorted(idx - set(pick))]
        total += 1
        if u_stat(xa, xb) >= observed:
            at_least_as_extreme += 1

    return observed, at_least_as_extreme / total


def plot_curves(ax, data, env_id, band="minmax"):
    """Learning curves: per-seed lines, an uncertainty band, and a center line.

    band="minmax" draws the observed range across seeds around the mean, which
    shows the spread exactly as it occurred. band="iqm" draws the interquartile
    mean with a percentile bootstrap interval, which is the aggregate Agarwal
    et al. (2021) recommend. Neither can extend past returns that occurred.
    """
    end_labels = []

    for index, (algo, (timesteps, curves)) in enumerate(data.items()):
        color = color_for(algo, index)

        # Individual seeds, drawn thin. At five runs this is the honest view:
        # it shows the spread as it is instead of only a summary of it.
        for curve in curves:
            ax.plot(timesteps, curve, color=color, linewidth=0.7, alpha=0.35)

        if band == "iqm":
            center, lo, hi = iqm_curve(curves)
            center_label = "IQM"
        else:
            center = curves.mean(axis=0)
            lo, hi = curves.min(axis=0), curves.max(axis=0)
            center_label = "mean"

        ax.fill_between(timesteps, lo, hi, color=color, alpha=0.13, linewidth=0)
        ax.plot(timesteps, center, color=color, linewidth=2,
                label=f"{algo.upper()} (n={curves.shape[0]})")
        end_labels.append([center[-1], timesteps[-1], algo.upper(), color])

    # Direct labels at the right edge, so identity never rests on color alone.
    # Arms that end close together would print on top of each other, so nudge
    # them apart vertically once the y range is known.
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    end_labels.sort()
    for i in range(1, len(end_labels)):
        if end_labels[i][0] - end_labels[i - 1][0] < gap:
            end_labels[i][0] = end_labels[i - 1][0] + gap
    for y, x, text, color in end_labels:
        ax.annotate(text, xy=(x, y), xytext=(6, 0), textcoords="offset points",
                    color=color, fontsize=10, fontweight="bold",
                    va="center", ha="left")

    ax.set_xlabel("environment steps", color="#52514e", fontsize=10)
    ax.set_ylabel("evaluation return", color="#52514e", fontsize=10)
    subtitle = ("IQM across seeds, band = 95% bootstrap" if band == "iqm"
                else "mean across seeds, band = min-max")
    ax.set_title(f"{env_id}: {subtitle}, "
                 f"smoothed over {SMOOTH_WINDOW} evaluations",
                 color=INK, fontsize=11, loc="left", pad=12)

    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.margins(x=0.08)


def print_final_table(data, window_hint):
    rows = []
    for algo, (_, curves) in data.items():
        n_evals = curves.shape[1]
        window = max(1, min(10, n_evals // 10))
        if n_evals < 20:
            print(f"Warning: only {n_evals} eval points for {algo}, final score "
                  f"averaged over the last {window}. Run longer before reporting.")

        final = curves[:, -window:].mean(axis=1)
        n = len(final)
        sem = final.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        ci = T_CRIT_95.get(n - 1, 1.96) * sem
        point = iqm(final)
        lo, hi = bootstrap_iqm_ci(final)
        rows.append((algo, n, window, final, point, (lo, hi), final.mean(), ci))

    # Width follows the longest arm name, so an ablation label such as
    # "sac_ls1000" does not push the numeric columns out of alignment.
    name_col = max(6, max(len(algo) for algo, *_ in rows) + 2)
    header = (f"{'arm':<{name_col}}{'seeds':>7}{'IQM':>9}"
              f"{'95% CI (bootstrap)':>22}{'mean':>9}{'95% CI (t)':>13}"
              f"{'min':>9}{'max':>9}")
    print("\nFINAL PERFORMANCE")
    print(header)
    print("-" * len(header))
    for algo, n, window, final, point, (lo, hi), mean, ci in rows:
        band = f"[{lo:.0f}, {hi:.0f}]"
        print(f"{algo.upper():<{name_col}}{n:>7}{point:>9.1f}{band:>22}"
              f"{mean:>9.1f}{('+/- %.1f' % ci):>13}"
              f"{final.min():>9.1f}{final.max():>9.1f}")

    print(f"\nScores are the mean of each seed's last {rows[0][2]} evaluation points.")
    print("IQM is the mean of the middle 50% of seeds, with a percentile bootstrap "
          "interval.\nThe mean and t interval are kept beside it for comparison; "
          "where the two\ndisagree, the mean is being moved by one extreme seed.")

    min_n = min(r[1] for r in rows)
    if min_n < 10:
        print(f"\nWith {min_n} seeds the bootstrap resamples a very small set and its "
              "interval is\noptimistic. Treat it as a better-behaved interval than the "
              "t one, not a tight one.")
    return rows


def print_separation(rows):
    """Compare the two algorithms by rank, not only by overlapping intervals."""
    if len(rows) != 2:
        return

    (algo_a, _, _, final_a, point_a, *_), (algo_b, _, _, final_b, point_b, *_) = rows
    if point_b < point_a:
        algo_a, algo_b = algo_b, algo_a
        final_a, final_b = final_b, final_a

    print("\nRANK COMPARISON")
    if final_b.min() > final_a.max():
        print(f"Every {algo_b.upper()} seed scored above every {algo_a.upper()} "
              f"seed ({final_b.min():.1f} > {final_a.max():.1f}).")
    else:
        print(f"{algo_b.upper()} and {algo_a.upper()} seed scores interleave.")

    u, p = mannwhitney_exact(final_a, final_b)
    if p is None:
        print("Too many seeds for exact enumeration, use scipy.stats.mannwhitneyu.")
        return

    n_a, n_b = len(final_a), len(final_b)
    floor = 1 / math.comb(n_a + n_b, n_a)

    # U counts the seed pairs in which one arm beat the other, so U / (n_a * n_b)
    # estimates P(a run of B beats a run of A). Agarwal et al. (2021) report this
    # as probability of improvement; it says how often, not by how much.
    prob = u / (n_a * n_b)
    print(f"P({algo_b.upper()} run beats {algo_a.upper()} run) = {prob:.2f} "
          f"({u:.0f} of {n_a * n_b} seed pairs)")
    print(f"One-sided exact Mann-Whitney ({algo_b.upper()} > {algo_a.upper()}): "
          f"U = {u:.1f}, p = {p:.4f}")
    print(f"Smallest p this design can produce with {n_a} vs {n_b} seeds: {floor:.4f}. "
          f"More seeds are the only way below it.")


def print_threshold_table(data, thresholds):
    """Sample efficiency: environment steps needed to reach each threshold."""
    print("\nSAMPLE EFFICIENCY (steps to first reach threshold, smoothed curve)")
    col = max(18, max(len(algo) for algo in data) + 4)
    header = f"{'threshold':>10}" + "".join(f"{algo.upper():>{col}}" for algo in data)
    print(header)
    print("-" * len(header))

    budget = max(timesteps[-1] for timesteps, _ in data.values())

    for threshold in thresholds:
        cells = []
        for algo, (timesteps, curves) in data.items():
            steps = steps_to_threshold(timesteps, curves, threshold)
            median, n_reached = censored_median(steps)
            n_seeds = len(steps)

            if np.isinf(median):
                cell = f"> {budget / 1000:.0f}k"
            else:
                cell = f"{median / 1000:.0f}k"
            if n_reached < n_seeds:
                cell += f" ({n_reached}/{n_seeds})"
            cells.append(f"{cell:>{col}}")
        print(f"{threshold:>10.0f}" + "".join(cells))

    print("\nMedian across all seeds. Seeds that never reached the threshold are "
          "censored at the budget, not dropped, so a count like (3/5) raises the "
          "reported cost instead of hiding it. '> 1000k' means more than half the "
          "seeds never got there within the budget.")


def main():
    global SMOOTH_WINDOW

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Hopper-v5")
    parser.add_argument("--results", default="results")
    parser.add_argument("--algos", nargs="+", default=["ppo", "sac"])
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[1000, 2000, 2500, 3000])
    parser.add_argument("--smooth", type=int, default=SMOOTH_WINDOW)
    parser.add_argument("--band", choices=["minmax", "iqm"], default="minmax",
                        help="minmax draws the observed range around the mean; "
                             "iqm draws the interquartile mean with a bootstrap "
                             "interval.")
    args = parser.parse_args()

    SMOOTH_WINDOW = args.smooth

    raw, smoothed = {}, {}
    for algo in args.algos:
        timesteps, curves = load_algo_results(args.results, args.env, algo)
        if curves is None:
            print(f"No results found for {algo}, skipping")
            continue
        raw[algo] = (timesteps, curves)
        sm, trim = smooth(curves, args.smooth)
        smoothed[algo] = (timesteps[trim], sm)

    if not raw:
        print("Nothing to aggregate.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    plot_curves(ax, smoothed, args.env, band=args.band)
    fig.tight_layout()

    suffix = "_iqm" if args.band == "iqm" else ""
    out_path = f"{args.env.lower()}_comparison{suffix}.png"
    fig.savefig(out_path, dpi=200, facecolor="#ffffff")
    print(f"Plot saved to {out_path}")

    rows = print_final_table(raw, args.smooth)
    print_separation(rows)
    print_threshold_table(smoothed, args.thresholds)

    max_seeds = max(r[1] for r in rows)
    if max_seeds < 5:
        print(f"\n{max_seeds} seeds is the bare minimum. Rerun with SEEDS=5 before "
              "reporting this anywhere.")


if __name__ == "__main__":
    main()
