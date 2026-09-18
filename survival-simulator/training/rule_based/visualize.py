"""
Visualize a rule_based sweep: reads training/rule_based/results/ (produced by
sweep.py) and writes PNGs to training/rule_based/results/plots/.

Uses a fixed, colorblind-safe categorical palette (Okabe-Ito) assigned by what
each color MEANS (death cause, score component), never cycled -- so "predator"
is always the same color across every chart.

Usage:
    python3 training/rule_based/visualize.py
"""
import glob
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Okabe-Ito colorblind-safe palette. Assigned by meaning, fixed everywhere.
INK = "#2B2B2B"
MUTED_INK = "#6B6B6B"
GRID = "#DDDDDD"
BLUE = "#0072B2"       # time bonus / generic primary
ORANGE = "#E69F00"     # starvation (old age)
GREEN = "#009E73"      # fruit bonus
VERMILLION = "#D55E00" # predator deaths / predation penalty
PURPLE = "#CC79A7"     # starvation (young)
SKYBLUE = "#56B4E9"    # secondary / individual seed traces

CAUSE_COLORS = {
    "predator": VERMILLION,
    "starvation_old": ORANGE,
    "starvation_young": PURPLE,
}
CAUSE_LABELS = {
    "predator": "Eaten by predator",
    "starvation_old": "Starved (past max_age)",
    "starvation_young": "Starved (young)",
}
CAUSE_ORDER = ["predator", "starvation_old", "starvation_young"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED_INK,
    "ytick.color": MUTED_INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


def load_sweep():
    """
    Only loads the seeds present in the CURRENT sweep_summary.csv, even if
    results/ also contains run_seed* folders from older/unrelated runs (e.g.
    leftover smoke tests, or a folder that couldn't be deleted). Without this
    filter, stray folders silently get averaged into every chart.
    """
    summary_path = os.path.join(RESULTS_DIR, "sweep_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No {summary_path} -- run sweep.py first.")
    summary = pd.read_csv(summary_path)
    wanted_seeds = set(summary["seed"].tolist())

    run_dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "run_seed*")))
    ticks_by_seed = {}
    deaths_by_seed = {}
    skipped = []
    for d in run_dirs:
        seed = int(os.path.basename(d).replace("run_seed", ""))
        if seed not in wanted_seeds:
            skipped.append(seed)
            continue
        ticks_path = os.path.join(d, "ticks.csv")
        deaths_path = os.path.join(d, "deaths.csv")
        if os.path.exists(ticks_path):
            ticks_by_seed[seed] = pd.read_csv(ticks_path)
        if os.path.exists(deaths_path):
            deaths_by_seed[seed] = pd.read_csv(deaths_path)
    if skipped:
        print(f"Ignoring {len(skipped)} run_seed* folder(s) not in sweep_summary.csv: {skipped}")
    return summary, ticks_by_seed, deaths_by_seed


def plot_extinction_time_distribution(summary, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    times = summary["extinction_time"].dropna()
    survived = summary["extinct"].eq(False).sum()

    bins = max(8, min(30, len(times) // 2 or 1))
    ax.hist(times, bins=bins, color=BLUE, edgecolor="white", linewidth=0.8)
    ax.axvline(times.median(), color=VERMILLION, linewidth=2, linestyle="--",
               label=f"median = {times.median():.0f}s")
    ax.set_xlabel("Extinction time (simulated seconds)")
    ax.set_ylabel("Number of seeds")
    title = f"Survival time across {len(summary)} seeds"
    if survived:
        title += f"  ({survived} survived the full run)"
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_population_over_time(ticks_by_seed, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Individual seed traces: light, so the shape of the ensemble reads clearly.
    for seed, df in ticks_by_seed.items():
        ax.plot(df["sim_time"], df["num_agents"], color=SKYBLUE, alpha=0.35, linewidth=1)

    # Mean trace, on a common time grid (seeds end at different times as they
    # go extinct, so align on the union of observed times and average only
    # over seeds still running at that time).
    if ticks_by_seed:
        max_t = max(df["sim_time"].max() for df in ticks_by_seed.values())
        grid = np.arange(0, max_t + 1, 1.0)
        stacked = []
        for df in ticks_by_seed.values():
            interp = np.interp(grid, df["sim_time"], df["num_agents"], right=0)
            stacked.append(interp)
        mean_pop = np.mean(stacked, axis=0)
        ax.plot(grid, mean_pop, color=VERMILLION, linewidth=2.2, label="Mean across seeds")

    ax.set_xlabel("Simulated time (s)")
    ax.set_ylabel("Living agents")
    ax.set_title(f"Population over time -- {len(ticks_by_seed)} seeds", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_death_cause_breakdown(summary, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    totals = {c: summary[f"deaths_{c}"].sum() for c in CAUSE_ORDER}
    total_deaths = sum(totals.values()) or 1

    labels = [CAUSE_LABELS[c] for c in CAUSE_ORDER]
    values = [totals[c] for c in CAUSE_ORDER]
    colors = [CAUSE_COLORS[c] for c in CAUSE_ORDER]

    bars = ax.bar(labels, values, color=colors, width=0.6)
    for bar, v in zip(bars, values):
        pct = 100 * v / total_deaths
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:,}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=10, color=INK)

    ax.set_ylabel("Total deaths across all seeds")
    ax.set_title(f"What kills the colony -- {total_deaths:,} deaths across {len(summary)} seeds",
                 loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_score_components(ticks_by_seed, summary, out_path):
    """Cumulative score, decomposed into its three additive terms, for the
    single seed closest to the median extinction time (a representative run,
    not an average -- averaging the decomposition across seeds of different
    lengths would be misleading)."""
    finite = summary.dropna(subset=["extinction_time"])
    if finite.empty:
        return
    median_time = finite["extinction_time"].median()
    rep_seed = int(finite.iloc[(finite["extinction_time"] - median_time).abs().argsort().iloc[0]]["seed"])
    df = ticks_by_seed[rep_seed]

    cum_time = df["time_bonus"].cumsum()
    cum_fruit = df["fruit_bonus"].cumsum()
    cum_pred = -df["predation_penalty"].cumsum()  # already negative-in-spirit; plot as area below 0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["sim_time"], cum_time, color=BLUE, linewidth=2, label="Time survived (+dt/tick)")
    ax.plot(df["sim_time"], cum_fruit, color=GREEN, linewidth=2, label="Fruit eaten (+energy/1000)")
    ax.plot(df["sim_time"], cum_pred, color=VERMILLION, linewidth=2, label="Predation penalty (-energy/100)")
    ax.plot(df["sim_time"], df["score"], color=INK, linewidth=1.4, linestyle="--", label="Total score")
    ax.axhline(0, color=GRID, linewidth=1)

    ax.set_xlabel("Simulated time (s)")
    ax.set_ylabel("Cumulative contribution to score")
    ax.set_title(f"Score is time, mostly -- seed {rep_seed} (extinction at {df['sim_time'].max():.0f}s)",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_energy_over_time(ticks_by_seed, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for seed, df in ticks_by_seed.items():
        ax.plot(df["sim_time"], df["mean_energy"], color=SKYBLUE, alpha=0.35, linewidth=1)

    if ticks_by_seed:
        max_t = max(df["sim_time"].max() for df in ticks_by_seed.values())
        grid = np.arange(0, max_t + 1, 1.0)
        stacked = [np.interp(grid, df["sim_time"], df["mean_energy"]) for df in ticks_by_seed.values()]
        mean_energy = np.mean(stacked, axis=0)
        ax.plot(grid, mean_energy, color=VERMILLION, linewidth=2.2, label="Mean across seeds")

    ax.set_xlabel("Simulated time (s)")
    ax.set_ylabel("Mean agent energy")
    ax.set_title("Colony energy over time", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_death_cause_by_time_bucket(deaths_by_seed, out_path, bucket_size: float = 100.0, min_seeds: int = 5):
    """
    Death cause as a SHARE of deaths within each sim-time bucket, pooled across
    all seeds -- answers "does predator pressure actually ramp up over the
    course of a run" (it does, structurally: predator spawn chance grows with
    env.time), separately from the whole-run totals in
    plot_death_cause_breakdown(), which mixes early- and late-run behavior
    together and can't show a trend within a run.

    Late buckets have few seeds still alive to contribute a death at all
    (extinction has usually already happened), so their share is noisy --
    buckets with fewer than `min_seeds` contributing seeds are dropped from
    the top panel rather than shown misleadingly. The bottom panel plots that
    seed count directly (as its own axis, not a second y-axis on the same
    plot) so the reliability of each point is visible, not hidden.
    """
    if not deaths_by_seed:
        return
    all_deaths = pd.concat(
        [df.assign(seed=seed) for seed, df in deaths_by_seed.items()],
        ignore_index=True,
    )
    if all_deaths.empty:
        return

    all_deaths["time_bucket"] = (all_deaths["sim_time"] // bucket_size * bucket_size).astype(int)
    pivot = all_deaths.pivot_table(index="time_bucket", columns="cause", values="agent_id",
                                    aggfunc="count", fill_value=0)
    for c in CAUSE_ORDER:
        if c not in pivot.columns:
            pivot[c] = 0
    pivot["total"] = pivot[CAUSE_ORDER].sum(axis=1)
    n_seeds = all_deaths.groupby("time_bucket")["seed"].nunique()
    pivot["n_seeds"] = n_seeds

    reliable = pivot[pivot["n_seeds"] >= min_seeds].copy()
    if reliable.empty:
        return
    for c in CAUSE_ORDER:
        reliable[f"{c}_share"] = reliable[c] / reliable["total"] * 100

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(8, 6.5), sharex=True, height_ratios=[3, 1]
    )

    x = reliable.index + bucket_size / 2  # bucket midpoints
    for c in CAUSE_ORDER:
        ax_top.plot(x, reliable[f"{c}_share"], color=CAUSE_COLORS[c], linewidth=2.2,
                    marker="o", markersize=4, label=CAUSE_LABELS[c])
    ax_top.set_ylabel("Share of deaths in this window (%)")
    ax_top.set_title("Does predator pressure ramp up over a run?", loc="left", fontweight="bold")
    ax_top.text(0.0, 1.06, f"Buckets with fewer than {min_seeds} seeds still alive are dropped",
                transform=ax_top.transAxes, fontsize=9, color=MUTED_INK)
    ax_top.legend(frameon=False, fontsize=9)
    ax_top.set_ylim(0, 100)

    ax_bottom.bar(x, reliable["n_seeds"], width=bucket_size * 0.8, color=SKYBLUE)
    ax_bottom.set_ylabel("Seeds\nstill alive", fontsize=9)
    ax_bottom.set_xlabel("Simulated time (s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    summary, ticks_by_seed, deaths_by_seed = load_sweep()

    plot_extinction_time_distribution(summary, os.path.join(PLOTS_DIR, "extinction_time_distribution.png"))
    plot_population_over_time(ticks_by_seed, os.path.join(PLOTS_DIR, "population_over_time.png"))
    plot_death_cause_breakdown(summary, os.path.join(PLOTS_DIR, "death_cause_breakdown.png"))
    plot_death_cause_by_time_bucket(deaths_by_seed, os.path.join(PLOTS_DIR, "death_cause_by_time_bucket.png"))
    plot_score_components(ticks_by_seed, summary, os.path.join(PLOTS_DIR, "score_components.png"))
    plot_energy_over_time(ticks_by_seed, os.path.join(PLOTS_DIR, "energy_over_time.png"))

    print(f"Wrote 6 charts to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
