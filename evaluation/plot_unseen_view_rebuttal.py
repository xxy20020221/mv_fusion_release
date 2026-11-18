#!/usr/bin/env python3
"""Rebuttal figure for the unseen-view question on one case (raw.npz of experiment_unseen_view_disagreement.py).

(a) Per-cell disagreement D (Eq. 10) under a correct edit, by visibility group, with mean and 95%
    bootstrap CI; dashed line = conflict threshold tau_C (Eq. 12).
(b) For one-view cells: how wrong the unseen view's guess is (its x0 error / the seeing view's)
    against D. D grows only when the unseen view guesses wrong.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_unseen_view_disagreement import pair_groups  # noqa: E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NEUTRAL, RIGHT, WRONG = "#8a8984", "#2a78d6", "#eb6834"   # reference gray + categorical slots 1, 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--pair", default="0-1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    d = np.load(a.raw)
    i, j = a.pair.split("-")
    g = pair_groups(d, a.pair)
    Dm, m = d[f"D_{a.pair}"], d["in_mask"]
    ref = Dm[g["both"]].mean()
    tau = (Dm[m].mean() + Dm[m].std()) / ref
    ratio = np.where(g["only_i"], d[f"guess_err_{j}"] / d[f"guess_err_{i}"], d[f"guess_err_{i}"] / d[f"guess_err_{j}"])
    groups = [("Seen by\nboth views", g["both"], NEUTRAL),
              ("Seen by one view;\nunseen view\nguessed right", g["one_right"], RIGHT),
              ("Seen by one view;\nunseen view\nguessed wrong", g["one_wrong"], WRONG)]
    rng = np.random.default_rng(0)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0), gridspec_kw={"width_ratios": [1.1, 1]}, facecolor=SURF)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURF)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    # (a) strip + mean/CI
    for k, (name, sel, col) in enumerate(groups):
        y = Dm[sel] / ref
        x = k + rng.uniform(-0.18, 0.18, len(y))
        ax1.scatter(x, y, s=22, color=col, alpha=0.55, edgecolors=SURF, linewidths=0.6, zorder=2)
        boot = [rng.choice(y, len(y)).mean() for _ in range(5000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ax1.plot([k + 0.3, k + 0.3], [lo, hi], color=INK, lw=2, solid_capstyle="round", zorder=3)
        ax1.scatter([k + 0.3], [y.mean()], s=60, color=col, edgecolors=INK, linewidths=1.2, zorder=4)
        ax1.text(k + 0.38, y.mean(), f"{y.mean():.2f}", va="center", color=INK, fontsize=9)
        ax1.text(k, -0.02, f"n = {len(y)}", transform=ax1.get_xaxis_transform(), ha="center", va="top",
                 color=INK2, fontsize=8.5)
    ax1.axhline(tau, color=INK2, lw=1, ls="--", zorder=1)
    ax1.text(2.62, tau, "conflict\nthreshold τ$_C$", va="center", color=INK2, fontsize=8.5)
    ax1.set_xticks(range(3), [n for n, _, _ in groups], fontsize=8.5)
    ax1.tick_params(axis="x", pad=16)
    ax1.set_xlim(-0.5, 3.0)
    ax1.set_ylabel("Disagreement D (relative to 'seen by both')")
    p_rw = mannwhitneyu(Dm[g["one_wrong"]], Dm[g["one_right"]], alternative="greater").pvalue
    p_rb = mannwhitneyu(Dm[g["one_right"]], Dm[g["both"]]).pvalue
    ax1.set_title(f"(a) Edit already correct: D by visibility\nright vs both: p = {p_rb:.2f}   wrong > right: p = {p_rw:.3f}",
                  loc="left", fontsize=10, color=INK)

    # (b) guess error vs D, one-view cells
    one = g["one_right"] | g["one_wrong"]
    xr, yd = ratio[one], Dm[one] / ref
    cols = np.where(xr <= 1, RIGHT, WRONG)
    ax2.scatter(xr, yd, s=26, c=cols, alpha=0.75, edgecolors=SURF, linewidths=0.6, zorder=2)
    ax2.axvline(1.0, color=INK2, lw=1, ls=":", zorder=1)
    ax2.axhline(tau, color=INK2, lw=1, ls="--", zorder=1)
    coef = np.polyfit(xr, yd, 1)
    xs = np.linspace(xr.min(), xr.max(), 50)
    ax2.plot(xs, np.polyval(coef, xs), color=INK, lw=1.5, zorder=3)
    rho, p = spearmanr(xr, yd)
    ax2.text(0.99, 0.02, "← guessed right", transform=ax2.get_xaxis_transform(), ha="right", va="bottom",
             color=INK2, fontsize=8.5)
    ax2.text(1.01, 0.02, "guessed wrong →", transform=ax2.get_xaxis_transform(), ha="left", va="bottom",
             color=INK2, fontsize=8.5)
    ax2.text(xr.max(), tau, "τ$_C$", va="bottom", ha="right", color=INK2, fontsize=8.5)
    ax2.set_xlabel("Unseen view's x$_0$ error / seeing view's x$_0$ error")
    ax2.set_ylabel("Disagreement D (relative to 'seen by both')")
    ax2.set_title(f"(b) One-view cells: D grows with the unseen view's guess error\nSpearman ρ = {rho:.2f} (p = {p:.3f}, n = {one.sum()})",
                  loc="left", fontsize=10, color=INK)
    if a.title:
        fig.suptitle(a.title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, facecolor=SURF)
    fig.savefig(a.out.with_suffix(".pdf"), facecolor=SURF)
    print(a.out)


if __name__ == "__main__":
    main()
