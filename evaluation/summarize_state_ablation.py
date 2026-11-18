#!/usr/bin/env python3
"""Table of the three-state vs two-state ablation (evaluate_state_ablation.py outputs of several items / seeds).

Per case, metrics are the mean over all scored views (inputs + held-out) and over the given seed
replicates. "Wins" counts case x seed pairs where the three-state variant has the lower LPIPS;
"seed spread" is the mean |seed1 - seed2| LPIPS of the same variant, i.e. the run-to-run noise level.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

MODES = {"three_state": "Three-state (C/S/U, U locked back)",
         "three_state_srcocc": "Three-state, source-occupied U locked back",
         "no_lockback": "(A) two-state C/stable, no lock-back",
         "lockback_nonconflict": "(B) two-state C/source, all non-C locked back"}
COLS = [("lpips", "LPIPS↓", 3), ("dino_i", "DINO-I↑", 3), ("edit_lpips", "Edit LPIPS↓", 3),
        ("edit_psnr", "Edit PSNR↑", 2), ("keep_psnr", "Keep PSNR↑", 2), ("keep_lpips", "Keep LPIPS↓", 3)]
LOWER_BETTER = ("lpips", "edit_lpips", "keep_lpips")


def view_mean(rows, mode, held_out=None):
    rr = [r for r in rows if r["mode"] == mode and (held_out is None or r["held_out"] == held_out)]
    return {k: float(np.nanmean([r[k] for r in rr])) for k, _, _ in COLS} if rr else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True, help="Comma-separated item dirs.")
    ap.add_argument("--suffixes", default="", help="Comma-separated seed-replicate suffixes ('' = seed 1, _s2, ...).")
    ap.add_argument("--output", type=Path, required=True, help="Output prefix (writes .md / .csv / .json).")
    args = ap.parse_args()
    suffixes = args.suffixes.split(",")

    runs, csv_rows = {}, []          # runs[item][suffix][mode] = entry
    for item in [Path(p) for p in args.items.split(",")]:
        for sfx in suffixes:
            r = json.loads((item / f"_state_ablation_eval{sfx}" / "metrics.json").read_text())
            vx = r["voxels"]
            for m in MODES:
                ch = vx["changed_voxels"][m]
                entry = {"all": view_mean(r["rows"], m), "inputs": view_mean(r["rows"], m, False),
                         "held_out": view_mean(r["rows"], m, True),
                         "U_preserved": 1.0 - ch["U"] / max(vx["region_cells"]["U"], 1),
                         "S_edited": ch["S"] / max(vx["region_cells"]["S"], 1),
                         "changed_voxels_total": int(sum(ch[k] for k in ("C", "S", "U", "N")))}
                runs.setdefault(item.name, {}).setdefault(sfx, {})[m] = entry
                csv_rows.append({"item": item.name, "seed_suffix": sfx or "_s1", "mode": m,
                                 **{f"all_{k}": v for k, v in entry["all"].items()},
                                 "U_preserved": entry["U_preserved"], "S_edited": entry["S_edited"],
                                 "changed_voxels_total": entry["changed_voxels_total"],
                                 **{f"{s}_cells": vx["region_cells"][s] for s in ("C", "S", "U", "N")}})

    def avg(item, m, key):
        return float(np.mean([runs[item][s][m]["all"][key] for s in suffixes]))

    items = list(runs)
    lines = [f"Seeds per case: {len(suffixes)}", "",
             "| Case | Variant | " + " | ".join(c[1] for c in COLS) + " | U kept | S edited |",
             "|---" * (len(COLS) + 4) + "|"]
    for name in items:
        for m in MODES:
            vals = " | ".join(f"{avg(name, m, k):.{p}f}" for k, _, p in COLS)
            u = np.mean([runs[name][s][m]["U_preserved"] for s in suffixes])
            se = np.mean([runs[name][s][m]["S_edited"] for s in suffixes])
            lines.append(f"| {name} | {m} | {vals} | {u:.0%} | {se:.0%} |")
    mean = {m: {k: float(np.mean([avg(i, m, k) for i in items])) for k, _, _ in COLS} for m in MODES}
    for m in MODES:
        vals = " | ".join(f"{mean[m][k]:.{p}f}" for k, _, p in COLS)
        lines.append(f"| **mean ({len(items)})** | {m} | {vals} | | |")

    lines += ["", "| Three-state variant | vs two-state | " + " | ".join(c[1] for c in COLS) + " | wins (LPIPS) |",
              "|---" * (len(COLS) + 3) + "|"]
    n_pairs = len(items) * len(suffixes)
    for base in ("three_state", "three_state_srcocc"):
        for m in ("no_lockback", "lockback_nonconflict"):
            diffs = [f"{(-1 if k in LOWER_BETTER else 1) * (mean[base][k] - mean[m][k]):+.{p}f}" for k, _, p in COLS]
            wins = sum(runs[i][s][base]["all"]["lpips"] < runs[i][s][m]["all"]["lpips"] for i in items for s in suffixes)
            lines.append(f"| {base} | {MODES[m]} | " + " | ".join(diffs) + f" | {wins}/{n_pairs} |")
    if len(suffixes) > 1:
        spread = {m: float(np.mean([abs(runs[i][suffixes[0]][m]["all"]["lpips"] - runs[i][suffixes[1]][m]["all"]["lpips"])
                                    for i in items])) for m in MODES}
        lines += ["", "Seed spread (mean |LPIPS seed1 - seed2| per case): "
                  + ", ".join(f"{m} {v:.4f}" for m, v in spread.items())]
    md = "\n".join(lines)
    args.output.with_suffix(".md").write_text(md + "\n")
    args.output.with_suffix(".json").write_text(json.dumps({"runs": runs, "mean": mean, "suffixes": suffixes}, indent=2))
    with args.output.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    print(md)


if __name__ == "__main__":
    main()
