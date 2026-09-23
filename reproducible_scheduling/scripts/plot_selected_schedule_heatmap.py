#!/usr/bin/env python3
"""Heatmap of each truck's selected per-step state across the 39-day
horizon -- driving / waiting / slow charging / fast charging /
abandonment -- one panel per scenario, using the SAME four-scenario
selection and order as plot_charger_stack_vs_uncoupled.py's PANELS /
dominance_charger_stack_vs_uncoupled.png: (a) no-limit benchmark, (b)
one-hour individual-wait, (c) priority rule (no waiting-time
constraint), (d) fleet-level wait <= 0.9 W_0. Built from each case's own
windowed_integer_lambda.csv selections (same source data and
path-resolution logic as plot_gini_soc_rank.py's per-case average-SoC
readout: path_label -> (iteration, truck) -> priced-path CSV under that
case's column-generation run, where reprice_exempt_trucks.py adds the
extra "price_exempt_trucks" pricing iteration these repriced cases rely
on).

For each selected column (one per truck per 10-day window), the state at
a given half-hour step is read directly off that column's own
priced-path CSV row for that step:
  - parked == 0                       -> driving
  - parked == 1, charging_decision 0  -> waiting
  - parked == 1, charging_decision 1  -> slow charging
  - parked == 1, charging_decision 2  -> fast charging
  - parked == 1, charging_decision 3  -> abandonment
A truck-window with no resolvable priced-path CSV (the unreliable
seed-formula-only fallback for horizon-exempt trucks) is left blank
("no data") rather than guessed.

Writes selected_schedule_heatmap.{png,pdf} into this directory.
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(EXPERIMENT_DIR)
ABANDON_39DAY = os.path.join(
    REPO, "price_problem_expriments_full_fleet_faster_waiting_time_abandon_39day")
PRIORITY_RULE_39DAY = os.path.join(
    REPO, "price_problem_expriments_full_fleet_faster_advance_network_priority_rule_39_day")
LABEL_RE = re.compile(r"priced_path_iter_(\d+)_truck_(\d+)")

NUM_DAYS = 39
NUM_STEPS = 48
TOTAL_STEPS = NUM_DAYS * NUM_STEPS
WINDOW_DAYS = 10
NUM_WINDOWS = -(-NUM_DAYS // WINDOW_DAYS)
WINDOW_BOUNDS = [(w * WINDOW_DAYS, min(NUM_DAYS, (w + 1) * WINDOW_DAYS)) for w in range(NUM_WINDOWS)]

CASES = {
    "a": dict(cg_dir=os.path.join(ABANDON_39DAY, "merged_column_generation_run"),
              case_dir=os.path.join(EXPERIMENT_DIR, "windowed_integer_resolve_10day_case_a_repriced"),
              label="One-hour individual-wait"),
    "b": dict(cg_dir=os.path.join(ABANDON_39DAY, "merged_column_generation_run_fleet0p9"),
              case_dir=os.path.join(EXPERIMENT_DIR, "windowed_integer_resolve_10day_case_b_repriced"),
              label="Fleet-level wait $\\leq 0.9\\,W_0$ (individual wait $\\leq$ 1 h)"),
    "c": dict(cg_dir=os.path.join(ABANDON_39DAY, "merged_column_generation_run_no_wait"),
              case_dir=os.path.join(EXPERIMENT_DIR, "windowed_integer_resolve_10day_case_c_repriced"),
              label="No-limit benchmark"),
    "d": dict(cg_dir=os.path.join(ABANDON_39DAY, "merged_column_generation_run_no_wait_fleet0p9"),
              case_dir=os.path.join(EXPERIMENT_DIR, "windowed_integer_resolve_10day_case_d_repriced"),
              label="No wait cap + fleet-wait budget (not part of the paper's canonical set)"),
    "e": dict(cg_dir=os.path.join(PRIORITY_RULE_39DAY, "merged_column_generation_run"),
              case_dir=os.path.join(PRIORITY_RULE_39DAY, "windowed_integer_resolve_10day_tol20_repriced"),
              label="Priority rule (no waiting-time constraint)"),
}

# Same four-scenario selection and order as plot_charger_stack_vs_uncoupled.py's
# PANELS / dominance_charger_stack_vs_uncoupled.png -- (d) no-wait-cap +
# fleet-budget is deliberately excluded, since that combination was never
# actually solved/used in the paper (see that script's docstring).
PANEL_ORDER = ["c", "a", "e", "b"]
PANEL_TAGS = {"c": "(a)", "a": "(b)", "e": "(c)", "b": "(d)"}

STATE_CODE = {"driving": 0, "waiting": 1, "slow charging": 2, "fast charging": 3, "abandonment": 4}
STATE_COLORS = ["#4C72B0", "#999999", "#DD8452", "#C44E52", "#000000"]
NO_DATA = -1


def resolve_path(cg_dir, it_num, tid):
    for suffix in ("price_all_trucks", "price_exempt_trucks"):
        pf = os.path.join(cg_dir, f"iteration_{it_num:02d}_{suffix}", f"truck_{tid}",
                           f"truck_{tid}_priced_path", "price_problem_shortest_path_selected_path.csv")
        if os.path.exists(pf):
            return pf
    return None


def row_state_codes(path_file, step0, step1):
    out = {}
    df = pd.read_csv(path_file, usecols=["to_time_index", "parked", "charging_decision"])
    sub = df[(df["to_time_index"] >= step0) & (df["to_time_index"] < step1)]
    for t1, parked, decision in zip(sub["to_time_index"].astype(int),
                                     sub["parked"].astype(int),
                                     sub["charging_decision"].astype(int)):
        if parked == 0:
            out[t1] = STATE_CODE["driving"]
        elif decision == 0:
            out[t1] = STATE_CODE["waiting"]
        elif decision == 1:
            out[t1] = STATE_CODE["slow charging"]
        elif decision == 2:
            out[t1] = STATE_CODE["fast charging"]
        elif decision == 3:
            out[t1] = STATE_CODE["abandonment"]
    return out


def build_case_matrix(cfg, truck_order):
    lam = pd.read_csv(os.path.join(cfg["case_dir"], "windowed_integer_lambda.csv"))
    lam = lam[lam["lambda"] > 0.5]
    row_of = {tid: r for r, tid in enumerate(truck_order)}
    mat = np.full((len(truck_order), TOTAL_STEPS), NO_DATA, dtype=int)
    for _, row in lam.iterrows():
        orig_id = int(row["original_id"])
        if orig_id not in row_of:
            continue
        m = LABEL_RE.match(str(row["path_label"]))
        if not m:
            continue
        it_num, tid = int(m.group(1)), int(m.group(2))
        pf = resolve_path(cfg["cg_dir"], it_num, tid)
        if pf is None:
            continue
        w = int(row["window"])
        step0, step1 = WINDOW_BOUNDS[w][0] * NUM_STEPS, WINDOW_BOUNDS[w][1] * NUM_STEPS
        r = row_of[orig_id]
        for t1, code in row_state_codes(pf, step0, step1).items():
            mat[r, t1] = code
    return mat


def main():
    lam_by_case = {
        c: pd.read_csv(os.path.join(CASES[c]["case_dir"], "windowed_integer_lambda.csv"))
        for c in PANEL_ORDER
    }
    truck_order = sorted(set().union(*(set(lam["original_id"].astype(int)) for lam in lam_by_case.values())))
    print(f"{len(truck_order)} trucks total (union across the four panel cases)")

    cmap = ListedColormap(["#FFFFFF"] + STATE_COLORS)
    norm = BoundaryNorm(np.arange(NO_DATA, len(STATE_COLORS) + 1) - 0.5, cmap.N)
    days = np.arange(TOTAL_STEPS) / NUM_STEPS

    fig, axes = plt.subplots(2, 2, figsize=(22, 18), constrained_layout=True)
    for ax, c in zip(axes.flat, PANEL_ORDER):
        cfg = CASES[c]
        mat = build_case_matrix(cfg, truck_order)
        ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm,
                  extent=[0, NUM_DAYS, len(truck_order), 0], interpolation="none")
        for w in range(1, NUM_WINDOWS):
            ax.axvline(w * WINDOW_DAYS, color="white", linewidth=0.8, alpha=0.7)
        ax.set_title(f"{PANEL_TAGS[c]} {cfg['label']}", loc="left", fontweight="bold", fontsize=13)
        ax.set_xlabel("Day")
        ax.set_ylabel("Truck (sorted by original id)")

    legend_handles = [Patch(facecolor="#FFFFFF", edgecolor="gray", label="no data")] + [
        Patch(facecolor=color, label=state) for state, color in zip(STATE_CODE, STATE_COLORS)
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.04), frameon=False, fontsize=12)
    fig.suptitle("Selected per-truck schedule by state, paper's four canonical scenarios", fontsize=16, y=1.08)

    out_base = os.path.join(EXPERIMENT_DIR, "selected_schedule_heatmap")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=150, bbox_inches="tight")
    print("wrote", out_base + ".png / .pdf")


if __name__ == "__main__":
    main()
