#!/usr/bin/env python3
"""Grouped bar chart of total fleet waiting-hours vs charging-hours (slow +
fast combined) in each selected schedule, across the paper's four
canonical scenarios (same selection/order as
plot_selected_schedule_heatmap.py / plot_charger_stack_vs_uncoupled.py:
(a) no-limit benchmark, (b) one-hour individual-wait, (c) priority rule,
(d) fleet-level wait <= 0.9 W_0) -- companion summary to
selected_schedule_heatmap.png, since the per-state differences between
cases are real but too sparse to read off the fleet-wide spaghetti
heatmap by eye.

Writes charging_vs_waiting_bars.{png,pdf} into this directory.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_selected_schedule_heatmap as h

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))

WAIT_COLOR = "#2a78d6"
CHARGE_COLOR = "#eb6834"
CASE_ORDER = h.PANEL_ORDER


def main():
    lam_by_case = {
        c: pd.read_csv(os.path.join(h.CASES[c]["case_dir"], "windowed_integer_lambda.csv"))
        for c in CASE_ORDER
    }
    truck_order = sorted(set().union(*(set(lam["original_id"].astype(int)) for lam in lam_by_case.values())))

    waiting_hours = []
    charging_hours = []
    for c in CASE_ORDER:
        mat = h.build_case_matrix(h.CASES[c], truck_order)
        n_wait = int((mat == h.STATE_CODE["waiting"]).sum())
        n_slow = int((mat == h.STATE_CODE["slow charging"]).sum())
        n_fast = int((mat == h.STATE_CODE["fast charging"]).sum())
        waiting_hours.append(n_wait * 0.5)
        charging_hours.append((n_slow + n_fast) * 0.5)
        print(f"case {c}: waiting={n_wait * 0.5:.0f}h  charging={((n_slow + n_fast) * 0.5):.0f}h "
              f"(slow={n_slow * 0.5:.0f}h, fast={n_fast * 0.5:.0f}h)")

    labels = [f"{h.PANEL_TAGS[c]} {h.CASES[c]['label']}" for c in CASE_ORDER]
    x = np.arange(len(CASE_ORDER))
    width = 0.36

    plt.rcParams.update({
        "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16,
        "xtick.labelsize": 11, "ytick.labelsize": 13, "legend.fontsize": 14,
    })
    # Waiting dominates charging by 10-70x in raw fleet-hours, so a single
    # shared linear scale (honest about the true magnitude gap) would flatten
    # the charging bars to invisibility -- panel (b) re-scales charging alone
    # so the a/b/c/d/e pattern the wait-cap/fleet-budget change actually
    # produces is legible, per dataviz guidance against mismatched-scale
    # single-axis bars.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 7.5), constrained_layout=True)

    bars_wait = ax1.bar(x - width / 2, waiting_hours, width, color=WAIT_COLOR, label="waiting")
    bars_charge = ax1.bar(x + width / 2, charging_hours, width, color=CHARGE_COLOR, label="charging (slow + fast)")
    for bars in (bars_wait, bars_charge):
        for b in bars:
            ax1.annotate(f"{b.get_height():,.0f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                         xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("Total fleet-hours over 39 days")
    ax1.set_title("(a) waiting vs. charging, true scale", loc="left", fontweight="bold")
    ax1.legend(loc="upper right", frameon=False)
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    bars_charge2 = ax2.bar(x, charging_hours, width * 1.4, color=CHARGE_COLOR)
    for b in bars_charge2:
        ax2.annotate(f"{b.get_height():,.0f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("Total charging fleet-hours over 39 days")
    ax2.set_title("(b) charging hours, rescaled", loc="left", fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Selected-schedule total waiting vs. charging hours, paper's four canonical scenarios",
                 fontsize=17, y=1.06)

    out_base = os.path.join(EXPERIMENT_DIR, "charging_vs_waiting_bars")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=150, bbox_inches="tight")
    print("wrote", out_base + ".png / .pdf")


if __name__ == "__main__":
    main()
