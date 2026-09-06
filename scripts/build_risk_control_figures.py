#!/usr/bin/env python3
"""Build manuscript figures for risk control, timing, and telemetry skew."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from open_leo_latency_routing.visualization import (
    IEEE_TEXT_WIDTH_IN,
    configure_ieee_figure_style,
    save_png_pdf_pair,
)


OUTPUT = REPO_ROOT / "results" / "transactions_evidence" / "figures"


def _style() -> None:
    configure_ieee_figure_style(base_font_size=9.0)


def build_timing_figure() -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(IEEE_TEXT_WIDTH_IN, 3.85),
        constrained_layout=True,
    )

    ax = axes[0]
    blocks = [
        (0.00, 0.55, "Train\nfit experts", "#1f4e79", ""),
        (0.55, 0.70, "Calibrate\nresiduals", "#4f81bd", "//"),
        (0.70, 0.85, "Select\npolicy", "#c55a11", "xx"),
        (0.85, 1.00, "Locked test\nprimary score", "#548235", "\\\\"),
    ]
    for start, end, label, color, hatch in blocks:
        ax.barh(
            0,
            end - start,
            left=start,
            height=0.45,
            color=color,
            edgecolor="white",
            hatch=hatch,
        )
        ax.text(
            (start + end) / 2,
            0,
            label,
            ha="center",
            va="center",
            color="white",
            weight="bold",
            fontsize=9,
        )
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Global wall-clock fraction or rolling-fold interval")
    ax.set_title("Closed chronological blocks: crossing targets are removed")
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    ax.axvspan(0.01, 0.18, 0.40, 0.78, color="#d9e8f5", ec="#1f4e79")
    ax.text(
        0.095,
        0.59,
        "Current bin $B_t$\nprobe + aggregate",
        ha="center",
        va="center",
        color="#17365d",
        weight="bold",
    )
    boundary_x = 0.205
    ax.axvline(boundary_x, ymin=0.31, ymax=0.88, color="#7f1d1d", lw=1.6)
    ax.text(
        boundary_x,
        0.91,
        "$b_{t+1}$ boundary",
        ha="center",
        va="bottom",
        color="#7f1d1d",
        weight="bold",
    )
    stages = [
        ("Finalize /\ncollect", 0.29),
        ("Infer /\nrank", 0.42),
        ("Disseminate", 0.55),
        ("Install\nchoice", 0.68),
    ]
    for index, (label, x) in enumerate(stages):
        ax.text(
            x,
            0.59,
            label,
            ha="center",
            va="center",
            color="#17365d",
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "#eaf1f8",
                "edgecolor": "#1f4e79",
                "linewidth": 1.1,
            },
            zorder=4,
        )
        left = boundary_x + 0.012 if index == 0 else stages[index - 1][1] + 0.055
        ax.annotate(
            "",
            xy=(x - 0.055, 0.59),
            xytext=(left, 0.59),
            arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 1.1},
        )
    ax.annotate(
        "",
        xy=(0.785, 0.59),
        xytext=(0.735, 0.59),
        arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 1.1},
    )
    ax.axvspan(0.80, 0.99, 0.40, 0.78, color="#e3f0dc", ec="#548235")
    ax.text(
        0.895,
        0.59,
        "Next-bin target $B_{t+1}$\nwhole-bin shadow outcome",
        ha="center",
        va="center",
        color="#375623",
        weight="bold",
    )
    ax.text(
        0.50,
        0.15,
        "$L_{ctrl}$ = finalize/collect + infer + disseminate + install; "
        "the replay cannot identify which next-bin probes follow installation.",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#7f1d1d",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0.02, 1.02)
    ax.axis("off")
    ax.set_title("Operational timing and the shadow-policy outcome")

    save_png_pdf_pair(fig, OUTPUT / "risk_control_timing.png")
    plt.close(fig)


def build_evidence_figure() -> None:
    rolling = pd.read_csv(
        REPO_ROOT / "results" / "commect_validation_gated_rolling" / "rolling_gate_selection_evidence.csv"
    )
    skew = pd.read_csv(
        REPO_ROOT
        / "results"
        / "commect_rolling_timestamp_sensitivity"
        / "rolling_timestamp_skew_policy_sensitivity.csv"
    )
    threshold = pd.read_csv(
        REPO_ROOT / "results" / "commect_threshold_gate_sensitivity" / "threshold_gate_evidence.csv"
    )

    fig, axes = plt.subplots(1, 3, figsize=(IEEE_TEXT_WIDTH_IN, 3.2))

    learned = rolling[rolling["policy"].ne("reactive")].copy()
    for name, marker, linestyle, color in (
        ("graph", "o", "-", "#1f77b4"),
        ("ensemble", "s", "--", "#d95f02"),
    ):
        subset = learned[learned["policy"].eq(name)]
        display_name = "Context" if name == "graph" else "Ensemble"
        aggregate_lcb = (
            "aggregate_actionable_success_delta_lcb"
            if "aggregate_actionable_success_delta_lcb" in subset
            else "simultaneous_lcb"
        )
        axes[0].plot(
            subset["rolling_fold"],
            subset[aggregate_lcb],
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=f"{display_name}: all",
        )
        if "opportunity_conditioned_success_delta_lcb" in subset:
            axes[0].plot(
                subset["rolling_fold"],
                subset["opportunity_conditioned_success_delta_lcb"],
                marker=marker,
                markerfacecolor="white",
                linestyle=":",
                color=color,
                label=f"{display_name}: opp.",
            )
    axes[0].axhline(
        -0.02,
        color="#7f1d1d",
        linestyle=":",
    )
    axes[0].set_xlabel("Rolling fold")
    axes[0].set_ylabel("Success-delta LCB")
    axes[0].set_title("(a) Admission bounds")
    axes[0].set_xticks(range(1, 6))
    axes[0].text(
        5,
        -0.04,
        "Admission boundary",
        ha="right",
        va="top",
        color="#7f1d1d",
        fontsize=8.25,
    )
    axes[0].legend(loc="center", frameon=False, fontsize=8.25)

    policies = {
        "reactive_greedy": ("Reactive", "#4d4d4d", "o", "-"),
        "qos_shielded_operational_selector": (
            "Shield",
            "#1f77b4",
            "s",
            "--",
        ),
        "validation_gated_qos_selector": (
            "Evidence gate",
            "#d95f02",
            "^",
            "-.",
        ),
    }
    order = ["le_500ms", "le_1000ms", "le_2000ms", "le_5000ms", "full"]
    labels = ["<=0.5", "<=1", "<=2", "<=5", "Full"]
    for policy, (label, color, marker, linestyle) in policies.items():
        subset = skew[skew["policy_name"].eq(policy)].set_index("skew_case").loc[order]
        axes[1].plot(
            labels,
            subset["success_rate_under_60ms"],
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=label,
        )
    axes[1].set_xlabel("Maximum cross-path timestamp skew (s)")
    axes[1].set_ylabel("Test success at 60 ms")
    axes[1].set_title("(b) Rolling timestamp-skew rebuild")
    axes[1].legend(loc="lower left", frameon=False, fontsize=8.25)

    selected = threshold[threshold["selected"].astype(bool)].sort_values("threshold_ms")
    colors = ["#1f77b4" if name == "graph" else "#4d4d4d" for name in selected["policy"]]
    threshold_bars = axes[2].bar(
        selected["threshold_ms"].astype(str),
        selected["success_rate"],
        color=colors,
        edgecolor="#303030",
        linewidth=0.45,
    )
    for patch, policy in zip(threshold_bars.patches, selected["policy"]):
        patch.set_hatch("//" if policy == "graph" else "")
    for index, row in enumerate(selected.itertuples(index=False)):
        display_name = "Context" if row.policy == "graph" else str(row.policy).capitalize()
        axes[2].text(
            index,
            row.success_rate / 2,
            display_name,
            ha="center",
            va="center",
            rotation=90,
            color="white",
            weight="bold",
            fontsize=8.25,
        )
    axes[2].set_ylim(0, 1.08)
    axes[2].set_xticks(
        np.arange(len(selected)),
        [str(int(value)) for value in selected["threshold_ms"]],
    )
    axes[2].set_xlabel("Latency objective (ms)")
    axes[2].set_ylabel("Selection-interval success")
    axes[2].set_title("(c) Objective-specific selection")

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.90,
        bottom=0.18,
        wspace=0.48,
    )

    save_png_pdf_pair(fig, OUTPUT / "risk_control_evidence.png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="results/transactions_evidence/figures",
    )
    args = parser.parse_args()
    global OUTPUT
    output_value = Path(args.output_dir)
    OUTPUT = output_value if output_value.is_absolute() else REPO_ROOT / output_value
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _style()
    build_timing_figure()
    build_evidence_figure()
    print(f"figures_written={OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
