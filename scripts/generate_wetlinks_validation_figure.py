#!/usr/bin/env python3
"""Generate the journal figure for independent WetLinks validation."""

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
    OPAQUE_GRID_COLOR,
    configure_ieee_figure_style,
    save_png_pdf_pair,
)


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", default="results/wetlinks_longitudinal_validation"
    )
    parser.add_argument(
        "--output", default="results/figures/wetlinks_longitudinal_validation"
    )
    args = parser.parse_args()

    results_dir = _resolve(args.results_dir)
    metrics = pd.read_csv(results_dir / "late_holdout_model_metrics.csv")
    transfer = pd.read_csv(results_dir / "cross_site_transfer_summary.csv")
    risk = pd.read_csv(results_dir / "risk_diagnostics.csv").iloc[0]

    labels = {
        "persistence": "Persistence",
        "temporal_ridge": "Temporal",
        "context_ridge": "Context",
        "calibrated_fusion": "Fusion",
    }
    metrics = metrics[metrics["model"].isin(labels)].copy()
    metrics["label"] = metrics["model"].map(labels)
    colors = ["#7a7f87", "#0b6e75", "#d47b29", "#386cb0"]

    configure_ieee_figure_style(base_font_size=9.0)
    fig, axes = plt.subplots(1, 3, figsize=(IEEE_TEXT_WIDTH_IN, 3.05))

    x = np.arange(len(metrics))
    metric_bars = axes[0].bar(
        x,
        metrics["mae_ms"],
        color=colors,
        width=0.72,
        edgecolor="#303030",
        linewidth=0.45,
    )
    for patch, hatch in zip(metric_bars.patches, ["", "//", "xx", "\\\\"]):
        patch.set_hatch(hatch)
    axes[0].set_xticks(x, metrics["label"], rotation=20, ha="right")
    axes[0].set_ylabel("MAE (ms)")
    axes[0].set_title("(a) Late-period forecast error")
    axes[0].grid(axis="y", color=OPAQUE_GRID_COLOR)
    for index, value in enumerate(metrics["mae_ms"]):
        axes[0].text(index, value + 0.07, f"{value:.2f}", ha="center", fontsize=9)

    site_labels = (
        transfer["held_out_target_site"].str.replace("wetlinks/", "", regex=False)
    )
    tx = np.arange(len(transfer))
    width = 0.35
    axes[1].bar(
        tx - width / 2,
        transfer["persistence_mae_ms"],
        width,
        label="Persistence",
        color="#7a7f87",
        edgecolor="#303030",
        linewidth=0.45,
    )
    axes[1].bar(
        tx + width / 2,
        transfer["transferred_temporal_mae_ms"],
        width,
        label="Transferred temporal",
        color="#0b6e75",
        edgecolor="#303030",
        linewidth=0.45,
        hatch="//",
    )
    axes[1].set_xticks(tx, site_labels)
    axes[1].set_ylabel("MAE (ms)")
    axes[1].set_title("(b) Unseen-site transfer")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    axes[1].grid(axis="y", color=OPAQUE_GRID_COLOR)

    risk_values = [
        abs(risk["spearman_disagreement_vs_absolute_error"]),
        risk["high_error_detection_auroc"],
        risk["empirical_upper_coverage"],
    ]
    risk_labels = [r"$|\rho(D,|e|)|$", "High-error\nAUROC", "Upper\ncoverage"]
    rx = np.arange(3)
    risk_bars = axes[2].bar(
        rx,
        risk_values,
        color=["#d47b29", "#d47b29", "#386cb0"],
        edgecolor="#303030",
        linewidth=0.45,
    )
    for patch, hatch in zip(risk_bars.patches, ["//", "xx", "\\\\"]):
        patch.set_hatch(hatch)
    axes[2].axhline(
        0.5,
        color="#6b6b6b",
        linestyle="--",
        linewidth=1,
    )
    axes[2].axhline(
        0.9,
        color="#0b6e75",
        linestyle=":",
        linewidth=1.2,
    )
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xticks(rx, risk_labels)
    axes[2].set_title("(c) Risk-signal audit")
    axes[2].text(
        0.03,
        0.5,
        "Chance AUROC",
        transform=axes[2].get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=8.25,
        color="#4d4d4d",
        bbox={"facecolor": "white", "edgecolor": "white", "pad": 0.5},
    )
    axes[2].text(
        0.03,
        0.9,
        "Nominal coverage",
        transform=axes[2].get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=8.25,
        color="#0b6e75",
        bbox={"facecolor": "white", "edgecolor": "white", "pad": 0.5},
    )
    axes[2].grid(axis="y", color=OPAQUE_GRID_COLOR)
    for index, value in enumerate(risk_values):
        axes[2].text(index, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)

    fig.suptitle(
        "Independent WetLinks Longitudinal Validation (5-minute horizon)",
        fontsize=10,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.79,
        bottom=0.22,
        wspace=0.40,
    )
    output = _resolve(args.output)
    png_path, pdf_path = save_png_pdf_pair(fig, output)
    plt.close(fig)
    print(f"figures_written={png_path},{pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
