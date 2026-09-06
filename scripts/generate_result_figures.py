#!/usr/bin/env python3
"""Generate paper-ready figures from the current result files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd

SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.config import load_config
from open_leo_latency_routing.data.loaders import ensure_parent
from open_leo_latency_routing.visualization import (
    IEEE_TEXT_WIDTH_IN,
    configure_ieee_figure_style,
    save_png_pdf_pair,
)

plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.title_fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _save_plot(path: Path) -> None:
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def _write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    title: str,
    body: str,
    color: str,
    *,
    title_fontsize: float = 9.2,
    body_fontsize: float = 7.8,
) -> None:
    box = FancyBboxPatch(
        xy,
        size[0],
        size[1],
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor=color,
        facecolor="#ffffff",
    )
    ax.add_patch(box)
    x, y = xy
    w, h = size
    ax.text(
        x + w / 2.0,
        y + h * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=title_fontsize,
        weight="bold",
    )
    ax.text(
        x + w / 2.0,
        y + h * 0.33,
        body,
        ha="center",
        va="center",
        fontsize=body_fontsize,
        wrap=True,
    )


def _generate_system_overview_figure(config: dict[str, object], out_dir: Path) -> None:
    """Create a paper-facing overview of the forecasting-to-decision pipeline."""

    dataset_cfg = config["dataset"]
    forecast_cfg = config["forecasting"]
    graph_cfg = config["graph"]
    opt_cfg = config["optimization"]
    stress_cfg = config.get("stress", {})

    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    boxes = [
        ((0.02, 0.55), (0.17, 0.26), "Evidence sources", "COMMECT + LENS/WetLinks\n+ orbital simulation", "#2b6cb0"),
        ((0.23, 0.55), (0.17, 0.26), "Temporal expert", "Ridge on path history\nno graph features", "#2f855a"),
        ((0.44, 0.55), (0.17, 0.26), "Context expert", "matched Ridge capacity\ncontext-only features", "#dd6b20"),
        ((0.65, 0.55), (0.15, 0.26), "Risk calibration", "paired residual covariance\nand bounded risk inputs", "#805ad5"),
        ((0.84, 0.55), (0.14, 0.26), "Evidence gate", "group evidence passes\nor reactive abstention", "#c53030"),
        ((0.23, 0.13), (0.57, 0.20), "Evaluation", "chronological/site holdout, measured transfer, and explicitly injected degradation stress", "#718096"),
    ]
    for xy, size, title, body, color in boxes:
        _draw_box(ax, xy, size, title, body, color)

    arrow_style = dict(arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color="#4a5568")
    for start, end in [
        ((0.19, 0.68), (0.23, 0.68)),
        ((0.40, 0.68), (0.44, 0.68)),
        ((0.61, 0.68), (0.65, 0.68)),
        ((0.80, 0.68), (0.84, 0.68)),
        ((0.72, 0.55), (0.72, 0.33)),
    ]:
        ax.add_patch(FancyArrowPatch(start, end, **arrow_style))

    caption = (
        f"Latency budget: {opt_cfg['latency_budget_ms']} ms | "
        f"matched expert family: Ridge | "
        f"fusion/risk quantities: fitted before selection | "
        f"forecast: {forecast_cfg['horizon_seconds']} s | "
        f"stress fractions: burst={stress_cfg.get('burst_fraction', 'n/a')}, "
        f"outage={stress_cfg.get('outage_session_fraction', 'n/a')}, "
        f"structural={stress_cfg.get('structural_location_fraction', 'n/a')}"
    )
    ax.text(0.5, 0.93, "Opportunity-aware evidence gating under measured and injected shifts", ha="center", va="center", fontsize=13, weight="bold")
    ax.text(0.5, 0.07, caption, ha="center", va="center", fontsize=8.0)
    ensure_parent(out_dir / "system_overview.png")
    _save_plot(out_dir / "system_overview.png")


def _generate_leo_system_model_figure(config: dict[str, object], out_dir: Path) -> None:
    """Draw the physical access-path and controller timing model."""

    configure_ieee_figure_style(base_font_size=9.0)
    fig, ax = plt.subplots(figsize=(IEEE_TEXT_WIDTH_IN, 3.35))
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    _draw_box(
        ax,
        (0.02, 0.39),
        (0.17, 0.22),
        "Ground client",
        "state samples\nand service traffic",
        "#2b6cb0",
        title_fontsize=9.5,
        body_fontsize=9,
    )
    _draw_box(
        ax,
        (0.81, 0.39),
        (0.17, 0.22),
        "Service endpoint",
        "gateway / PoP /\nremote application",
        "#2f855a",
        title_fontsize=9.5,
        body_fontsize=9,
    )
    _draw_box(
        ax,
        (0.33, 0.76),
        (0.34, 0.16),
        "Access-path controller",
        "infer, validate or abstain, and install",
        "#805ad5",
        title_fontsize=9.5,
        body_fontsize=9,
    )

    satellite_positions = [(0.31, 0.47), (0.50, 0.56), (0.69, 0.47)]
    for index, (x_pos, y_pos) in enumerate(satellite_positions, start=1):
        circle = plt.Circle(
            (x_pos, y_pos),
            0.055,
            facecolor="#fff7ed",
            edgecolor="#dd6b20",
            linewidth=1.5,
        )
        ax.add_patch(circle)
        ax.text(
            x_pos,
            y_pos,
            f"LEO {index}",
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            bbox={"facecolor": "#fff7ed", "edgecolor": "#fff7ed", "pad": 0.3},
        )

    route_colors = ["#2b6cb0", "#dd6b20", "#2f855a"]
    for (x_pos, y_pos), color in zip(satellite_positions, route_colors):
        ax.plot([0.19, x_pos - 0.05], [0.50, y_pos], color=color, linewidth=1.8)
        ax.plot([x_pos + 0.05, 0.81], [y_pos, 0.50], color=color, linewidth=1.8)

    # COMMECT validates transfer to heterogeneous concurrent access, so the
    # system view makes the terrestrial alternative explicit rather than
    # visually implying that every validation path is satellite-only.
    ax.plot(
        [0.19, 0.81],
        [0.36, 0.36],
        color="#718096",
        linewidth=1.6,
        linestyle="--",
    )
    ax.text(
        0.50,
        0.325,
        "Concurrent terrestrial alternative (COMMECT shadow replay)",
        ha="center",
        va="center",
        fontsize=9,
        color="#4a5568",
    )

    arrow_style = dict(arrowstyle="-|>", mutation_scale=13, linewidth=1.2, color="#4a5568")
    ax.add_patch(FancyArrowPatch((0.105, 0.61), (0.39, 0.76), **arrow_style))
    ax.add_patch(FancyArrowPatch((0.61, 0.76), (0.895, 0.61), **arrow_style))
    ax.text(0.22, 0.705, "Collection", fontsize=9, rotation=20, ha="center")
    ax.text(0.78, 0.70, "Disseminate / install", fontsize=9, rotation=-20, ha="center")

    ax.text(
        0.50,
        0.185,
        "Control time = collection + inference + dissemination + installation\n"
        "Primary replay endpoint: selected-path next-bin network RTT",
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7fafc", edgecolor="#718096"),
    )
    ax.text(
        0.50,
        0.055,
        "Network RTT and controller timing are reported separately.",
        ha="center",
        va="center",
        fontsize=9,
    )
    ax.text(
        0.50,
        0.975,
        "Concurrent LEO and heterogeneous access-path selection",
        ha="center",
        va="center",
        fontsize=10.5,
        weight="bold",
    )
    ensure_parent(out_dir / "leo_system_model.png")
    save_png_pdf_pair(fig, out_dir / "leo_system_model.png", dpi=600)
    plt.close(fig)


def _write_notation_table(config: dict[str, object], out_dir: Path) -> None:
    """Write a short notation table that can be pasted into the manuscript."""

    horizon = config["forecasting"]["horizon_seconds"]
    snapshot = config["graph"]["snapshot_seconds"]
    opt_cfg = config["optimization"]
    content = """
# Mathematical Notation Table

| Symbol | Meaning |
| --- | --- |
| $x_{p,t}$ | observable feature vector for path $p$ at decision time $t$ |
| $\\hat y^T_{p,t+1}$ | temporal-history expert prediction |
| $\\hat y^C_{p,t+1}$ | context-feature expert prediction |
| $\\sigma_T^2,\\sigma_C^2$ | validation residual variances of the matched-estimator-family experts |
| $s_{TC}$ | paired temporal--context residual covariance on the shared calibration block |
| $\\tilde\\Delta_{p,t}$ | residual-scaled disagreement, $|\\hat y^T-\\hat y^C|/\\sqrt{\\sigma_T^2+\\sigma_C^2-2s_{TC}}$ |
| $\\mu_{p,t}$ | deterministic covariance-aware linear-pool prediction |
| $V_{p,t}$ | estimated error variance of the deterministic linear pool |
| $\\hat e_{p,t}$ | validation-fitted expected residual risk |
| $j^*$ | pre-test evidence-gated fallback in $\\{R,C,E\\}$ |
| $B$ | evaluated network-RTT QoS threshold (ms) |
| $L_{\\mathrm{ctrl}}$ | separately reported collection, inference, dissemination, and installation latency |
| $H$ | forecast horizon, __HORIZON__ s by default |
| $\\Delta_t$ | evaluated decision window, __SNAPSHOT__ s in the base table |
| $\\mathcal{P}_t$ | candidate service paths at time $t$ |

The calibrated mixture is

$$
\\mu_{p,t}=w_T\\hat y^T_{p,t+1}+w_C\\hat y^C_{p,t+1},
\\qquad w_C=1-w_T,
\\qquad
w_T=\\Pi_{[0,1]}\\!\\left(
\\frac{\\sigma_C^2-s_{TC}}{\\sigma_T^2+\\sigma_C^2-2s_{TC}}
\\right),
$$

$$
V_{p,t}=w_T^2\\sigma_T^2+w_C^2\\sigma_C^2+2w_Tw_Cs_{TC}.
$$

Manual score weights are replaced by non-negative regression on validation
residuals:

$$
\\hat e_{p,t}=b+\\theta_\\Delta\\tilde\\Delta_{p,t}
+\\theta_E\\sigma^{\\mathrm{ens}}_{p,t}+\\theta_R R_{p,t},
\\qquad \\theta_\\Delta,\\theta_E,\\theta_R\\ge0.
$$

The residual-risk score is a diagnostic ablation. The deployed candidate
$j^*$ is frozen on a separate policy-selection interval and defaults to
reactive unless independent collection groups satisfy the opportunity,
aggregate-success non-inferiority, post-hoc opportunity-conditioned success
non-inferiority, and bounded-CVaR evidence checks. Disagreement is not a
guaranteed outage certificate.

Injected degradation is applied only to evaluation data and is reported
separately from measured temporal and site shifts.
"""
    content = dedent(content).replace("__HORIZON__", str(horizon)).replace("__SNAPSHOT__", str(snapshot))
    _write_text(out_dir / "notation_table.md", content)


def _write_reviewer_notes(config: dict[str, object], out_dir: Path) -> None:
    """Summarize the reviewer-facing design choices in one compact note."""

    content = f"""
# Reviewer-Facing Notes

## What the method is

This repo implements a lightweight, evidence-gated operational decision rule. It is not a new prediction family. Matched-capacity experts provide separate temporal-history and peer-context views. A disjoint policy-selection interval may admit a frozen learned fallback only after enough opportunity-bearing independent groups, simultaneous aggregate and post-hoc opportunity-conditioned success non-inferiority, and bounded-CVaR checks; otherwise the rule abstains to reactive selection.

## Why the disagreement score is used

Disagreement is one input to a non-negative residual-risk ablation fitted on calibration data. The code reports correlation and shared-failure diagnostics. It does not claim that disagreement is a mathematically guaranteed outage detector.

## Why the ensemble uncertainty selector can win under degradation

The ensemble selector aggregates multiple bootstrapped temporal models, so it reacts to both mean error and forecast spread. It is a comparator, not an assumed winner; the evidence gate can reject it when success or tail evidence is insufficient.

## Generalization statement

The pipeline includes a canonical trace adapter and an independent CC BY 4.0 Starlink IRTT validation dataset (DOI 10.17632/479v4mym7j.2). That dataset validates predictor and risk behavior but cannot support alternative-path decisions because it contains repeated single-endpoint experiments.

## Structural-shift statement

The mild/moderate/severe shifts are injected only into evaluation data. The manuscript should state that explicitly in the title or abstract to avoid overclaiming real outage-trace availability.

## Default evaluation horizon

Default forecasting horizon: {config['forecasting']['horizon_seconds']} s. Default graph snapshot interval: {config['graph']['snapshot_seconds']} s.
"""
    _write_text(out_dir / "reviewer_notes.md", dedent(content))


def _ordered_scenarios(frame: pd.DataFrame, column: str = "scenario_name") -> list[str]:
    preferred = ["base", "burst", "outage", "structural"]
    present = frame[column].dropna().unique().tolist()
    ordered = [name for name in preferred if name in present]
    ordered.extend([name for name in present if name not in ordered])
    return ordered


def _plot_base_forecasting(forecast_metrics: pd.DataFrame, graph_metrics: pd.DataFrame, out_path: Path) -> None:
    base_forecast = forecast_metrics.copy()
    graph = graph_metrics.copy()
    combined = pd.concat(
        [
            base_forecast[["model_name", "mae"]].assign(group="Temporal"),
            graph[["model_name", "mae"]].assign(group="Graph-aware"),
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2b6cb0" if group == "Temporal" else "#dd6b20" for group in combined["group"]]
    ax.bar(combined["model_name"], combined["mae"], color=colors)
    ax.set_title("Forecasting Error on the Base Evaluation Split", pad=14)
    ax.set_ylabel("MAE (ms)")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for index, value in enumerate(combined["mae"]):
        ax.text(index, value + 0.08, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    handles = [
        plt.Line2D([0], [0], color="#2b6cb0", lw=8, label="Temporal"),
        plt.Line2D([0], [0], color="#dd6b20", lw=8, label="Graph-aware"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True)
    _save_plot(out_path)


def _plot_policy_comparison(policy_summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    base = policy_summary.copy()
    axes[0].bar(base["policy_name"], base["mean_realized_latency_ms"], color="#2f855a")
    axes[0].set_title("Decision Quality on Base Data", pad=14)
    axes[0].set_ylabel("Mean Realized Latency (ms)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1].bar(base["policy_name"], base["mean_decision_gap_ms"], color="#805ad5")
    axes[1].set_title("Decision Gap on Base Data", pad=14)
    axes[1].set_ylabel("Mean Decision Gap (ms)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    _save_plot(out_path)


def _plot_ablation_with_ci(ablation_summary: pd.DataFrame, out_path: Path) -> None:
    ordered_names = [
        "predictive_temporal_only",
        "predictive_graph_only",
        "predictive_graph_greedy",
    ]
    labels = ["Temporal-only", "Graph-only", "Graph-aware"]
    base_summary = ablation_summary[ablation_summary["scenario_name"] == "base"].copy()
    frame = base_summary.set_index("policy_name").loc[ordered_names].reset_index()
    means = frame["mean_realized_latency_ms"].to_numpy()
    lower = means - frame["realized_next_latency_ms_ci_lower"].to_numpy()
    upper = frame["realized_next_latency_ms_ci_upper"].to_numpy() - means

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(labels, means, color=["#718096", "#2b6cb0", "#dd6b20", "#2f855a"])
    ax.errorbar(labels, means, yerr=[lower, upper], fmt="none", ecolor="black", elinewidth=1.5, capsize=4)
    ax.set_title("Ablation on Base Decision Windows (95% Bootstrap CI)", pad=14)
    ax.set_ylabel("Mean Realized Latency (ms)")
    ax.set_xlabel("Policy Variant")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for index, value in enumerate(means):
        ax.text(index, value + 0.12, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    _save_plot(out_path)


def _plot_stress_forecasting(stress_forecast: pd.DataFrame, out_path: Path) -> None:
    pivot = stress_forecast.pivot(index="scenario_name", columns="model_name", values="mae")
    ordered = pivot.loc[_ordered_scenarios(stress_forecast)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered.plot(kind="bar", ax=ax, color=["#4a5568", "#2b6cb0", "#38a169"], width=0.78)
    ax.set_title("Forecasting Robustness Under Burst and Outage Shifts", pad=14)
    ax.set_ylabel("MAE (ms)")
    ax.set_xlabel("Scenario")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="Model", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)
    _save_plot(out_path)


def _plot_stress_policy(policy_stress: pd.DataFrame, out_path: Path) -> None:
    policy_label_map = {
        "random": "Random",
        "reactive_greedy": "Reactive",
        "predictive_greedy": "Temporal",
        "predictive_graph_greedy": "Graph",
        "predictive_simple_fusion_greedy": "Fusion",
        "predictive_consensus_greedy": "Consensus",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.0), sharex=True)
    latency = policy_stress.pivot(index="scenario_name", columns="policy_name", values="mean_realized_latency_ms")
    decision_gap = policy_stress.pivot(index="scenario_name", columns="policy_name", values="mean_decision_gap_ms")
    ordered = _ordered_scenarios(policy_stress)
    column_order = [name for name in policy_label_map if name in latency.columns]
    latency = latency.loc[ordered, column_order].rename(columns=policy_label_map)
    decision_gap = decision_gap.loc[ordered, column_order].rename(columns=policy_label_map)
    display_index = [name.title() for name in ordered]
    latency.index = display_index
    decision_gap.index = display_index

    latency.plot(kind="line", marker="o", ax=axes[0], legend=False)
    axes[0].set_title("Latency Across Stress Scenarios", pad=6)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_xlabel("")
    axes[0].grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    axes[0].set_ylim(39, 51)

    decision_gap.plot(kind="line", marker="o", ax=axes[1])
    axes[1].set_title("Decision Gap Across Stress Scenarios", pad=6)
    axes[1].set_ylabel("Decision Gap (ms)")
    axes[1].set_xlabel("")
    axes[1].grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    axes[1].legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        frameon=True,
        title="Policy",
        columnspacing=0.8,
        handlelength=1.4,
    )
    _save_plot(out_path)


def _plot_disagreement_uncertainty(disagreement_summary: pd.DataFrame, out_path: Path) -> None:
    plot_frame = disagreement_summary[
        disagreement_summary["policy_name"].isin(
            [
                "predictive_greedy",
                "predictive_graph_greedy",
                "predictive_simple_fusion_greedy",
                "predictive_consensus_greedy",
            ]
        )
    ].copy()
    plot_frame = plot_frame[plot_frame["scenario_name"].isin(["base", "outage", "structural"])]
    if plot_frame.empty:
        return

    label_map = {
        "predictive_greedy": "Temporal",
        "predictive_graph_greedy": "Graph-aware",
        "predictive_simple_fusion_greedy": "Fusion",
        "predictive_consensus_greedy": "Consensus",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.0))
    for scenario_name, group in plot_frame.groupby("scenario_name", sort=False):
        consensus = group[group["policy_name"] == "predictive_consensus_greedy"]
        axes[0].plot(
            consensus["disagreement_bin"],
            consensus["mean_decision_gap_ms"],
            marker="o",
            label=scenario_name.title(),
        )
    axes[0].set_title("Consensus Decision Gap by Disagreement", pad=6)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Decision Gap (ms)")
    axes[0].grid(True, linestyle="--", alpha=0.35, linewidth=0.6)

    structural = plot_frame[plot_frame["scenario_name"] == "structural"].copy()
    if not structural.empty:
        pivot = structural.pivot(index="disagreement_bin", columns="policy_name", values="mean_realized_latency_ms")
        pivot = pivot.rename(columns=label_map)
        pivot = pivot[[col for col in ["Temporal", "Graph-aware", "Fusion", "Consensus"] if col in pivot.columns]]
        pivot.plot(kind="bar", ax=axes[1], width=0.78)
    axes[1].set_title("Structural Latency by Disagreement", pad=6)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.6)
    axes[1].legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.07),
        ncol=2,
        frameon=True,
        title="Policy",
        columnspacing=0.8,
        handlelength=1.2,
    )
    axes[0].legend(loc="upper left", frameon=True, title="Scenario")
    _save_plot(out_path)


def _plot_penalty_sweep(penalty_sweep: pd.DataFrame, out_path: Path) -> None:
    plot_frame = penalty_sweep[penalty_sweep["scenario_name"].isin(["base", "outage", "structural"])].copy()
    if plot_frame.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for scenario_name, group in plot_frame.groupby("scenario_name", sort=False):
        axes[0].plot(
            group["disagreement_penalty"],
            group["mean_realized_latency_ms"],
            marker="o",
            linewidth=2.2,
            label=scenario_name,
        )
        axes[1].plot(
            group["disagreement_penalty"],
            group["mean_decision_gap_ms"],
            marker="o",
            linewidth=2.2,
            label=scenario_name,
        )
    axes[0].set_title("Consensus Penalty Sweep: Latency", pad=14)
    axes[0].set_xlabel("Disagreement penalty $\\lambda$")
    axes[0].set_ylabel("Mean realized latency (ms)")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[1].set_title("Consensus Penalty Sweep: Decision Gap", pad=14)
    axes[1].set_xlabel("Disagreement penalty $\\lambda$")
    axes[1].set_ylabel("Mean decision gap (ms)")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, title="Scenario")
    _save_plot(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-metrics", default="results/temporal_forecasting/temporal_forecast_metrics.csv")
    parser.add_argument("--graph-metrics", default="results/graph_forecasting/graph_forecast_metrics.csv")
    parser.add_argument("--policy-summary", default="results/decision_policy_evaluation/decision_policy_summary.csv")
    parser.add_argument("--ablation-summary", default="results/ablation_evaluation/ablation_policy_summary.csv")
    parser.add_argument("--stress-forecast-metrics", default="results/robustness_evaluation/temporal_forecast_robustness_metrics.csv")
    parser.add_argument("--stress-policy-summary", default="results/robustness_evaluation/decision_policy_robustness_summary.csv")
    parser.add_argument("--disagreement-summary", default="results/robustness_evaluation/disagreement_uncertainty_summary.csv")
    parser.add_argument("--penalty-sweep", default="results/robustness_evaluation/consensus_penalty_sweep.csv")
    parser.add_argument("--plots-dir", default="results/figures")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--manuscript-assets-only",
        action="store_true",
        help=(
            "Regenerate the system, controller, notation, and reviewer-note "
            "assets without requiring the legacy result tables."
        ),
    )
    args = parser.parse_args()

    config = load_config(_resolve_repo_path(args.config))
    plots_dir = ensure_parent(_resolve_repo_path(f"{args.plots_dir}/.keep")).parent
    manuscript_dir = ensure_parent(plots_dir / "manuscript_assets/.keep").parent
    if args.manuscript_assets_only:
        _generate_system_overview_figure(config, manuscript_dir)
        _generate_leo_system_model_figure(config, manuscript_dir)
        _write_notation_table(config, manuscript_dir)
        _write_reviewer_notes(config, manuscript_dir)
        print(f"manuscript_assets_written={manuscript_dir}")
        for file_path in sorted(manuscript_dir.glob("*")):
            print(file_path)
        return 0

    forecast_metrics = pd.read_csv(_resolve_repo_path(args.forecast_metrics))
    graph_metrics = pd.read_csv(_resolve_repo_path(args.graph_metrics))
    policy_summary = pd.read_csv(_resolve_repo_path(args.policy_summary))
    ablation_summary_path = _resolve_repo_path(args.ablation_summary)
    stress_forecast = pd.read_csv(_resolve_repo_path(args.stress_forecast_metrics))
    stress_policy = pd.read_csv(_resolve_repo_path(args.stress_policy_summary))
    disagreement_summary_path = _resolve_repo_path(args.disagreement_summary)
    penalty_sweep_path = _resolve_repo_path(args.penalty_sweep)

    _plot_base_forecasting(forecast_metrics, graph_metrics, plots_dir / "base_forecast_model_mae.png")
    _plot_policy_comparison(policy_summary, plots_dir / "base_decision_policy_comparison.png")
    if ablation_summary_path.exists():
        ablation_summary = pd.read_csv(ablation_summary_path)
        _plot_ablation_with_ci(ablation_summary, plots_dir / "ablation_base_latency_ci.png")
    _plot_stress_forecasting(stress_forecast, plots_dir / "robustness_forecast_model_mae.png")
    _plot_stress_policy(stress_policy, plots_dir / "robustness_decision_policy_comparison.png")
    if disagreement_summary_path.exists():
        disagreement_summary = pd.read_csv(disagreement_summary_path)
        _plot_disagreement_uncertainty(disagreement_summary, plots_dir / "disagreement_uncertainty_analysis.png")
    if penalty_sweep_path.exists():
        penalty_sweep = pd.read_csv(penalty_sweep_path)
        _plot_penalty_sweep(penalty_sweep, plots_dir / "consensus_penalty_sweep.png")
    _generate_system_overview_figure(config, manuscript_dir)
    _generate_leo_system_model_figure(config, manuscript_dir)
    _write_notation_table(config, manuscript_dir)
    _write_reviewer_notes(config, manuscript_dir)

    print(f"plots_written={plots_dir}")
    print(f"manuscript_assets_written={manuscript_dir}")
    for file_path in sorted(plots_dir.glob("*.png")):
        print(file_path)
    for file_path in sorted(manuscript_dir.glob("*")):
        print(file_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
