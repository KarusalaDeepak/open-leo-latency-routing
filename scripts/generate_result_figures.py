#!/usr/bin/env python3
"""Generate paper-ready figures from the current result files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

import matplotlib.pyplot as plt
import pandas as pd

SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_leo_latency_routing.data.loaders import ensure_parent


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _save_plot(path: Path) -> None:
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


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

    axes[1].bar(base["policy_name"], base["mean_regret_ms"], color="#805ad5")
    axes[1].set_title("Decision Regret on Base Data", pad=14)
    axes[1].set_ylabel("Mean Regret (ms)")
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
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    latency = policy_stress.pivot(index="scenario_name", columns="policy_name", values="mean_realized_latency_ms")
    regret = policy_stress.pivot(index="scenario_name", columns="policy_name", values="mean_regret_ms")
    ordered = _ordered_scenarios(policy_stress)
    latency = latency.loc[ordered]
    regret = regret.loc[ordered]

    latency.plot(kind="line", marker="o", ax=axes[0], linewidth=2.2, legend=False)
    axes[0].set_title("Policy Latency Across Stress Scenarios", pad=14)
    axes[0].set_ylabel("Mean Realized Latency (ms)")
    axes[0].set_xlabel("Scenario")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    regret.plot(kind="line", marker="o", ax=axes[1], linewidth=2.2)
    axes[1].set_title("Policy Regret Across Stress Scenarios", pad=14)
    axes[1].set_ylabel("Mean Regret (ms)")
    axes[1].set_xlabel("Scenario")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, title="Policy")
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
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for scenario_name, group in plot_frame.groupby("scenario_name", sort=False):
        consensus = group[group["policy_name"] == "predictive_consensus_greedy"]
        axes[0].plot(
            consensus["disagreement_bin"],
            consensus["mean_regret_ms"],
            marker="o",
            linewidth=2.2,
            label=f"{scenario_name} / Consensus",
        )
    axes[0].set_title("Consensus Regret Across Disagreement Bins", pad=14)
    axes[0].set_xlabel("Window disagreement bin")
    axes[0].set_ylabel("Mean regret (ms)")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    structural = plot_frame[plot_frame["scenario_name"] == "structural"].copy()
    if not structural.empty:
        pivot = structural.pivot(index="disagreement_bin", columns="policy_name", values="mean_realized_latency_ms")
        pivot = pivot.rename(columns=label_map)
        pivot = pivot[[col for col in ["Temporal", "Graph-aware", "Fusion", "Consensus"] if col in pivot.columns]]
        pivot.plot(kind="bar", ax=axes[1], width=0.78)
    axes[1].set_title("Structural Shift Latency by Disagreement Bin", pad=14)
    axes[1].set_xlabel("Window disagreement bin")
    axes[1].set_ylabel("Mean realized latency (ms)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, title="Policy")
    axes[0].legend(loc="upper left", frameon=True)
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
            group["mean_regret_ms"],
            marker="o",
            linewidth=2.2,
            label=scenario_name,
        )
    axes[0].set_title("Consensus Penalty Sweep: Latency", pad=14)
    axes[0].set_xlabel("Disagreement penalty $\\lambda$")
    axes[0].set_ylabel("Mean realized latency (ms)")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[1].set_title("Consensus Penalty Sweep: Regret", pad=14)
    axes[1].set_xlabel("Disagreement penalty $\\lambda$")
    axes[1].set_ylabel("Mean regret (ms)")
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
    args = parser.parse_args()

    forecast_metrics = pd.read_csv(_resolve_repo_path(args.forecast_metrics))
    graph_metrics = pd.read_csv(_resolve_repo_path(args.graph_metrics))
    policy_summary = pd.read_csv(_resolve_repo_path(args.policy_summary))
    ablation_summary_path = _resolve_repo_path(args.ablation_summary)
    stress_forecast = pd.read_csv(_resolve_repo_path(args.stress_forecast_metrics))
    stress_policy = pd.read_csv(_resolve_repo_path(args.stress_policy_summary))
    disagreement_summary_path = _resolve_repo_path(args.disagreement_summary)
    penalty_sweep_path = _resolve_repo_path(args.penalty_sweep)

    plots_dir = ensure_parent(_resolve_repo_path(f"{args.plots_dir}/.keep")).parent
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

    print(f"plots_written={plots_dir}")
    for file_path in sorted(plots_dir.glob("*.png")):
        print(file_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
