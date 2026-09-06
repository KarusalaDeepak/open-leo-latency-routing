#!/usr/bin/env python3
"""Build a prospective, analytical evidence-gate design-sensitivity artifact.

This module deliberately does not reselect a gate configuration from measured
selection or test outcomes.  It reports exact zero-harm success-screen floors
and best-possible bounded-CVaR non-vacuity floors under a declared finite
family.  The current one-drive COMMECT result is repeated only as the invariant
fact that every requested applicability floor fails.

The CVaR floor is intentionally optimistic: every independent group contains
one observation, reactive latency equals the clipping cap, candidate latency
equals zero, and every group bears an opportunity.  It is therefore a lower
bound on the number of groups needed for *any* sample to pass the implemented
tail screen, not a power claim for a practical network effect.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "gate_design_sensitivity.yaml"

MAIN_OUTPUT_NAME = "gate_design_sensitivity.csv"
CANONICAL_OUTPUT_NAME = "gate_design_sensitivity_canonical_reference.csv"
METADATA_OUTPUT_NAME = "gate_design_sensitivity_metadata.json"

NO_TEST_TIME_TUNING_WARNING = (
    "Prospective design sensitivity only: do not choose a gate configuration "
    "after inspecting policy-selection or test outcomes. If configurations "
    "are outcome-selected, they must be added to the protected multiplicity "
    "family before data are observed."
)


def _require_unique_numeric_sequence(
    value: object,
    *,
    name: str,
    expected_length: int,
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise ValueError(f"{name} must contain exactly {expected_length} values")
    numbers = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in numbers):
        raise ValueError(f"{name} must contain only finite numbers")
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"{name} must not contain duplicate values")
    return numbers


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the dedicated design-only configuration."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("design-sensitivity config must be a mapping")
    if raw.get("schema_version") != 1:
        raise ValueError("design-sensitivity config schema_version must be 1")
    if raw.get("artifact_kind") != "prospective_gate_design_sensitivity":
        raise ValueError("config must be explicitly labeled prospective design")

    contract = raw.get("statistical_contract")
    grid = raw.get("requested_grid")
    canonical = raw.get("canonical_reference")
    measured = raw.get("measured_reference")
    if not all(
        isinstance(item, dict)
        for item in (contract, grid, canonical, measured)
    ):
        raise ValueError("config contract, grid, and references must be mappings")

    alpha = float(contract["familywise_alpha"])
    quantile = float(contract["cvar_quantile"])
    learned_count = int(contract["learned_candidate_count"])
    endpoint_families = int(contract["endpoint_family_count_per_candidate_use"])
    spacing = float(contract["cvar_grid_spacing_ms"])
    planned_uses = tuple(int(item) for item in contract["planned_gate_uses"])
    if not 0.0 < alpha < 1.0:
        raise ValueError("familywise alpha must be in (0, 1)")
    if not 0.0 < quantile < 1.0:
        raise ValueError("CVaR quantile must be in (0, 1)")
    if learned_count < 1 or endpoint_families != 4:
        raise ValueError("the production contract requires candidates >=1 and four families")
    if spacing <= 0.0 or planned_uses != (1, 5):
        raise ValueError("the declared artifact requires 0.05-ms spacing and U=(1,5)")

    n_min = _require_unique_numeric_sequence(
        grid.get("minimum_opportunity_bearing_groups"),
        name="minimum opportunity-bearing groups",
        expected_length=4,
    )
    epsilon = _require_unique_numeric_sequence(
        grid.get("success_noninferiority_margins"),
        name="success noninferiority margins",
        expected_length=3,
    )
    delta = _require_unique_numeric_sequence(
        grid.get("practical_cvar_gain_ms"),
        name="practical CVaR gains",
        expected_length=3,
    )
    caps = _require_unique_numeric_sequence(
        grid.get("latency_cap_ms"),
        name="latency caps",
        expected_length=3,
    )
    if n_min != (5.0, 10.0, 20.0, 50.0):
        raise ValueError("n_min grid must be exactly 5, 10, 20, 50")
    if epsilon != (0.01, 0.02, 0.05):
        raise ValueError("epsilon grid must be exactly .01, .02, .05")
    if delta != (1.0, 5.0, 10.0):
        raise ValueError("delta grid must be exactly 1, 5, 10 ms")
    if caps != (60.0, 200.0, 1000.0):
        raise ValueError("latency-cap grid must be exactly 60, 200, 1000 ms")
    if not all(float(item).is_integer() and item >= 1 for item in n_min):
        raise ValueError("n_min values must be positive integer counts")
    if not all(0.0 < item < 1.0 for item in epsilon):
        raise ValueError("success margins must be in (0, 1)")
    if not all(item > 0.0 for item in (*delta, *caps)):
        raise ValueError("gain margins and caps must be positive")

    if int(measured["independent_collection_groups"]) != 1:
        raise ValueError("the measured reference must retain COMMECT as one drive")
    if int(measured["opportunity_bearing_collection_groups"]) != 1:
        raise ValueError("the measured opportunity-bearing group count must be one")
    if bool(measured["admitted"]):
        raise ValueError("the measured reference must remain fail-closed")
    if float(canonical["latency_cap_ms"]) != 60_000.0:
        raise ValueError("canonical reference cap must remain 60,000 ms")

    raw["_config_path"] = str(path)
    return raw


def endpoint_alpha(config: dict[str, Any], planned_uses: int) -> float:
    contract = config["statistical_contract"]
    return float(contract["familywise_alpha"]) / (
        int(contract["endpoint_family_count_per_candidate_use"])
        * int(contract["learned_candidate_count"])
        * int(planned_uses)
    )


def zero_harm_success_group_floor(epsilon: float, alpha_endpoint: float) -> int:
    """Exact minimum G for a zero-harm Clopper--Pearson UCB <= epsilon."""

    if not 0.0 < epsilon < 1.0 or not 0.0 < alpha_endpoint < 1.0:
        raise ValueError("epsilon and endpoint alpha must be in (0, 1)")
    floor = max(1, math.ceil(math.log(alpha_endpoint) / math.log1p(-epsilon)))

    def passes(groups: int) -> bool:
        harmful_probability_ucb = 1.0 - alpha_endpoint ** (1.0 / groups)
        return harmful_probability_ucb <= epsilon

    while not passes(floor):
        floor += 1
    while floor > 1 and passes(floor - 1):
        floor -= 1
    return floor


def grid_contract(
    latency_cap_ms: float,
    spacing_ms: float,
    quantile: float,
) -> tuple[int, float, float]:
    """Return grid points, realized spacing, and worst half-cell correction."""

    intervals = round(latency_cap_ms / spacing_ms)
    if not math.isclose(
        intervals * spacing_ms,
        latency_cap_ms,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise ValueError("latency cap must be an integer multiple of grid spacing")
    points = int(intervals) + 1
    realized_spacing = round(latency_cap_ms / (points - 1), 12)
    tail_probability = 1.0 - quantile
    lipschitz = max(1.0, quantile / tail_probability)
    correction = round(0.5 * realized_spacing * lipschitz, 12)
    return points, realized_spacing, correction


def best_case_cvar_gain_lcb_ms(
    groups: int,
    *,
    latency_cap_ms: float,
    quantile: float,
    grid_points: int,
    grid_correction_ms: float,
    alpha_endpoint: float,
) -> float:
    """Implemented-bound LCB for the maximally favorable cap-to-zero sample."""

    if groups < 1:
        raise ValueError("group count must be positive")
    tail_probability = 1.0 - quantile
    concentration = math.sqrt(
        math.log((2.0 * grid_points) / alpha_endpoint) / (2.0 * groups)
    )
    if concentration <= quantile:
        reactive_lcb = max(0.0, latency_cap_ms - grid_correction_ms)
    else:
        reactive_lcb = max(
            0.0,
            latency_cap_ms * (1.0 - concentration) / tail_probability
            - grid_correction_ms,
        )
    candidate_ucb = latency_cap_ms * min(
        1.0,
        concentration / tail_probability,
    )
    return reactive_lcb - candidate_ucb


def best_case_cvar_group_floor(
    practical_gain_ms: float,
    *,
    latency_cap_ms: float,
    quantile: float,
    grid_points: int,
    grid_correction_ms: float,
    alpha_endpoint: float,
) -> int:
    """Minimum G at which even the cap-to-zero sample can pass the tail gate."""

    if practical_gain_ms + grid_correction_ms >= latency_cap_ms:
        raise ValueError("gain plus grid correction must be below the cap")
    tail_probability = 1.0 - quantile
    maximum_concentration = tail_probability * (
        1.0
        - (practical_gain_ms + grid_correction_ms) / latency_cap_ms
    )
    floor = max(
        1,
        math.ceil(
            math.log((2.0 * grid_points) / alpha_endpoint)
            / (2.0 * maximum_concentration**2)
        ),
    )

    def passes(groups: int) -> bool:
        return best_case_cvar_gain_lcb_ms(
            groups,
            latency_cap_ms=latency_cap_ms,
            quantile=quantile,
            grid_points=grid_points,
            grid_correction_ms=grid_correction_ms,
            alpha_endpoint=alpha_endpoint,
        ) >= practical_gain_ms

    while not passes(floor):
        floor += 1
    while floor > 1 and passes(floor - 1):
        floor -= 1
    return floor


def _common_contract(config: dict[str, Any]) -> dict[str, object]:
    contract = config["statistical_contract"]
    measured = config["measured_reference"]
    return {
        "artifact_scope": "prospective_design_only_not_measured_efficacy",
        "familywise_alpha": float(contract["familywise_alpha"]),
        "cvar_quantile": float(contract["cvar_quantile"]),
        "learned_candidate_count": int(contract["learned_candidate_count"]),
        "endpoint_family_count_per_candidate_use": int(
            contract["endpoint_family_count_per_candidate_use"]
        ),
        "measured_independent_group_count": int(
            measured["independent_collection_groups"]
        ),
        "measured_opportunity_group_count": int(
            measured["opportunity_bearing_collection_groups"]
        ),
        "measured_fixed_raw_opportunity_epochs": int(
            measured["fixed_raw_opportunity_epochs"]
        ),
        "measured_rolling_raw_opportunity_epochs_min": int(
            measured["rolling_raw_opportunity_epochs_min"]
        ),
        "measured_rolling_raw_opportunity_epochs_max": int(
            measured["rolling_raw_opportunity_epochs_max"]
        ),
        "measured_admitted": False,
        "measured_selection": "reactive",
        "measured_reason": str(measured["reason"]).strip(),
        "configuration_selection_allowed": False,
        "no_test_time_tuning_warning": NO_TEST_TIME_TUNING_WARNING,
    }


def build_requested_rows(config: dict[str, Any]) -> list[dict[str, object]]:
    """Build exactly the requested 4 x 3 x 3 x 3 Cartesian rows."""

    contract = config["statistical_contract"]
    grid = config["requested_grid"]
    common = _common_contract(config)
    quantile = float(contract["cvar_quantile"])
    spacing = float(contract["cvar_grid_spacing_ms"])
    rows: list[dict[str, object]] = []
    for n_min in grid["minimum_opportunity_bearing_groups"]:
        for epsilon in grid["success_noninferiority_margins"]:
            for practical_gain in grid["practical_cvar_gain_ms"]:
                for cap in grid["latency_cap_ms"]:
                    n_min_int = int(n_min)
                    epsilon_float = float(epsilon)
                    gain_float = float(practical_gain)
                    cap_float = float(cap)
                    points, realized_spacing, correction = grid_contract(
                        cap_float,
                        spacing,
                        quantile,
                    )
                    row: dict[str, object] = {
                        **common,
                        "n_min_opportunity_bearing_groups": n_min_int,
                        "epsilon_aggregate": epsilon_float,
                        "epsilon_opportunity": epsilon_float,
                        "practical_cvar_gain_ms": gain_float,
                        "latency_cap_ms": cap_float,
                        "cvar_grid_points": points,
                        "cvar_grid_spacing_ms": realized_spacing,
                        "cvar_grid_correction_ms": correction,
                        "zero_harm_assumption": True,
                        "all_groups_opportunity_bearing_assumption": True,
                        "tail_floor_sample": "reactive_at_cap_candidate_at_zero",
                    }
                    for planned_uses in (1, 5):
                        alpha_e = endpoint_alpha(config, planned_uses)
                        success_floor = zero_harm_success_group_floor(
                            epsilon_float,
                            alpha_e,
                        )
                        tail_floor = best_case_cvar_group_floor(
                            gain_float,
                            latency_cap_ms=cap_float,
                            quantile=quantile,
                            grid_points=points,
                            grid_correction_ms=correction,
                            alpha_endpoint=alpha_e,
                        )
                        prefix = f"u{planned_uses}"
                        row[f"{prefix}_planned_gate_uses"] = planned_uses
                        row[f"{prefix}_endpoint_alpha"] = alpha_e
                        row[f"{prefix}_success_zero_harm_group_floor"] = (
                            success_floor
                        )
                        row[f"{prefix}_tail_best_case_group_floor"] = tail_floor
                        row[f"{prefix}_joint_best_case_group_floor"] = max(
                            n_min_int,
                            success_floor,
                            tail_floor,
                        )
                    rows.append(row)
    if len(rows) != 108:
        raise AssertionError(f"requested Cartesian table has {len(rows)} rows")
    return rows


def build_canonical_reference_rows(
    config: dict[str, Any],
) -> list[dict[str, object]]:
    """Build a separate reference for the submitted 60,000-ms cap."""

    contract = config["statistical_contract"]
    canonical = config["canonical_reference"]
    common = _common_contract(config)
    quantile = float(contract["cvar_quantile"])
    spacing = float(contract["cvar_grid_spacing_ms"])
    cap = float(canonical["latency_cap_ms"])
    epsilon = float(canonical["success_noninferiority_margin"])
    n_min = int(canonical["minimum_opportunity_bearing_groups"])
    points, realized_spacing, correction = grid_contract(
        cap,
        spacing,
        quantile,
    )
    rows: list[dict[str, object]] = []
    for practical_gain in config["requested_grid"]["practical_cvar_gain_ms"]:
        gain = float(practical_gain)
        row: dict[str, object] = {
            **common,
            "canonical_reference": True,
            "n_min_opportunity_bearing_groups": n_min,
            "epsilon_aggregate": epsilon,
            "epsilon_opportunity": epsilon,
            "practical_cvar_gain_ms": gain,
            "latency_cap_ms": cap,
            "cvar_grid_points": points,
            "cvar_grid_spacing_ms": realized_spacing,
            "cvar_grid_correction_ms": correction,
            "zero_harm_assumption": True,
            "all_groups_opportunity_bearing_assumption": True,
            "tail_floor_sample": "reactive_at_cap_candidate_at_zero",
        }
        for planned_uses in (1, 5):
            alpha_e = endpoint_alpha(config, planned_uses)
            success_floor = zero_harm_success_group_floor(epsilon, alpha_e)
            tail_floor = best_case_cvar_group_floor(
                gain,
                latency_cap_ms=cap,
                quantile=quantile,
                grid_points=points,
                grid_correction_ms=correction,
                alpha_endpoint=alpha_e,
            )
            prefix = f"u{planned_uses}"
            row[f"{prefix}_planned_gate_uses"] = planned_uses
            row[f"{prefix}_endpoint_alpha"] = alpha_e
            row[f"{prefix}_success_zero_harm_group_floor"] = success_floor
            row[f"{prefix}_tail_best_case_group_floor"] = tail_floor
            row[f"{prefix}_joint_best_case_group_floor"] = max(
                n_min,
                success_floor,
                tail_floor,
            )
        rows.append(row)
    return rows


def _csv_value(value: object) -> object:
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty design table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_metadata(
    config: dict[str, Any],
    requested_rows: list[dict[str, object]],
    canonical_rows: list[dict[str, object]],
    output_hashes: dict[str, str],
) -> dict[str, object]:
    contract = config["statistical_contract"]
    return {
        "schema_version": 1,
        "artifact_kind": "prospective_gate_design_sensitivity",
        "measured_efficacy_evidence": False,
        "configuration_selection_allowed": False,
        "warning": NO_TEST_TIME_TUNING_WARNING,
        "requested_cartesian_row_count": len(requested_rows),
        "canonical_reference_row_count": len(canonical_rows),
        "requested_grid": config["requested_grid"],
        "canonical_reference": config["canonical_reference"],
        "measured_reference": config["measured_reference"],
        "statistical_contract": contract,
        "formulas": {
            "endpoint_alpha": "alpha/(4*K*U)",
            "zero_harm_success_group_floor": (
                "ceil(log(alpha_endpoint)/log(1-epsilon)), adjusted to the "
                "smallest integer satisfying 1-alpha_endpoint^(1/G)<=epsilon"
            ),
            "grid_points": "latency_cap_ms/grid_spacing_ms + 1",
            "grid_correction_ms": (
                "0.5*grid_spacing_ms*max(1,q/(1-q))"
            ),
            "tail_best_case_group_floor": (
                "smallest G whose production bounded-CVaR gain LCB reaches "
                "delta for reactive=latency_cap and candidate=0"
            ),
            "joint_best_case_group_floor": (
                "max(n_min, success_zero_harm_group_floor, "
                "tail_best_case_group_floor) when every group bears an opportunity"
            ),
        },
        "assumptions": [
            "K=2 frozen learned candidates and U in {1,5} are declared before data.",
            "Collection groups are mutually independent and exchangeable after conditioning on every frozen artifact.",
            "Both exact success floors assume zero harmful groups.",
            "Every group is opportunity-bearing for the joint best-case floor.",
            "The tail floor uses the maximally favorable cap-to-zero sample and is a non-vacuity floor, not statistical power for a practical effect.",
            "The latency cap is an estimand-defining engineering bound, not a parameter to choose after inspecting outcomes.",
            "The measured COMMECT campaign remains one group and is never reinterpreted as admission evidence.",
        ],
        "output_sha256": output_hashes,
    }


def generate_artifact(
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Generate deterministic CSV/JSON outputs in a new artifact directory."""

    requested_rows = build_requested_rows(config)
    canonical_rows = build_canonical_reference_rows(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / MAIN_OUTPUT_NAME
    canonical_path = output_dir / CANONICAL_OUTPUT_NAME
    _write_csv(main_path, requested_rows)
    _write_csv(canonical_path, canonical_rows)
    hashes = {
        MAIN_OUTPUT_NAME: _sha256(main_path),
        CANONICAL_OUTPUT_NAME: _sha256(canonical_path),
    }
    metadata = build_metadata(config, requested_rows, canonical_rows, hashes)
    (output_dir / METADATA_OUTPUT_NAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return requested_rows, canonical_rows, metadata


def _resolve_output(config: dict[str, Any], override: str | None) -> Path:
    value = override if override is not None else str(config["output_dir"])
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the prospective evidence-gate design-sensitivity table."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir")
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = load_config(Path(args.config))
    output_dir = _resolve_output(config, args.output_dir)
    requested, canonical, _ = generate_artifact(config, output_dir)
    print(f"requested_rows={len(requested)}")
    print(f"canonical_reference_rows={len(canonical)}")
    print(f"outputs={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
