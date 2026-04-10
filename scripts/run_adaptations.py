#!/usr/bin/env python3
"""Run adaptation analysis for Yellowstone and Sierra del Divisor.

Produces:
  - fig_yellowstone_risk_surface.png
  - fig_yellowstone_allocation.png
  - fig_sierra_risk_surface.png
  - fig_sierra_allocation.png
  - fig_adaptation_comparison.png
  - table_adaptation_parameters.csv
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.park import load_park
from src.risk import build_risk_surface
from src.resources import load_resource_types, make_empty_allocation
from src.allocator import greedy_allocate_fast
from src.scores import (
    ppi, ppi_disaggregated, coverage_fraction, equity_index,
    disruption_score, robustness_score, response_time,
)
from src.plotting import (
    plot_risk_surface, plot_allocation, plot_adaptation_comparison,
)

SEED = 42
ROOT = Path(__file__).parent.parent
RESOURCES = ROOT / "configs" / "resources.yaml"


def run_park(config_name: str, prefix: str, park_label: str) -> dict:
    """Run baseline analysis for one park."""
    config_path = ROOT / "configs" / config_name
    print(f"\n--- {park_label} ---")

    park = load_park(config_path, seed=SEED)
    print(f"  Grid: {park.nx}x{park.ny}, {park.n_cells} interior cells")

    resource_types, kernel = load_resource_types(RESOURCES, park.config)
    rs = build_risk_surface(park, season="dry")
    print(f"  Total risk: {rs.total_risk:.2f}")

    plot_risk_surface(park, rs, name=f"fig_{prefix}_risk_surface")
    print(f"  -> fig_{prefix}_risk_surface.png")

    # Allocate with park's current rangers (baseline: rangers only)
    n_rangers = park.config.get("current_rangers", 100)
    budget = n_rangers * 1.0
    caps = {rt.name: (n_rangers if rt.name == "ranger_foot_team" else 0)
            for rt in resource_types}

    alloc = make_empty_allocation(park, rs, resource_types, budget, caps, kernel)
    alloc = greedy_allocate_fast(alloc, verbose=True)

    plot_allocation(alloc, name=f"fig_{prefix}_allocation",
                    title_suffix=f"Baseline Allocation ({n_rangers} rangers)")
    print(f"  -> fig_{prefix}_allocation.png")

    # Scores
    scores = {
        "park": park_label,
        "area_km2": park.config["area_km2"],
        "current_rangers": n_rangers,
        "grid_cells": park.n_cells,
        "ppi": ppi(alloc),
        "coverage_50": coverage_fraction(alloc, threshold=0.5),
        "coverage_30": coverage_fraction(alloc, threshold=0.3),
        "equity_min_mean": equity_index(alloc)["min_mean_ratio"],
        "equity_gini": equity_index(alloc)["gini"],
        "disruption": disruption_score(alloc),
        "robustness_det": robustness_score(alloc, randomized=False),
        "robustness_rand": robustness_score(alloc, randomized=True, seed=SEED),
        "response_time_hrs": response_time(alloc),
    }
    disagg = ppi_disaggregated(alloc)
    scores["ppi_disaggregated"] = disagg

    print(f"  PPI: {scores['ppi']:.4f}")
    return scores


def main() -> None:
    print("=" * 60)
    print("ADAPTATION ANALYSIS — THREE PARKS")
    print("=" * 60)
    t0 = time.time()

    tables_dir = ROOT / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_dir = ROOT / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run all three parks
    etosha = run_park("etosha.yaml", "etosha", "Etosha")
    yellowstone = run_park("yellowstone.yaml", "yellowstone", "Yellowstone")
    sierra = run_park("sierra_del_divisor.yaml", "sierra", "Sierra del Divisor")

    all_parks = [etosha, yellowstone, sierra]
    park_names = [p["park"] for p in all_parks]

    # Comparison figure
    metrics = {
        "PPI": [p["ppi"] for p in all_parks],
        "Coverage\n(p≥0.3)": [p["coverage_30"] for p in all_parks],
        "Equity\n(min/mean)": [p["equity_min_mean"] for p in all_parks],
        "Robustness\n(random)": [p["robustness_rand"] for p in all_parks],
    }
    plot_adaptation_comparison(park_names, metrics, name="fig_adaptation_comparison")
    print("\n  -> fig_adaptation_comparison.png")

    # Adaptation parameters table
    param_rows = []
    for p in all_parks:
        row = {
            "park": p["park"],
            "area_km2": p["area_km2"],
            "rangers": p["current_rangers"],
            "grid_cells": p["grid_cells"],
            "ppi": round(p["ppi"], 4),
            "coverage_30": round(p["coverage_30"], 3),
            "equity": round(p["equity_min_mean"], 3),
            "disruption": round(p["disruption"], 2),
            "response_time_hrs": round(p["response_time_hrs"], 2),
        }
        param_rows.append(row)

    df_params = pd.DataFrame(param_rows)
    df_params.to_csv(tables_dir / "table_adaptation_parameters.csv", index=False)
    print("  -> table_adaptation_parameters.csv")

    # Save full results
    serializable = []
    for p in all_parks:
        s = {k: v for k, v in p.items() if k != "ppi_disaggregated"}
        s["ppi_disaggregated"] = p["ppi_disaggregated"]
        serializable.append(s)

    with open(results_dir / "adaptation_results.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()
