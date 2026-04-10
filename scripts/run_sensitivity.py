#!/usr/bin/env python3
"""Run all sensitivity/scenario analyses for Etosha.

Produces:
  - fig_randomization_robustness.png
  - fig_dry_season_shift.png
  - fig_wildfire_reallocation.png
  - fig_tech_human_pareto.png
  - fig_diminishing_returns_budget.png
  - fig_diminishing_returns_rangers.png
  - fig_equity_tradeoff.png
  - fig_disruption_pareto.png
  - fig_staffing_inverse.png
  - table_scenario_comparison.csv
  - table_sensitivity_summary.csv
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
from src.allocator import greedy_allocate_fast, equity_repair
from src.scores import (
    ppi, ppi_disaggregated, coverage_fraction, equity_index,
    disruption_score, robustness_score, response_time, cost_efficiency,
)
from src.scenarios import (
    run_baseline, run_dry_season, run_wildfire,
    run_tech_human_pareto, run_diminishing_returns_budget,
    run_diminishing_returns_rangers, run_equity_sweep,
    run_disruption_pareto, run_staffing_inverse,
)
from src.plotting import (
    plot_robustness_comparison, plot_comparison_maps,
    plot_pareto, plot_diminishing_returns, plot_equity_tradeoff,
    plot_staffing_inverse,
)

SEED = 42
ROOT = Path(__file__).parent.parent
CONFIG = ROOT / "configs" / "etosha.yaml"
RESOURCES = ROOT / "configs" / "resources.yaml"


def main() -> None:
    print("=" * 60)
    print("ETOSHA — SENSITIVITY & SCENARIO ANALYSIS")
    print("=" * 60)
    t0 = time.time()

    park = load_park(CONFIG, seed=SEED)
    resource_types, kernel = load_resource_types(RESOURCES, park.config)

    tables_dir = ROOT / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_dir = ROOT / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Baseline ----
    print("\n[1/9] Baseline...")
    baseline_alloc, baseline_rs, baseline_scores = run_baseline(
        park, resource_types, kernel, season="dry", seed=SEED, verbose=True
    )
    print(f"  Baseline PPI: {baseline_scores['ppi']:.4f}")

    # ---- 2. Randomization ----
    print("\n[2/9] Randomization robustness...")
    det_score = robustness_score(baseline_alloc, randomized=False)
    rand_score = robustness_score(baseline_alloc, randomized=True, seed=SEED)
    print(f"  Deterministic: {det_score:.3f}, Randomized: {rand_score:.3f}")
    plot_robustness_comparison(det_score, rand_score, name="fig_randomization_robustness", park_name="Etosha")
    print("  -> fig_randomization_robustness.png")

    # ---- 3. Dry Season ----
    print("\n[3/9] Dry season shift...")
    # Wet season baseline for comparison
    wet_rs = build_risk_surface(park, season="wet")
    wet_budget = baseline_alloc.budget_limit
    wet_caps = {rt.name: int(baseline_alloc.supply_caps[k])
                for k, rt in enumerate(baseline_alloc.resource_types)}
    wet_alloc = make_empty_allocation(park, wet_rs, resource_types, wet_budget, wet_caps, kernel)
    wet_alloc = greedy_allocate_fast(wet_alloc, verbose=False)

    dry_alloc, dry_rs, dry_scores = run_dry_season(
        park, resource_types, kernel, baseline_alloc, seed=SEED
    )
    print(f"  Dry PPI: {dry_scores['ppi']:.4f}, Wet PPI: {ppi(wet_alloc):.4f}")
    plot_comparison_maps(wet_alloc, dry_alloc,
                         name="fig_dry_season_shift",
                         title_before="Wet Season Allocation",
                         title_after="Dry Season Allocation",
                         suptitle="Etosha — Seasonal Resource Shift")
    print("  -> fig_dry_season_shift.png")

    # ---- 4. Wildfire ----
    print("\n[4/9] Wildfire reallocation...")
    fire_alloc, fire_rs, fire_scores = run_wildfire(
        park, resource_types, kernel, baseline_alloc,
        burn_center=(120, 70), burn_radius=20.0, seed=SEED
    )
    print(f"  Post-fire PPI: {fire_scores['ppi']:.4f} (baseline: {baseline_scores['ppi']:.4f})")
    plot_comparison_maps(baseline_alloc, fire_alloc,
                         name="fig_wildfire_reallocation",
                         title_before="Pre-Fire Allocation",
                         title_after="Post-Fire Reallocation",
                         suptitle="Etosha — Wildfire Response")
    print("  -> fig_wildfire_reallocation.png")

    # ---- 5. Tech vs Human Pareto ----
    print("\n[5/9] Tech vs Human Pareto...")
    pareto = run_tech_human_pareto(
        park, resource_types, kernel, total_budget=400.0, n_points=11
    )
    # Find elbow (max curvature)
    ppis = pareto["ppi"]
    fracs = pareto["ranger_fraction"]
    # Simple elbow: point farthest from the line connecting endpoints
    line_start = np.array([fracs[0], ppis[0]])
    line_end = np.array([fracs[-1], ppis[-1]])
    line_dir = line_end - line_start
    line_len = np.linalg.norm(line_dir)
    if line_len > 1e-12:
        dists_to_line = []
        for i in range(len(fracs)):
            pt = np.array([fracs[i], ppis[i]])
            cross = abs(np.cross(line_dir, line_start - pt))
            dists_to_line.append(cross / line_len)
        elbow_idx = int(np.argmax(dists_to_line))
    else:
        elbow_idx = len(fracs) // 2

    plot_pareto(fracs, ppis, "Ranger Budget Fraction", "PPI",
                name="fig_tech_human_pareto",
                title="Etosha — Rangers vs. Drones Pareto Frontier",
                highlight_idx=elbow_idx,
                highlight_label=f"Elbow ({fracs[elbow_idx]:.0%} rangers)")
    print(f"  Elbow at {fracs[elbow_idx]:.0%} rangers")
    print("  -> fig_tech_human_pareto.png")

    # ---- 6. Diminishing Returns ----
    print("\n[6/9] Diminishing returns...")
    dr_budget = run_diminishing_returns_budget(park, resource_types, kernel)
    plot_diminishing_returns(dr_budget["budgets"], dr_budget["ppi"],
                            "Total Budget", name="fig_diminishing_returns_budget",
                            title="Etosha — PPI vs. Total Budget")
    print("  -> fig_diminishing_returns_budget.png")

    dr_rangers = run_diminishing_returns_rangers(park, resource_types, kernel)
    plot_diminishing_returns(dr_rangers["ranger_counts"], dr_rangers["ppi"],
                            "Number of Rangers", name="fig_diminishing_returns_rangers",
                            title="Etosha — PPI vs. Ranger Count",
                            vline=295, vline_label="Current (295 rangers)")
    print("  -> fig_diminishing_returns_rangers.png")

    # ---- 7. Equity ----
    print("\n[7/9] Equity tradeoff...")
    eq_sweep = run_equity_sweep(park, resource_types, kernel, baseline_alloc)
    plot_equity_tradeoff(eq_sweep["p_min"], eq_sweep["ppi"], eq_sweep["equity"],
                         name="fig_equity_tradeoff", park_name="Etosha")
    print("  -> fig_equity_tradeoff.png")

    # ---- 8. Disruption Pareto ----
    print("\n[8/9] Disruption Pareto...")
    disr_pareto = run_disruption_pareto(park, resource_types, kernel, n_points=9)
    plot_pareto(disr_pareto["disruption"], disr_pareto["ppi"],
                "Disruption Score", "PPI",
                name="fig_disruption_pareto",
                title="Etosha — Protection vs. Wildlife Disruption")
    print("  -> fig_disruption_pareto.png")

    # ---- 9. Staffing Inverse ----
    print("\n[9/9] Staffing inverse...")
    staffing = run_staffing_inverse(park, resource_types, kernel)
    plot_staffing_inverse(staffing["targets"], staffing["rangers_needed"],
                          name="fig_staffing_inverse", park_name="Etosha",
                          current_rangers=295)
    print("  -> fig_staffing_inverse.png")

    # ---- Tables ----
    print("\nSaving tables...")

    # Scenario comparison table
    scenario_rows = [
        {"scenario": "baseline", **{k: v for k, v in baseline_scores.items()
                                     if not isinstance(v, dict)}},
        {"scenario": "dry_season", **{k: v for k, v in dry_scores.items()
                                       if not isinstance(v, dict)}},
        {"scenario": "wildfire", **{k: v for k, v in fire_scores.items()
                                     if not isinstance(v, dict)}},
    ]
    # Add equity and randomization info
    scenario_rows[0]["robustness_det"] = det_score
    scenario_rows[0]["robustness_rand"] = rand_score

    df_scenarios = pd.DataFrame(scenario_rows)
    df_scenarios.to_csv(tables_dir / "table_scenario_comparison.csv", index=False)
    print("  -> table_scenario_comparison.csv")

    # Sensitivity summary
    sens_rows = []
    for i, b in enumerate(dr_budget["budgets"]):
        sens_rows.append({"parameter": "budget", "value": float(b), "ppi": dr_budget["ppi"][i]})
    for i, n in enumerate(dr_rangers["ranger_counts"]):
        sens_rows.append({"parameter": "rangers", "value": float(n), "ppi": dr_rangers["ppi"][i]})
    for i, pm in enumerate(eq_sweep["p_min"]):
        sens_rows.append({"parameter": "equity_floor", "value": float(pm),
                          "ppi": eq_sweep["ppi"][i]})

    df_sens = pd.DataFrame(sens_rows)
    df_sens.to_csv(tables_dir / "table_sensitivity_summary.csv", index=False)
    print("  -> table_sensitivity_summary.csv")

    # Save results
    all_results = {
        "baseline": baseline_scores,
        "dry_season": dry_scores,
        "wildfire": fire_scores,
        "tech_human_pareto": {
            "fractions": pareto["ranger_fraction"].tolist(),
            "ppi": pareto["ppi"].tolist(),
            "elbow_idx": elbow_idx,
        },
        "diminishing_budget": {
            "budgets": dr_budget["budgets"].tolist(),
            "ppi": dr_budget["ppi"].tolist(),
        },
        "diminishing_rangers": {
            "rangers": dr_rangers["ranger_counts"].tolist(),
            "ppi": dr_rangers["ppi"].tolist(),
        },
        "equity": {
            "p_min": eq_sweep["p_min"].tolist(),
            "ppi": eq_sweep["ppi"].tolist(),
            "equity": eq_sweep["equity"].tolist(),
        },
        "staffing": {
            "targets": staffing["targets"].tolist(),
            "rangers": staffing["rangers_needed"].tolist(),
        },
    }
    with open(results_dir / "etosha_sensitivity.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()
