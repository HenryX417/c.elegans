#!/usr/bin/env python3
"""Run Etosha baseline: risk surface, allocation, PPI, temporal simulation.

Produces:
  - fig_etosha_risk_surface.png
  - fig_etosha_baseline_allocation.png
  - fig_etosha_ppi_disaggregated.png
  - fig_temporal_30day.png
  - table_disaggregated_ppi.csv
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.park import load_park
from src.risk import build_risk_surface
from src.resources import load_resource_types, make_empty_allocation
from src.allocator import greedy_allocate_fast, lp_upper_bound, compute_ppi
from src.scores import ppi, ppi_disaggregated, coverage_fraction, equity_index, disruption_score, robustness_score, response_time
from src.temporal import simulate
from src.plotting import (
    plot_risk_surface, plot_allocation, plot_ppi_disaggregated, plot_temporal,
)

SEED = 42
ROOT = Path(__file__).parent.parent
CONFIG = ROOT / "configs" / "etosha.yaml"
RESOURCES = ROOT / "configs" / "resources.yaml"


def main() -> None:
    print("=" * 60)
    print("ETOSHA NATIONAL PARK — BASELINE RUN")
    print("=" * 60)

    # 1. Load park
    t0 = time.time()
    park = load_park(CONFIG, seed=SEED)
    print(f"\nPark loaded: {park.name}")
    print(f"  Grid: {park.nx}x{park.ny} = {park.nx * park.ny} cells, {park.n_cells} interior")

    # 2. Risk surface
    rs = build_risk_surface(park, season="dry")
    print(f"  Total risk: {rs.total_risk:.2f}")
    plot_risk_surface(park, rs, name="fig_etosha_risk_surface")
    print("  -> fig_etosha_risk_surface.png")

    # 3. Resources + allocation (baseline: rangers only)
    resource_types, kernel = load_resource_types(RESOURCES, park.config)
    n_rangers = park.config["current_rangers"]
    budget = n_rangers * 1.0
    caps = {rt.name: (n_rangers if rt.name == "ranger_foot_team" else 0)
            for rt in resource_types}

    alloc = make_empty_allocation(park, rs, resource_types, budget, caps, kernel)
    print(f"\nRunning greedy allocation (budget={budget:.0f}, rangers={n_rangers})...")
    alloc = greedy_allocate_fast(alloc, verbose=True)

    # 4. Scores
    ppi_val = ppi(alloc)
    disagg = ppi_disaggregated(alloc)
    cov50 = coverage_fraction(alloc, threshold=0.5)
    cov30 = coverage_fraction(alloc, threshold=0.3)
    eq = equity_index(alloc)
    disr = disruption_score(alloc)
    rob_det = robustness_score(alloc, randomized=False)
    rob_rand = robustness_score(alloc, randomized=True, seed=SEED)
    resp = response_time(alloc)

    print(f"\n--- SCORES ---")
    print(f"  PPI:              {ppi_val:.4f}")
    print(f"  Coverage (p≥0.5): {cov50:.3f}")
    print(f"  Coverage (p≥0.3): {cov30:.3f}")
    print(f"  Equity min/mean:  {eq['min_mean_ratio']:.3f}")
    print(f"  Equity Gini:      {eq['gini']:.3f}")
    print(f"  Disruption:       {disr:.2f}")
    print(f"  Robustness (det): {rob_det:.3f}")
    print(f"  Robustness (rnd): {rob_rand:.3f}")
    print(f"  Response time:    {resp:.2f} hrs")

    # LP upper bound
    print("\nComputing LP upper bound...")
    ub = lp_upper_bound(alloc)
    if ub < float("inf"):
        print(f"  LP upper bound: {ub:.4f}")
        print(f"  Greedy/LP ratio: {ppi_val/ub:.3f} (theoretical min: {1-1/np.e:.3f})")
    else:
        print("  LP solver unavailable or infeasible")

    # 5. Plots
    plot_allocation(alloc, name="fig_etosha_baseline_allocation")
    print("\n  -> fig_etosha_baseline_allocation.png")

    plot_ppi_disaggregated(disagg, name="fig_etosha_ppi_disaggregated", park_name="Etosha")
    print("  -> fig_etosha_ppi_disaggregated.png")

    # 6. Temporal simulation
    print("\nRunning 30-day temporal simulation...")
    sim = simulate(alloc, n_days=30, threat_rate=5.0, randomized_patrols=True,
                   route_pool_size=5, seed=SEED, verbose=True)
    print(f"  Mean PPI: {sim.mean_ppi:.3f}")
    print(f"  Threats: {sim.total_threats} arrived, {sim.total_intercepted} intercepted "
          f"({sim.total_intercepted/max(sim.total_threats,1)*100:.1f}%)")

    intercepted_ts = np.array([d.threats_intercepted for d in sim.days])
    plot_temporal(sim.ppi_timeseries, intercepted_ts, name="fig_temporal_30day", park_name="Etosha")
    print("  -> fig_temporal_30day.png")

    # 7. Save tables
    tables_dir = ROOT / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Disaggregated PPI table
    rows = []
    for s, v in disagg["by_species"].items():
        rows.append({"category": "species", "name": s, "ppi": v})
    for t, v in disagg["by_threat"].items():
        rows.append({"category": "threat", "name": t, "ppi": v})
    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "table_disaggregated_ppi.csv", index=False)
    print("  -> table_disaggregated_ppi.csv")

    # 8. Save results JSON
    results_dir = ROOT / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    elapsed = time.time() - t0
    result = {
        "park": park.name,
        "seed": SEED,
        "season": "dry",
        "budget": budget,
        "ppi": ppi_val,
        "ppi_disaggregated": disagg,
        "coverage_50": cov50,
        "coverage_30": cov30,
        "equity": eq,
        "disruption": disr,
        "robustness_det": rob_det,
        "robustness_rand": rob_rand,
        "response_time_hrs": resp,
        "lp_upper_bound": ub if ub < float("inf") else None,
        "sim_mean_ppi": sim.mean_ppi,
        "sim_total_threats": sim.total_threats,
        "sim_total_intercepted": sim.total_intercepted,
        "elapsed_s": elapsed,
    }
    with open(results_dir / "etosha_baseline.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()
