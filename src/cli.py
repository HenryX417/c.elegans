"""CLI entry point: python -m src.cli run --config ..."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def run(args: argparse.Namespace) -> None:
    """Run a scenario and produce figures + tables."""
    from .park import load_park
    from .resources import load_resource_types, make_empty_allocation
    from .risk import build_risk_surface
    from .allocator import greedy_allocate_fast, compute_ppi
    from .scores import ppi, ppi_disaggregated
    from .plotting import plot_risk_surface, plot_allocation, plot_ppi_disaggregated

    config_path = Path(args.config)
    resource_path = Path(args.resources) if args.resources else config_path.parent / "resources.yaml"
    seed = args.seed

    print(f"Loading park config: {config_path}")
    park = load_park(config_path, seed=seed)
    print(f"  Grid: {park.nx}x{park.ny} = {park.nx * park.ny} cells, {park.n_cells} interior")

    resource_types, kernel = load_resource_types(resource_path, park.config)
    print(f"  Resource types: {[rt.name for rt in resource_types]}")

    season = args.season or "dry"
    print(f"  Building risk surface (season={season})...")
    rs = build_risk_surface(park, season=season)
    print(f"  Total risk: {rs.total_risk:.2f}")

    # Plot risk surface
    park_prefix = config_path.stem
    plot_risk_surface(park, rs, name=f"fig_{park_prefix}_risk_surface")
    print(f"  Saved: fig_{park_prefix}_risk_surface.png")

    # Allocate
    budget = args.budget or park.config.get("current_rangers", 100) * 1.0
    caps = None
    if not args.all_resources:
        caps = {"ranger_foot_team": park.config.get("current_rangers", 100)}

    alloc = make_empty_allocation(park, rs, resource_types, budget, caps, kernel)
    print(f"  Running greedy allocation (budget={budget:.0f})...")
    t0 = time.time()
    alloc = greedy_allocate_fast(alloc, verbose=args.verbose)
    elapsed = time.time() - t0
    print(f"  Allocation done in {elapsed:.1f}s")

    # Scores
    ppi_val = ppi(alloc)
    disagg = ppi_disaggregated(alloc)
    print(f"  PPI = {ppi_val:.4f}")
    print(f"  PPI by species: {disagg['by_species']}")
    print(f"  PPI by threat: {disagg['by_threat']}")

    # Plot allocation
    plot_allocation(alloc, name=f"fig_{park_prefix}_baseline_allocation")
    print(f"  Saved: fig_{park_prefix}_baseline_allocation.png")

    # Plot disaggregated PPI
    plot_ppi_disaggregated(disagg, name=f"fig_{park_prefix}_ppi_disaggregated", park_name=park.name)
    print(f"  Saved: fig_{park_prefix}_ppi_disaggregated.png")

    # Save results JSON
    results_dir = Path(__file__).parent.parent / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "park": park.name,
        "config": str(config_path),
        "config_hash": park.config_hash(),
        "seed": seed,
        "season": season,
        "budget": budget,
        "ppi": ppi_val,
        "ppi_disaggregated": disagg,
        "elapsed_s": elapsed,
    }
    result_path = results_dir / f"{park_prefix}_baseline_{seed}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Saved: {result_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wildlife Protection Model CLI")
    subparsers = parser.add_subparsers(dest="command")

    # run command
    run_parser = subparsers.add_parser("run", help="Run a scenario")
    run_parser.add_argument("--config", required=True, help="Path to park YAML config")
    run_parser.add_argument("--resources", help="Path to resources YAML (default: same dir as config)")
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--season", default="dry")
    run_parser.add_argument("--budget", type=float, default=None)
    run_parser.add_argument("--all-resources", action="store_true",
                            help="Allow all resource types (not just rangers)")
    run_parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
