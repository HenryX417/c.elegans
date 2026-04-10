"""Scenario mode wrappers.

Each scenario modifies the baseline config or allocation, re-runs the solver,
and produces comparison metrics and figures. All scenarios are config-driven
via YAML files in configs/scenarios/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .allocator import compute_ppi, equity_repair, greedy_allocate_fast
from .park import Park, load_park
from .resources import (
    Allocation,
    load_resource_types,
    make_empty_allocation,
)
from .risk import RiskSurface, build_risk_surface
from .scores import (
    coverage_fraction,
    disruption_score,
    equity_index,
    ppi,
    ppi_disaggregated,
    robustness_score,
    response_time,
)


def _default_budget(park: Park, resource_types: list, scenario_cfg: dict) -> float:
    """Compute default budget from park staff + scenario overrides."""
    rangers = scenario_cfg.get("rangers", park.config.get("current_rangers", 100))
    # Budget = rangers * cost of ranger_foot_team + extra tech budget
    ranger_cost = 1.0
    for rt in resource_types:
        if rt.name == "ranger_foot_team":
            ranger_cost = rt.cost_per_unit
            break
    tech_budget = scenario_cfg.get("tech_budget", 0)
    return rangers * ranger_cost + tech_budget


def run_baseline(
    park: Park,
    resource_types: list,
    kernel: str,
    season: str = "dry",
    seed: int = 42,
    verbose: bool = False,
) -> tuple[Allocation, RiskSurface, dict]:
    """Run baseline scenario: current resources, no extras.

    Returns:
        (allocation, risk_surface, scores_dict)
    """
    rs = build_risk_surface(park, season=season)
    budget = park.config.get("current_rangers", 100) * 1.0  # ranger cost = 1.0
    caps = {"ranger_foot_team": park.config.get("current_rangers", 100)}

    alloc = make_empty_allocation(park, rs, resource_types, budget, caps, kernel)
    alloc = greedy_allocate_fast(alloc, verbose=verbose)

    scores = _compute_all_scores(alloc)
    return alloc, rs, scores


def run_dry_season(
    park: Park, resource_types: list, kernel: str, baseline_alloc: Allocation,
    seed: int = 42, verbose: bool = False,
) -> tuple[Allocation, RiskSurface, dict]:
    """Dry season mode: boosted waterhole-proximity weights."""
    rs = build_risk_surface(park, season="dry")
    budget = baseline_alloc.budget_limit
    caps_dict = {rt.name: int(baseline_alloc.supply_caps[k])
                 for k, rt in enumerate(baseline_alloc.resource_types)}

    alloc = make_empty_allocation(park, rs, resource_types, budget, caps_dict, kernel)
    alloc = greedy_allocate_fast(alloc, verbose=verbose)

    scores = _compute_all_scores(alloc)
    return alloc, rs, scores


def run_wildfire(
    park: Park, resource_types: list, kernel: str, baseline_alloc: Allocation,
    burn_center: tuple[float, float] = (120, 70), burn_radius: float = 20.0,
    seed: int = 42, verbose: bool = False,
) -> tuple[Allocation, RiskSurface, dict]:
    """Wildfire mode: mark cells as burned/inaccessible, boost adjacent risk."""
    rs = build_risk_surface(park, season="dry")

    # Mark burned cells
    dists_to_center = np.linalg.norm(
        park.cell_centers - np.array(burn_center), axis=1
    )
    burned = dists_to_center <= burn_radius
    adjacent = (dists_to_center > burn_radius) & (dists_to_center <= burn_radius * 1.5)

    # Zero risk in burned cells (inaccessible), boost risk in adjacent (displacement)
    rs_modified = RiskSurface(
        park=rs.park,
        risk_st=rs.risk_st.copy(),
        risk_by_threat=rs.risk_by_threat.copy(),
        risk_by_species=rs.risk_by_species.copy(),
        risk=rs.risk.copy(),
        species_names=rs.species_names,
        threat_names=rs.threat_names,
        species_weights=rs.species_weights,
        threat_weights=rs.threat_weights,
    )
    rs_modified.risk[burned] = 0.0
    rs_modified.risk_by_threat[burned] = 0.0
    rs_modified.risk[adjacent] *= 2.0  # displaced animals + opportunistic poachers
    rs_modified.risk_by_threat[adjacent] *= 2.0

    budget = baseline_alloc.budget_limit
    caps_dict = {rt.name: int(baseline_alloc.supply_caps[k])
                 for k, rt in enumerate(baseline_alloc.resource_types)}

    alloc = make_empty_allocation(park, rs_modified, resource_types, budget, caps_dict, kernel)
    alloc = greedy_allocate_fast(alloc, verbose=verbose)

    scores = _compute_all_scores(alloc)
    return alloc, rs_modified, scores


def run_tech_human_pareto(
    park: Park, resource_types: list, kernel: str,
    total_budget: float = 400.0, n_points: int = 11,
    seed: int = 42, verbose: bool = False,
) -> dict[str, Any]:
    """Tech vs Human Pareto: sweep budget split between rangers and drones."""
    rs = build_risk_surface(park, season="dry")

    fractions = np.linspace(0, 1, n_points)
    results = {"ranger_fraction": fractions, "ppi": [], "cost": [], "disruption": []}

    for frac in fractions:
        ranger_budget = total_budget * frac
        drone_budget = total_budget * (1 - frac)

        ranger_cost = 1.0
        drone_cost = 3.0
        for rt in resource_types:
            if rt.name == "ranger_foot_team":
                ranger_cost = rt.cost_per_unit
            elif rt.name == "drone":
                drone_cost = rt.cost_per_unit

        caps = {
            "ranger_foot_team": int(ranger_budget / ranger_cost),
            "drone": int(drone_budget / drone_cost),
        }
        # Zero out other types
        for rt in resource_types:
            if rt.name not in caps:
                caps[rt.name] = 0

        alloc = make_empty_allocation(park, rs, resource_types, total_budget, caps, kernel)
        alloc = greedy_allocate_fast(alloc, verbose=False)

        results["ppi"].append(ppi(alloc))
        results["cost"].append(alloc.total_cost())
        results["disruption"].append(disruption_score(alloc))

    results["ppi"] = np.array(results["ppi"])
    results["disruption"] = np.array(results["disruption"])
    return results


def run_diminishing_returns_budget(
    park: Park, resource_types: list, kernel: str,
    budgets: NDArray | None = None, seed: int = 42,
) -> dict[str, Any]:
    """Sweep total budget and plot PPI."""
    rs = build_risk_surface(park, season="dry")
    if budgets is None:
        budgets = np.array([50, 100, 150, 200, 300, 400, 500, 700, 1000])

    ppis = []
    for b in budgets:
        alloc = make_empty_allocation(park, rs, resource_types, float(b), None, kernel)
        alloc = greedy_allocate_fast(alloc, verbose=False)
        ppis.append(ppi(alloc))

    return {"budgets": budgets, "ppi": np.array(ppis)}


def run_diminishing_returns_rangers(
    park: Park, resource_types: list, kernel: str,
    ranger_counts: NDArray | None = None, seed: int = 42,
) -> dict[str, Any]:
    """Sweep ranger count and plot PPI (rangers only)."""
    rs = build_risk_surface(park, season="dry")
    if ranger_counts is None:
        ranger_counts = np.array([50, 100, 150, 200, 250, 295, 350, 400, 500, 600])

    ranger_cost = 1.0
    for rt in resource_types:
        if rt.name == "ranger_foot_team":
            ranger_cost = rt.cost_per_unit
            break

    ppis = []
    for n in ranger_counts:
        budget = float(n) * ranger_cost
        caps = {rt.name: (int(n) if rt.name == "ranger_foot_team" else 0)
                for rt in resource_types}
        alloc = make_empty_allocation(park, rs, resource_types, budget, caps, kernel)
        alloc = greedy_allocate_fast(alloc, verbose=False)
        ppis.append(ppi(alloc))

    return {"ranger_counts": ranger_counts, "ppi": np.array(ppis)}


def run_equity_sweep(
    park: Park, resource_types: list, kernel: str, baseline_alloc: Allocation,
    p_min_values: NDArray | None = None,
) -> dict[str, Any]:
    """Sweep equity floor and report PPI vs equity."""
    if p_min_values is None:
        p_min_values = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])

    ppis = []
    equities = []

    for pm in p_min_values:
        if pm <= 0:
            ppis.append(ppi(baseline_alloc))
            eq = equity_index(baseline_alloc)
            equities.append(eq["min_mean_ratio"])
        else:
            repaired = equity_repair(baseline_alloc, p_min=pm)
            ppis.append(ppi(repaired))
            eq = equity_index(repaired)
            equities.append(eq["min_mean_ratio"])

    return {
        "p_min": p_min_values,
        "ppi": np.array(ppis),
        "equity": np.array(equities),
    }


def run_disruption_pareto(
    park: Park, resource_types: list, kernel: str,
    n_points: int = 9, max_budget: float = 400.0,
) -> dict[str, Any]:
    """Multi-objective: PPI vs Disruption by varying disruption penalty."""
    rs = build_risk_surface(park, season="dry")

    penalties = np.linspace(0, 2.0, n_points)
    ppis = []
    disruptions = []

    for penalty in penalties:
        # Modify risk to penalize high-disruption placements
        # Effective risk = risk - penalty * disruption_potential
        alloc = make_empty_allocation(park, rs, resource_types, max_budget, None, kernel)

        if penalty > 0:
            # Adjust effective risk by subtracting disruption cost
            modified_risk = rs.risk.copy()
            for k, rt in enumerate(resource_types):
                # Reduce attractiveness of high-density cells for high-disruption resources
                modified_risk -= penalty * rt.disruption_delta * park.animal_density * 0.1

            modified_risk = np.maximum(modified_risk, 0)
            # Create modified risk surface
            rs_mod = RiskSurface(
                park=rs.park, risk_st=rs.risk_st.copy(),
                risk_by_threat=rs.risk_by_threat.copy(),
                risk_by_species=rs.risk_by_species.copy(),
                risk=modified_risk,
                species_names=rs.species_names, threat_names=rs.threat_names,
                species_weights=rs.species_weights, threat_weights=rs.threat_weights,
            )
            alloc = make_empty_allocation(park, rs_mod, resource_types, max_budget, None, kernel)

        alloc = greedy_allocate_fast(alloc, verbose=False)
        # Score using original risk surface
        alloc_scored = Allocation(
            park=alloc.park, risk_surface=rs, resource_types=alloc.resource_types,
            units=alloc.units, budget_used=alloc.budget_used,
            budget_limit=alloc.budget_limit, supply_caps=alloc.supply_caps,
            kernel=alloc.kernel,
        )
        ppis.append(ppi(alloc_scored))
        disruptions.append(disruption_score(alloc_scored))

    return {
        "disruption_penalty": penalties,
        "ppi": np.array(ppis),
        "disruption": np.array(disruptions),
    }


def run_staffing_inverse(
    park: Park, resource_types: list, kernel: str,
    targets: NDArray | None = None,
) -> dict[str, Any]:
    """For each target PPI, find minimum rangers needed."""
    from .scores import staffing_required

    rs = build_risk_surface(park, season="dry")
    if targets is None:
        targets = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    # Create template allocation
    alloc_template = make_empty_allocation(park, rs, resource_types, 1000, None, kernel)

    rangers_needed = []
    for t in targets:
        n = staffing_required(alloc_template, target_ppi=t, max_rangers=800)
        rangers_needed.append(n)

    return {
        "targets": targets,
        "rangers_needed": np.array(rangers_needed),
    }


def _compute_all_scores(alloc: Allocation) -> dict[str, Any]:
    """Compute all standard scores for an allocation."""
    return {
        "ppi": ppi(alloc),
        "ppi_disaggregated": ppi_disaggregated(alloc),
        "coverage_50": coverage_fraction(alloc, threshold=0.5),
        "coverage_30": coverage_fraction(alloc, threshold=0.3),
        "equity": equity_index(alloc),
        "disruption": disruption_score(alloc),
        "robustness_det": robustness_score(alloc, randomized=False),
        "robustness_rand": robustness_score(alloc, randomized=True),
        "response_time": response_time(alloc),
        "total_cost": alloc.total_cost(),
    }
