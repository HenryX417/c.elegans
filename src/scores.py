"""Score functions for evaluating allocation quality.

Every score function takes an Allocation and returns a scalar or dict.
All are deterministic given the allocation state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .park import Park
from .resources import (
    Allocation,
    compute_detection_prob,
    compute_effective_units,
)
from .risk import RiskSurface


def _get_detection(alloc: Allocation) -> NDArray[np.float64]:
    """Helper: compute (N, T) detection probability array for allocation."""
    effective = compute_effective_units(
        alloc.park, alloc.resource_types, alloc.units, alloc.kernel,
    )
    return compute_detection_prob(effective, alloc.resource_types, alloc.risk_surface.threat_names)


def _cell_p_overall(alloc: Allocation, p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Risk-weighted average detection probability per cell.

    p_overall[i] = sum_t risk[i,t] * p[i,t] / sum_t risk[i,t]
    """
    risk_t = alloc.risk_surface.risk_by_threat  # (N, T)
    risk_sum = risk_t.sum(axis=1)
    p_overall = np.zeros(len(alloc.park.cell_centers))
    nonzero = risk_sum > 1e-12
    p_overall[nonzero] = (risk_t[nonzero] * p[nonzero]).sum(axis=1) / risk_sum[nonzero]
    return p_overall


def ppi(alloc: Allocation) -> float:
    """Overall Park Protection Index ∈ [0, 1].

    PPI = Σ_i Σ_t risk[i,t] · p[i,t] / Σ_i Σ_t risk[i,t]
    """
    rs = alloc.risk_surface
    p = _get_detection(alloc)
    num = (rs.risk_by_threat * p).sum()
    den = rs.risk_by_threat.sum()
    return float(num / den) if den > 1e-12 else 0.0


def ppi_disaggregated(alloc: Allocation) -> dict[str, dict[str, float]]:
    """PPI broken down by species and by threat.

    Returns:
        {"by_species": {name: ppi, ...}, "by_threat": {name: ppi, ...}}
    """
    rs = alloc.risk_surface
    p = _get_detection(alloc)  # (N, T)
    interior = alloc.park.inside_mask

    result: dict[str, dict[str, float]] = {"by_species": {}, "by_threat": {}}

    # By threat: PPI_t = sum_i risk_by_threat[i,t] * p[i,t] / sum_i risk_by_threat[i,t]
    for ti, tname in enumerate(rs.threat_names):
        den = rs.risk_by_threat[interior, ti].sum()
        if den > 1e-12:
            num = (rs.risk_by_threat[interior, ti] * p[interior, ti]).sum()
            result["by_threat"][tname] = float(num / den)
        else:
            result["by_threat"][tname] = 0.0

    # By species: PPI_s = Σ_i Σ_t w_t * risk_st[i,s,t] * p[i,t]
    #                   / Σ_i Σ_t w_t * risk_st[i,s,t]
    # Weight by threat_weights so high-priority threats dominate species PPI.
    int_idx = np.where(interior)[0]
    for si, sname in enumerate(rs.species_names):
        num = 0.0
        den = 0.0
        for ti in range(len(rs.threat_names)):
            tw = float(rs.threat_weights[ti])
            risk_slice = rs.risk_st[int_idx, si, ti]      # (N_int,)
            det_slice = p[int_idx, ti]                     # (N_int,) — threat-specific
            num += tw * float((risk_slice * det_slice).sum())
            den += tw * float(risk_slice.sum())
        result["by_species"][sname] = float(num / den) if den > 1e-12 else 0.0

    return result


def coverage_fraction(alloc: Allocation, threshold: float = 0.5) -> float:
    """Fraction of interior cells with overall detection ≥ threshold."""
    p = _get_detection(alloc)
    p_overall = _cell_p_overall(alloc, p)
    interior = alloc.park.inside_mask
    n_int = interior.sum()
    if n_int == 0:
        return 0.0
    return float((p_overall[interior] >= threshold).sum() / n_int)


def equity_index(alloc: Allocation) -> dict[str, float]:
    """Equity metrics: min/mean ratio and Gini coefficient of cell detection.

    Returns:
        {"min_mean_ratio": float, "gini": float}
    """
    p = _get_detection(alloc)
    p_overall = _cell_p_overall(alloc, p)
    interior = alloc.park.inside_mask
    vals = p_overall[interior]

    if len(vals) == 0 or vals.mean() < 1e-12:
        return {"min_mean_ratio": 0.0, "gini": 1.0}

    min_mean = float(vals.min() / vals.mean())

    # Gini coefficient
    sorted_vals = np.sort(vals)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    gini = float((2 * (index * sorted_vals).sum() / (n * sorted_vals.sum())) - (n + 1) / n)
    gini = max(0.0, min(1.0, gini))

    return {"min_mean_ratio": min_mean, "gini": gini}


def disruption_score(alloc: Allocation) -> float:
    """Total wildlife disruption from deployed resources.

    disruption = Σ_i Σ_k δ[k] · units[i,k] · exp(-dist_wh[i] / 10)

    Uses waterhole proximity as a proxy for animal presence: resources
    deployed near waterholes cause more disruption because animals
    concentrate there.
    """
    wh_factor = np.exp(-alloc.park.dist_waterhole / 10.0)  # (N,)
    total = 0.0
    for k, rt in enumerate(alloc.resource_types):
        placed = alloc.units[:, k].astype(np.float64)
        total += rt.disruption_delta * (placed * wh_factor).sum()
    return float(total)


def robustness_score(
    alloc: Allocation,
    randomized: bool = False,
    n_samples: int = 20,
    seed: int = 42,
) -> float:
    """PPI under adversarial poacher best-response (Stackelberg model).

    The adversary chooses the cell maximizing risk * (1 - expected_detection).
    For deterministic patrols the adversary knows the fixed routes (pool[0]).
    For randomized patrols the adversary faces the expected detection
    over random route selections, which is strictly better for the defender
    (Stackelberg security games).

    Returns:
        Detection probability at the adversary's chosen cell.
    """
    from .temporal import build_route_pools

    rs = alloc.risk_surface
    park = alloc.park
    interior = park.inside_mask
    N = len(park.cell_centers)

    # Base detection from ALL resources (Gaussian spread = deterrent effect)
    base_p = _get_detection(alloc)

    # Build route pools (same builder as temporal.simulate)
    route_pools = build_route_pools(alloc, route_pool_size=5, seed=seed)

    risk = rs.risk.copy()
    risk[~interior] = 0.0

    # Threat-specific patrol boost (consistent with temporal.simulate)
    _patrol_boost = {
        "poaching": 0.8,
        "unauthorized_entry": 0.8,
        "human_wildlife_conflict": 0.5,
        "wildfire": 0.1,
    }
    patrol_boost = np.array(
        [_patrol_boost.get(t, 0.5) for t in rs.threat_names], dtype=np.float64
    )

    def _compute_p_overall(route_selector) -> NDArray[np.float64]:
        """Compute overall detection for all cells given route selection."""
        daily_p = base_p.copy()
        for pool in route_pools:
            if pool:
                route = route_selector(pool)
                for c in route.cells:
                    if interior[c]:
                        daily_p[c, :] = np.clip(daily_p[c, :] + patrol_boost, 0.0, 1.0)
        return _cell_p_overall(alloc, daily_p)

    if not randomized:
        # Deterministic: adversary knows base deployment AND patrol routes (pool[0]).
        # Adversary picks the cell that maximizes risk*(1-p) with full info.
        p_det = _compute_p_overall(lambda pool: pool[0])
        adversary_payoff = risk * (1.0 - p_det)
        adversary_cell = int(np.argmax(adversary_payoff))
        return float(p_det[adversary_cell])

    # Randomized: adversary faces expected detection over route realizations.
    # Adversary picks cell maximizing risk*(1-E[p]); detection = E[p] there.
    rng = np.random.default_rng(seed)
    p_samples = []
    for _ in range(n_samples):
        p_s = _compute_p_overall(lambda pool: pool[rng.integers(len(pool))])
        p_samples.append(p_s)
    p_expected = np.mean(p_samples, axis=0)
    adversary_payoff = risk * (1.0 - p_expected)
    adversary_cell = int(np.argmax(adversary_payoff))
    return float(p_expected[adversary_cell])


def cost_efficiency(alloc: Allocation) -> dict[str, float]:
    """PPI per unit cost, overall and per resource type.

    Returns:
        {"overall": ppi/total_cost, "by_type": {name: ppi_contribution/cost, ...}}
    """
    total_ppi = ppi(alloc)
    total_cost = alloc.total_cost()

    result: dict[str, float] = {}
    result["overall"] = total_ppi / total_cost if total_cost > 1e-12 else 0.0

    # Per-type: marginal contribution (approximate by removing each type)
    by_type: dict[str, float] = {}
    for k, rt in enumerate(alloc.resource_types):
        type_cost = float(alloc.units[:, k].sum() * rt.cost_per_unit)
        if type_cost < 1e-12:
            by_type[rt.name] = 0.0
            continue
        # Compute PPI without this resource type
        temp = alloc.copy()
        temp.units[:, k] = 0
        ppi_without = ppi(temp)
        contribution = total_ppi - ppi_without
        by_type[rt.name] = contribution / type_cost

    result["by_type"] = by_type
    return result


def response_time(alloc: Allocation) -> float:
    """Expected area-weighted response time from nearest ranger team.

    For each interior cell, compute travel time to the nearest cell with
    ranger_foot_team or vehicle_patrol units. Area-weighted average.

    Returns:
        Average response time in hours.
    """
    park = alloc.park
    interior = park.inside_mask
    int_idx = np.where(interior)[0]

    # Find cells with mobile resources (rangers or vehicles)
    ranger_types = []
    speeds = []
    for k, rt in enumerate(alloc.resource_types):
        if rt.name in ("ranger_foot_team", "vehicle_patrol"):
            ranger_types.append(k)
            speeds.append(rt.extra.get("speed_kmh", 4.0))

    if not ranger_types:
        return float("inf")

    # Cells with any mobile resource
    mobile_mask = np.zeros(len(park.cell_centers), dtype=bool)
    cell_speed = np.zeros(len(park.cell_centers))
    for k, spd in zip(ranger_types, speeds):
        has_units = alloc.units[:, k] > 0
        mobile_mask |= has_units
        cell_speed[has_units] = max(cell_speed[has_units].max() if has_units.any() else 0, spd)

    mobile_cells = np.where(mobile_mask)[0]
    if len(mobile_cells) == 0:
        return float("inf")

    # Average speed of responding units
    avg_speed = np.mean([speeds[i] for i in range(len(speeds))])

    # Distance from each interior cell to nearest mobile cell
    from scipy.spatial import distance as sp_dist
    dists = sp_dist.cdist(park.cell_centers[int_idx], park.cell_centers[mobile_cells])
    min_dist = dists.min(axis=1)  # km

    # Travel time = distance / speed
    times = min_dist / avg_speed  # hours

    # Area-weighted average (all cells equal area)
    return float(times.mean())


def staffing_required(
    alloc: Allocation,
    target_ppi: float,
    max_rangers: int = 1000,
) -> int:
    """Inverse-solve: minimum ranger teams to achieve target PPI.

    Binary search over number of ranger_foot_team units, re-running greedy
    allocation each time. This is approximate but fast.

    Args:
        alloc: Template allocation (used for park, risk surface, other resources).
        target_ppi: Desired PPI level.
        max_rangers: Upper bound on search.

    Returns:
        Minimum number of ranger teams needed, or max_rangers if not achievable.
    """
    from .resources import make_empty_allocation

    # Find ranger type index
    ranger_k = -1
    for k, rt in enumerate(alloc.resource_types):
        if rt.name == "ranger_foot_team":
            ranger_k = k
            break

    if ranger_k < 0:
        return max_rangers

    lo, hi = 0, max_rangers

    while lo < hi:
        mid = (lo + hi) // 2

        # Create allocation with mid rangers as budget
        caps = {rt.name: (mid if rt.name == "ranger_foot_team" else 0)
                for rt in alloc.resource_types}
        budget = mid * alloc.resource_types[ranger_k].cost_per_unit

        temp = make_empty_allocation(
            alloc.park, alloc.risk_surface, alloc.resource_types,
            budget_limit=budget, supply_caps=caps, kernel=alloc.kernel,
        )
        from .allocator import greedy_allocate_fast
        temp = greedy_allocate_fast(temp)
        current_ppi = ppi(temp)

        if current_ppi >= target_ppi:
            hi = mid
        else:
            lo = mid + 1

    return lo
