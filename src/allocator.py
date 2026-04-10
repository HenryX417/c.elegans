"""Resource allocation optimizer.

Greedy submodular maximization of the Park Protection Index (PPI),
plus an LP relaxation upper bound via PuLP/CBC.

The greedy algorithm places one resource unit at a time in the cell/type
pair that maximizes marginal PPI gain per unit cost. Because PPI is a
weighted sum of (1 - exp(-...)) terms — a monotone submodular function
of the allocation — the greedy algorithm achieves a (1 - 1/e) ≈ 63%
approximation guarantee (Nemhauser, Wolsey, Fisher 1978).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance as sp_dist

from .park import Park
from .resources import (
    Allocation,
    ResourceType,
    compute_detection_prob,
    compute_effective_units,
)
from .risk import RiskSurface


def compute_ppi(allocation: Allocation) -> float:
    """Compute the Park Protection Index.

    PPI = sum_i sum_t risk[i,t] * p[i,t] / sum_i sum_t risk[i,t]

    PPI ∈ [0, 1], where 1 means every unit of risk is fully detected.
    """
    rs = allocation.risk_surface
    effective = compute_effective_units(
        allocation.park, allocation.resource_types,
        allocation.units, allocation.kernel,
    )
    p = compute_detection_prob(effective, allocation.resource_types, rs.threat_names)

    # risk_by_threat is (N, T), p is (N, T)
    numerator = (rs.risk_by_threat * p).sum()
    denominator = rs.risk_by_threat.sum()
    if denominator < 1e-12:
        return 0.0
    return float(numerator / denominator)


def _marginal_gain_fast(
    allocation: Allocation,
    cell_idx: int,
    res_idx: int,
    current_effective: NDArray[np.float64],
    current_numerator: float,
    total_risk: float,
) -> float:
    """Fast marginal PPI gain from placing one unit of res_idx in cell_idx.

    Instead of recomputing the full PPI, we compute only the change in the
    numerator caused by the added unit's influence on nearby cells.
    """
    park = allocation.park
    rt = allocation.resource_types[res_idx]
    rs = allocation.risk_surface
    radius = rt.coverage_radius_km

    # Cells within influence radius
    center = park.cell_centers[cell_idx]
    dists = np.linalg.norm(park.cell_centers - center, axis=1)

    if allocation.kernel == "gaussian":
        mask = dists < 3 * radius  # truncate at 3 sigma
        influence = np.exp(-0.5 * (dists[mask] / radius) ** 2)
    else:
        mask = dists <= radius
        influence = np.ones(mask.sum(), dtype=np.float64)

    if mask.sum() == 0:
        return 0.0

    # Terrain modifier
    affected_cells = np.where(mask)[0]
    terrain_mods = np.ones(len(affected_cells), dtype=np.float64)
    for ci, tname in enumerate(park.terrain_names):
        t_mask = park.terrain[affected_cells] == ci
        terrain_mods[t_mask] = rt.terrain_modifier.get(tname, 1.0)

    delta_effective = influence * terrain_mods  # (affected,)

    # For each threat, compute change in detection probability
    T = len(rs.threat_names)
    gain = 0.0
    for ti, tname in enumerate(rs.threat_names):
        alpha_kt = rt.detection_alpha.get(tname, 0.0)
        if alpha_kt < 1e-12:
            continue
        # Old exponent for these cells
        old_exp = current_effective[affected_cells, :].copy()
        # Build alpha vector for all resource types
        alpha_vec = np.array([
            rr.detection_alpha.get(tname, 0.0) for rr in allocation.resource_types
        ])
        old_sum = old_exp @ alpha_vec  # (affected,) — total alpha*effective
        old_p = 1.0 - np.exp(-old_sum)
        new_sum = old_sum + alpha_kt * delta_effective
        new_p = 1.0 - np.exp(-new_sum)
        delta_p = new_p - old_p
        # Marginal PPI gain from this threat
        risk_t = rs.risk_by_threat[affected_cells, ti]
        gain += (risk_t * delta_p).sum()

    return gain / total_risk if total_risk > 1e-12 else 0.0


def greedy_allocate(
    allocation: Allocation,
    verbose: bool = False,
) -> Allocation:
    """Greedy submodular maximization of PPI.

    At each step, place one unit of one resource type in the cell that
    maximizes marginal PPI gain per unit cost, subject to budget and
    supply constraints.

    # submodular: greedy gives (1-1/e) ≈ 0.632 approximation bound
    # see Nemhauser, Wolsey, Fisher (1978)

    Args:
        allocation: Starting allocation (typically empty).
        verbose: Print progress every 50 steps.

    Returns:
        Updated allocation with resources placed.
    """
    alloc = allocation.copy()
    park = alloc.park
    rs = alloc.risk_surface
    interior = park.inside_mask

    total_risk = rs.risk_by_threat.sum()
    if total_risk < 1e-12:
        return alloc

    # Precompute current effective units
    current_effective = compute_effective_units(
        park, alloc.resource_types, alloc.units, alloc.kernel,
    )

    current_p = compute_detection_prob(current_effective, alloc.resource_types, rs.threat_names)
    current_numerator = (rs.risk_by_threat * current_p).sum()

    K = alloc.K
    costs = np.array([rt.cost_per_unit for rt in alloc.resource_types])
    interior_cells = np.where(interior)[0]

    step = 0
    while alloc.budget_used < alloc.budget_limit:
        best_gain_per_cost = -1.0
        best_cell = -1
        best_res = -1

        for k in range(K):
            if alloc.total_units()[k] >= alloc.supply_caps[k]:
                continue
            if costs[k] + alloc.budget_used > alloc.budget_limit:
                continue

            # Evaluate marginal gain for each interior cell
            for ci in interior_cells:
                gain = _marginal_gain_fast(
                    alloc, ci, k, current_effective,
                    current_numerator, total_risk,
                )
                gpc = gain / costs[k]
                if gpc > best_gain_per_cost:
                    best_gain_per_cost = gpc
                    best_cell = ci
                    best_res = k

        if best_cell < 0 or best_gain_per_cost < 1e-12:
            break  # no beneficial placement possible

        # Place the unit
        alloc.units[best_cell, best_res] += 1
        alloc.budget_used += costs[best_res]

        # Update effective units incrementally
        current_effective = compute_effective_units(
            park, alloc.resource_types, alloc.units, alloc.kernel,
        )
        current_p = compute_detection_prob(current_effective, alloc.resource_types, rs.threat_names)
        current_numerator = (rs.risk_by_threat * current_p).sum()

        step += 1
        if verbose and step % 50 == 0:
            ppi = current_numerator / total_risk
            print(f"  Step {step}: PPI={ppi:.4f}, budget={alloc.budget_used:.1f}/{alloc.budget_limit:.1f}")

    if verbose:
        ppi = current_numerator / total_risk
        print(f"  Done: {step} placements, PPI={ppi:.4f}, budget={alloc.budget_used:.1f}")

    return alloc


def greedy_allocate_fast(
    allocation: Allocation,
    verbose: bool = False,
    batch_size: int = 10,
) -> Allocation:
    """Batch-greedy submodular maximization of PPI.

    Uses vectorized marginal gain computation with batch placement:
    each iteration, compute gains for all (cell, type) pairs, then place
    the top batch_size units simultaneously before recomputing gains.

    For batch_size=1, this is exact greedy (slow). Larger batches trade
    a small amount of optimality for major speedup. The (1-1/e) guarantee
    still holds for batch_size=1; for larger batches it degrades gracefully.

    # submodular: greedy gives (1-1/e) ≈ 0.632 approximation bound
    # see Nemhauser, Wolsey, Fisher (1978)
    """
    alloc = allocation.copy()
    park = alloc.park
    rs = alloc.risk_surface
    interior = park.inside_mask
    interior_idx = np.where(interior)[0]
    N_int = len(interior_idx)

    total_risk = rs.risk_by_threat[interior].sum()
    if total_risk < 1e-12:
        return alloc

    K = alloc.K
    T = len(rs.threat_names)
    costs = np.array([rt.cost_per_unit for rt in alloc.resource_types])

    int_coords = park.cell_centers[interior_idx]

    # Build alpha matrix (K, T)
    alpha_matrix = np.zeros((K, T), dtype=np.float64)
    for k, rt in enumerate(alloc.resource_types):
        for ti, tname in enumerate(rs.threat_names):
            alpha_matrix[k, ti] = rt.detection_alpha.get(tname, 0.0)

    # Current exponent accumulator: (N_int, T)
    current_effective_int = compute_effective_units(
        park, alloc.resource_types, alloc.units, alloc.kernel,
    )[interior_idx]
    current_exponent = current_effective_int @ alpha_matrix

    risk_int = rs.risk_by_threat[interior_idx]  # (N_int, T)

    # Precompute pairwise distances ONCE
    pairwise_dists = sp_dist.cdist(int_coords, int_coords)  # (N_int, N_int)

    # Build coverage kernels only for placeable resource types
    kernel_influences: list[NDArray | None] = [None] * K
    terrain_int = park.terrain[interior_idx]

    for k, rt in enumerate(alloc.resource_types):
        if alloc.supply_caps[k] <= 0:
            continue
        if costs[k] > alloc.budget_limit:
            continue

        radius = rt.coverage_radius_km
        if alloc.kernel == "gaussian":
            inf = np.exp(-0.5 * (pairwise_dists / radius) ** 2)
            inf[pairwise_dists > 3 * radius] = 0.0
        else:
            inf = (pairwise_dists <= radius).astype(np.float64)

        for ci, tname_t in enumerate(park.terrain_names):
            row_mask = terrain_int == ci
            mod = rt.terrain_modifier.get(tname_t, 1.0)
            if abs(mod - 1.0) > 1e-9:
                inf[row_mask, :] *= mod

        kernel_influences[k] = inf

    type_totals = alloc.units.sum(axis=0).copy()
    step = 0

    while alloc.budget_used < alloc.budget_limit:
        # Compute marginal gains for ALL (cell, resource_type) pairs
        all_gains = np.full((N_int, K), -np.inf, dtype=np.float64)

        for k in range(K):
            if kernel_influences[k] is None:
                continue
            if type_totals[k] >= alloc.supply_caps[k]:
                continue
            if costs[k] + alloc.budget_used > alloc.budget_limit:
                continue

            inf_k = kernel_influences[k]
            alpha_k = alpha_matrix[k]

            gains = np.zeros(N_int, dtype=np.float64)
            for ti in range(T):
                if alpha_k[ti] < 1e-12:
                    continue
                miss = np.exp(-current_exponent[:, ti])
                delta_term = 1.0 - np.exp(-alpha_k[ti] * inf_k)
                weighted = risk_int[:, ti] * miss
                gains += weighted @ delta_term

            all_gains[:, k] = gains / (total_risk * costs[k])

        # Find global best
        flat_best = np.argmax(all_gains)
        best_cell_int, best_res = divmod(int(flat_best), K)

        if all_gains[best_cell_int, best_res] < 1e-12:
            break

        # Batch placement: take top-B distinct cells for the best resource type
        # (placing in the same type avoids cross-type interference issues)
        remaining_budget = alloc.budget_limit - alloc.budget_used
        remaining_supply = int(alloc.supply_caps[best_res] - type_totals[best_res])
        max_placeable = min(
            batch_size,
            remaining_supply,
            int(remaining_budget / costs[best_res]),
        )

        if max_placeable <= 0:
            break

        k = best_res
        gains_k = all_gains[:, k].copy()
        placed_this_batch = 0

        for _ in range(max_placeable):
            j = int(np.argmax(gains_k))
            if gains_k[j] < 1e-12:
                break

            full_idx = interior_idx[j]
            alloc.units[full_idx, k] += 1
            alloc.budget_used += costs[k]
            type_totals[k] += 1
            placed_this_batch += 1

            # Update exponent from this placement
            inf_placed = kernel_influences[k][:, j]
            for ti in range(T):
                current_exponent[:, ti] += alpha_matrix[k, ti] * inf_placed

            # Suppress this cell and nearby cells from being re-picked in this batch
            # (avoid stacking too many units in overlapping coverage)
            gains_k[j] = -np.inf
            # Also reduce gains of nearby cells (within one radius)
            radius = alloc.resource_types[k].coverage_radius_km
            nearby = pairwise_dists[j] < radius
            gains_k[nearby] *= 0.5  # dampen, don't eliminate

        step += placed_this_batch

        if verbose and step % 50 < placed_this_batch:
            p = 1.0 - np.exp(-current_exponent)
            ppi_val = (risk_int * p).sum() / total_risk
            print(f"  Step {step}: PPI={ppi_val:.4f}, budget={alloc.budget_used:.1f}/{alloc.budget_limit:.1f}")

    if verbose:
        p = 1.0 - np.exp(-current_exponent)
        ppi_val = (risk_int * p).sum() / total_risk
        print(f"  Done: {step} placements, PPI={ppi_val:.4f}, budget={alloc.budget_used:.1f}")

    return alloc


def lp_upper_bound(allocation: Allocation) -> float:
    """LP relaxation upper bound on PPI via PuLP/CBC.

    Relaxes the detection model: for each cell i and threat t, define
    p[i,t] = min(1, sum_k alpha[k,t] * eff[k,i]) as a linear upper
    bound on the true 1-exp(-...) detection. Because the linear function
    dominates the concave exponential everywhere, the LP objective is an
    upper bound on the true PPI.

    We model the coverage kernel explicitly:
      eff[i,k] = sum_j kernel[i,j] * x[j,k]
    so that the LP accounts for resource spillover to nearby cells.

    Returns:
        Upper bound on PPI (float). Returns inf if solver fails.
    """
    try:
        import pulp
    except ImportError:
        return float("inf")

    park = allocation.park
    rs = allocation.risk_surface
    interior = np.where(park.inside_mask)[0]
    N_int = len(interior)
    K = allocation.K
    T = len(rs.threat_names)

    if N_int == 0:
        return 0.0

    total_risk = rs.risk_by_threat[interior].sum()
    if total_risk < 1e-12:
        return 0.0

    # Precompute kernel matrices (same as greedy, but only for interior)
    int_coords = park.cell_centers[interior]
    pairwise = sp_dist.cdist(int_coords, int_coords)
    terrain_int = park.terrain[interior]

    kernel_mats: list[NDArray] = []
    for k, rt in enumerate(allocation.resource_types):
        radius = rt.coverage_radius_km
        if allocation.kernel == "gaussian":
            km = np.exp(-0.5 * (pairwise / radius) ** 2)
            km[pairwise > 3 * radius] = 0.0
        else:
            km = (pairwise <= radius).astype(np.float64)
        for ci, tn in enumerate(park.terrain_names):
            row_mask = terrain_int == ci
            mod = rt.terrain_modifier.get(tn, 1.0)
            if abs(mod - 1.0) > 1e-9:
                km[row_mask, :] *= mod
        kernel_mats.append(km)

    prob = pulp.LpProblem("PPI_upper_bound", pulp.LpMaximize)

    # Decision variables: x[j, k] = continuous units at interior cell j, type k
    x = {}
    for j in range(N_int):
        for k in range(K):
            x[j, k] = pulp.LpVariable(f"x_{j}_{k}", lowBound=0)

    # Detection proxy per cell per threat, capped at 1
    p_var = {}
    for i in range(N_int):
        for ti in range(T):
            p_var[i, ti] = pulp.LpVariable(f"p_{i}_{ti}", lowBound=0, upBound=1)

    # Objective: maximize sum_i sum_t risk[i,t] * p[i,t] / total_risk
    obj_terms = []
    for i in range(N_int):
        ci = interior[i]
        for ti in range(T):
            coeff = rs.risk_by_threat[ci, ti] / total_risk
            if coeff > 1e-12:
                obj_terms.append(coeff * p_var[i, ti])
    prob += pulp.lpSum(obj_terms)

    # Detection constraint: p[i,t] <= sum_k alpha[k,t] * eff[i,k]
    # where eff[i,k] = sum_j kernel_k[i,j] * x[j,k]
    costs = [rt.cost_per_unit for rt in allocation.resource_types]
    for i in range(N_int):
        for ti, tname in enumerate(rs.threat_names):
            # sum_k alpha[k,t] * sum_j kernel_k[i,j] * x[j,k]
            terms = []
            for k in range(K):
                alpha_kt = allocation.resource_types[k].detection_alpha.get(tname, 0.0)
                if alpha_kt < 1e-12:
                    continue
                km_row = kernel_mats[k][i]  # (N_int,)
                for j in range(N_int):
                    if km_row[j] > 1e-6:
                        terms.append(alpha_kt * km_row[j] * x[j, k])
            if terms:
                prob += p_var[i, ti] <= pulp.lpSum(terms)
            else:
                prob += p_var[i, ti] <= 0

    # Budget constraint
    budget_terms = []
    for j in range(N_int):
        for k in range(K):
            budget_terms.append(costs[k] * x[j, k])
    prob += pulp.lpSum(budget_terms) <= allocation.budget_limit

    # Supply caps
    for k in range(K):
        cap = allocation.supply_caps[k]
        if cap < 999999:
            prob += pulp.lpSum([x[j, k] for j in range(N_int)]) <= cap

    # Solve (suppress output, time limit 30s)
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))

    if prob.status == pulp.constants.LpStatusOptimal:
        return float(pulp.value(prob.objective))
    return float("inf")


def equity_repair(
    allocation: Allocation,
    p_min: float = 0.1,
    verbose: bool = False,
) -> Allocation:
    """Post-hoc equity repair: ensure every high-risk cell has minimum detection.

    After greedy allocation, identify cells where overall detection probability
    is below p_min and force-allocate the cheapest resource until satisfied.
    Reports the PPI cost of equity enforcement.

    Args:
        allocation: Completed greedy allocation.
        p_min: Minimum detection probability floor.
        verbose: Print repair steps.

    Returns:
        Updated allocation with equity floor enforced.
    """
    alloc = allocation.copy()
    park = alloc.park
    rs = alloc.risk_surface
    interior = np.where(park.inside_mask)[0]

    effective = compute_effective_units(
        park, alloc.resource_types, alloc.units, alloc.kernel,
    )
    p = compute_detection_prob(effective, alloc.resource_types, rs.threat_names)

    # Overall detection per cell: average across threats weighted by risk
    risk_t = rs.risk_by_threat  # (N, T)
    risk_sum = risk_t.sum(axis=1)
    p_overall = np.zeros(len(park.cell_centers))
    nonzero = risk_sum > 1e-12
    p_overall[nonzero] = (risk_t[nonzero] * p[nonzero]).sum(axis=1) / risk_sum[nonzero]

    # Find violating cells with meaningful risk
    risk_threshold = rs.risk[interior].mean() * 0.1  # only repair cells with non-trivial risk
    violating = []
    for ci in interior:
        if rs.risk[ci] > risk_threshold and p_overall[ci] < p_min:
            violating.append(ci)

    if not violating and verbose:
        print("  Equity: no violations found.")
        return alloc

    # Sort cheapest resource first
    cheapest_k = int(np.argmin([rt.cost_per_unit for rt in alloc.resource_types]))

    repairs = 0
    for ci in violating:
        while p_overall[ci] < p_min:
            alloc.units[ci, cheapest_k] += 1
            alloc.budget_used += alloc.resource_types[cheapest_k].cost_per_unit
            # Recompute for this cell
            effective = compute_effective_units(
                park, alloc.resource_types, alloc.units, alloc.kernel,
            )
            p = compute_detection_prob(effective, alloc.resource_types, rs.threat_names)
            if risk_sum[ci] > 1e-12:
                p_overall[ci] = (risk_t[ci] * p[ci]).sum() / risk_sum[ci]
            repairs += 1
            if repairs > 5000:  # safety limit
                break
        if repairs > 5000:
            break

    if verbose:
        print(f"  Equity repair: {repairs} units added to {len(violating)} cells, "
              f"extra cost={repairs * alloc.resource_types[cheapest_k].cost_per_unit:.1f}")

    return alloc
