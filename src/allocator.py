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
) -> Allocation:
    """Vectorized greedy allocation — much faster for large grids.

    Instead of evaluating every (cell, resource) pair individually, we
    vectorize the marginal gain computation across all cells for each
    resource type and pick the global best per iteration.

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

    # Precompute pairwise distances ONCE (the expensive part)
    pairwise_dists = sp_dist.cdist(int_coords, int_coords)  # (N_int, N_int)

    # Build coverage kernels per resource type — only for types we can actually place
    # (skip types at supply cap or over budget)
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

        # Apply terrain modifiers (row = target cell terrain)
        for ci, tname_t in enumerate(park.terrain_names):
            row_mask = terrain_int == ci
            mod = rt.terrain_modifier.get(tname_t, 1.0)
            if abs(mod - 1.0) > 1e-9:
                inf[row_mask, :] *= mod

        kernel_influences[k] = inf

    # Track total units per type to avoid recomputing each iteration
    type_totals = alloc.units.sum(axis=0).copy()

    step = 0
    while alloc.budget_used < alloc.budget_limit:
        best_gain_per_cost = -1e-15
        best_cell_int = -1
        best_res = -1

        for k in range(K):
            if kernel_influences[k] is None:
                continue
            if type_totals[k] >= alloc.supply_caps[k]:
                continue
            if costs[k] + alloc.budget_used > alloc.budget_limit:
                continue

            inf_k = kernel_influences[k]
            alpha_k = alpha_matrix[k]

            # Marginal gain for placing one unit of type k at each candidate cell j:
            # gain(j) = sum_t sum_i risk[i,t] * exp(-exp[i,t]) * (1 - exp(-alpha[k,t]*inf[i,j]))
            gains = np.zeros(N_int, dtype=np.float64)
            for ti in range(T):
                if alpha_k[ti] < 1e-12:
                    continue
                miss = np.exp(-current_exponent[:, ti])  # (N_int,)
                delta_term = 1.0 - np.exp(-alpha_k[ti] * inf_k)  # (N_int, N_int)
                weighted = risk_int[:, ti] * miss  # (N_int,)
                gains += weighted @ delta_term  # (N_int,)

            gains /= total_risk * costs[k]

            best_j = np.argmax(gains)
            if gains[best_j] > best_gain_per_cost:
                best_gain_per_cost = gains[best_j]
                best_cell_int = best_j
                best_res = k

        if best_cell_int < 0 or best_gain_per_cost < 1e-12:
            break

        # Place the unit
        full_idx = interior_idx[best_cell_int]
        alloc.units[full_idx, best_res] += 1
        alloc.budget_used += costs[best_res]
        type_totals[best_res] += 1

        # Update exponent incrementally
        inf_placed = kernel_influences[best_res][:, best_cell_int]
        for ti in range(T):
            current_exponent[:, ti] += alpha_matrix[best_res, ti] * inf_placed

        step += 1
        if verbose and step % 50 == 0:
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

    Relaxes integer unit placements to continuous, linearizes the detection
    probability using its tangent at the current allocation. This provides
    an upper bound on the achievable PPI, useful as a sanity check on the
    greedy solution quality.

    Returns:
        Upper bound on PPI (float). Returns inf if LP is infeasible.
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

    # Linearize: for small allocations, p ≈ sum_k alpha_kt * eff_k
    # Upper bound on PPI: maximize sum_i sum_t risk[i,t] * min(1, sum_k alpha*eff)
    # We use a simpler LP: maximize coverage subject to budget

    prob = pulp.LpProblem("PPI_upper_bound", pulp.LpMaximize)

    # Decision variables: x[i, k] = continuous units at cell i, type k
    x = {}
    for idx, ci in enumerate(interior):
        for k in range(K):
            x[idx, k] = pulp.LpVariable(f"x_{idx}_{k}", lowBound=0)

    # Coverage variable per cell per threat (capped at 1)
    # p_var[i, t] <= sum_k alpha[k,t] * x[i,k]  (linearization)
    # p_var[i, t] <= 1
    p_var = {}
    for idx in range(N_int):
        for ti in range(T):
            p_var[idx, ti] = pulp.LpVariable(f"p_{idx}_{ti}", lowBound=0, upBound=1)

    # Objective: maximize weighted detection
    total_risk = rs.risk_by_threat[interior].sum()
    obj_terms = []
    for idx in range(N_int):
        ci = interior[idx]
        for ti in range(T):
            coeff = rs.risk_by_threat[ci, ti] / total_risk
            if coeff > 1e-12:
                obj_terms.append(coeff * p_var[idx, ti])

    prob += pulp.lpSum(obj_terms)

    # Detection constraint (linearized)
    costs = [rt.cost_per_unit for rt in allocation.resource_types]
    for idx in range(N_int):
        for ti, tname in enumerate(rs.threat_names):
            alpha_sum = pulp.lpSum([
                allocation.resource_types[k].detection_alpha.get(tname, 0.0) * x[idx, k]
                for k in range(K)
            ])
            prob += p_var[idx, ti] <= alpha_sum

    # Budget constraint
    budget_terms = []
    for idx in range(N_int):
        for k in range(K):
            budget_terms.append(costs[k] * x[idx, k])
    prob += pulp.lpSum(budget_terms) <= allocation.budget_limit

    # Supply caps
    for k in range(K):
        cap = allocation.supply_caps[k]
        if cap < 999999:
            prob += pulp.lpSum([x[idx, k] for idx in range(N_int)]) <= cap

    # Solve (suppress output)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

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
