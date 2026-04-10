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


def lp_upper_bound(allocation: Allocation, max_cells: int = 800) -> float:
    """LP relaxation upper bound on PPI via scipy/HiGHS.

    Uses piecewise-linear (tangent) upper approximation of the concave
    detection function f(z) = 1 - exp(-z). Since tangent lines of a
    concave function lie above it, constraining p <= min(tangent lines)
    gives a valid upper bound that is much tighter than the naive p <= z.

    Models coverage kernel spillover explicitly:
      z[i,t] = sum_j sum_k alpha[k,t] * kernel[i,j] * x[j,k]

    For large grids, subsamples the highest-risk cells.

    Returns:
        Upper bound on PPI (float). Returns inf if solver fails.
    """
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except ImportError:
        return float("inf")

    park = allocation.park
    rs = allocation.risk_surface
    interior = np.where(park.inside_mask)[0]
    N_full = len(interior)
    K = allocation.K
    T = len(rs.threat_names)

    if N_full == 0:
        return 0.0

    total_risk_full = rs.risk_by_threat[interior].sum()
    if total_risk_full < 1e-12:
        return 0.0

    # Subsample highest-risk cells for tractability
    if N_full > max_cells:
        cell_risks = rs.risk[interior]
        top_idx = np.argsort(cell_risks)[-max_cells:]
        lp_cells = interior[top_idx]
    else:
        lp_cells = interior

    N = len(lp_cells)
    total_risk = rs.risk_by_threat[lp_cells].sum()

    active_k = [k for k in range(K) if allocation.supply_caps[k] > 0]
    K_act = len(active_k)
    if K_act == 0:
        return 0.0

    costs = np.array([allocation.resource_types[k].cost_per_unit for k in active_k])

    # Variable layout: [x[0,0]..x[N-1,K_act-1], p[0,0]..p[N-1,T-1]]
    N_x = N * K_act
    N_p = N * T
    N_vars = N_x + N_p

    # Objective: minimize -sum(w * p)  (negate for maximization)
    c = np.zeros(N_vars)
    for i in range(N):
        for ti in range(T):
            w = rs.risk_by_threat[lp_cells[i], ti] / total_risk
            if w > 1e-12:
                c[N_x + i * T + ti] = -w

    # Build kernel matrices per active resource type
    int_coords = park.cell_centers[lp_cells]
    pairwise = sp_dist.cdist(int_coords, int_coords)
    terrain_lp = park.terrain[lp_cells]

    kernels = []
    for ak_idx, k in enumerate(active_k):
        rt = allocation.resource_types[k]
        radius = rt.coverage_radius_km
        if allocation.kernel == "gaussian":
            km = np.exp(-0.5 * (pairwise / radius) ** 2)
            km[pairwise > 3 * radius] = 0.0
        else:
            km = (pairwise <= radius).astype(np.float64)
        for ci_t, tn in enumerate(park.terrain_names):
            row_mask = terrain_lp == ci_t
            mod = rt.terrain_modifier.get(tn, 1.0)
            if abs(mod - 1.0) > 1e-9:
                km[row_mask, :] *= mod
        kernels.append(km)

    # Alpha matrix (K_act, T)
    alpha_matrix = np.zeros((K_act, T))
    for ak_idx, k in enumerate(active_k):
        rt = allocation.resource_types[k]
        for ti, tname in enumerate(rs.threat_names):
            alpha_matrix[ak_idx, ti] = rt.detection_alpha.get(tname, 0.0)

    # Tangent points for piecewise-linear envelope of 1-exp(-z)
    # f(z0) + f'(z0)(z-z0) = 1 - exp(-z0)(1+z0) + exp(-z0)*z
    tangent_z0 = np.array([0.0, 0.3, 0.7, 1.2, 2.0, 3.5])

    # Build sparse constraint matrix in COO format
    row_list = []
    col_list = []
    val_list = []
    b_list = []
    row_idx = 0

    for z0 in tangent_z0:
        fpz0 = float(np.exp(-z0))
        bval = float(1.0 - np.exp(-z0) * (1.0 + z0))

        for i in range(N):
            for ti in range(T):
                # p[i,ti] coefficient: +1
                row_list.append(row_idx)
                col_list.append(N_x + i * T + ti)
                val_list.append(1.0)

                # x coefficients: -fpz0 * alpha[k,ti] * kernel_k[i,j]
                for ak_idx in range(K_act):
                    a_kt = alpha_matrix[ak_idx, ti]
                    if a_kt < 1e-12:
                        continue
                    km_row = kernels[ak_idx][i]
                    nz_j = np.where(km_row > 1e-6)[0]
                    if len(nz_j) == 0:
                        continue
                    x_cols = (nz_j * K_act + ak_idx).tolist()
                    coeffs = (-fpz0 * a_kt * km_row[nz_j]).tolist()
                    row_list.extend([row_idx] * len(nz_j))
                    col_list.extend(x_cols)
                    val_list.extend(coeffs)

                b_list.append(bval)
                row_idx += 1

    # Budget constraint: sum costs[ak]*x[j,ak] <= budget
    for j in range(N):
        for ak_idx in range(K_act):
            row_list.append(row_idx)
            col_list.append(j * K_act + ak_idx)
            val_list.append(float(costs[ak_idx]))
    b_list.append(float(allocation.budget_limit))
    row_idx += 1

    # Supply caps
    for ak_idx, k in enumerate(active_k):
        cap = allocation.supply_caps[k]
        if cap < 999999:
            for j in range(N):
                row_list.append(row_idx)
                col_list.append(j * K_act + ak_idx)
                val_list.append(1.0)
            b_list.append(float(cap))
            row_idx += 1

    A_ub = coo_matrix(
        (val_list, (row_list, col_list)), shape=(row_idx, N_vars),
    ).tocsc()
    b_ub = np.array(b_list)

    bounds = [(0, None)] * N_x + [(0, 1.0)] * N_p

    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
        method="highs", options={"time_limit": 60.0},
    )

    if result.success:
        return float(-result.fun)
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
