"""Team Orienteering Problem solver for ranger patrol routing.

Each ranger team has a zone (assigned cells) and a time budget (shift hours).
The patrol route collects risk-weighted reward subject to the time constraint.
Solved with nearest-neighbor initialization + 2-opt + Or-opt local search.

This is the same algorithmic structure we used for HiMCM 2025 (Team #17401).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class PatrolRoute:
    """A single patrol route for one ranger team.

    Attributes:
        cells: Ordered list of cell indices visited.
        reward: Total risk-weighted reward collected.
        time_used: Total travel time in hours.
        time_budget: Maximum allowed time.
    """

    cells: list[int]
    reward: float
    time_used: float
    time_budget: float


def _travel_time(
    cell_a: int,
    cell_b: int,
    cell_centers: NDArray,
    speed_kmh: float,
) -> float:
    """Euclidean travel time between two cells in hours."""
    dist = np.linalg.norm(cell_centers[cell_a] - cell_centers[cell_b])
    return dist / speed_kmh


def _route_time(
    route: list[int],
    base_cell: int,
    cell_centers: NDArray,
    speed: float,
) -> float:
    """Total time for a route starting and ending at base_cell."""
    if not route:
        return 0.0
    t = _travel_time(base_cell, route[0], cell_centers, speed)
    for i in range(len(route) - 1):
        t += _travel_time(route[i], route[i + 1], cell_centers, speed)
    t += _travel_time(route[-1], base_cell, cell_centers, speed)
    return t


def _route_reward(route: list[int], rewards: NDArray) -> float:
    """Total reward from visiting cells in the route."""
    if not route:
        return 0.0
    return float(rewards[route].sum())


def nearest_neighbor_init(
    zone_cells: NDArray[np.intp],
    base_cell: int,
    rewards: NDArray[np.float64],
    cell_centers: NDArray[np.float64],
    speed_kmh: float,
    time_budget: float,
) -> list[int]:
    """Nearest-neighbor greedy construction for team orienteering.

    Start at base, greedily add the nearest unvisited cell with positive
    reward that doesn't violate the time budget (including return to base).
    """
    route: list[int] = []
    visited = set()
    current = base_cell
    time_used = 0.0

    candidates = [c for c in zone_cells if rewards[c] > 0 and c != base_cell]

    while candidates:
        best_cell = -1
        best_ratio = -1.0
        best_time = 0.0

        for c in candidates:
            if c in visited:
                continue
            t_to = _travel_time(current, c, cell_centers, speed_kmh)
            t_return = _travel_time(c, base_cell, cell_centers, speed_kmh)
            total_if_added = time_used + t_to + t_return
            if total_if_added > time_budget:
                continue
            # Ratio: reward per time
            ratio = rewards[c] / (t_to + 1e-6)
            if ratio > best_ratio:
                best_ratio = ratio
                best_cell = c
                best_time = t_to

        if best_cell < 0:
            break

        route.append(best_cell)
        visited.add(best_cell)
        time_used += best_time
        current = best_cell
        candidates = [c for c in candidates if c not in visited]

    return route


def two_opt(
    route: list[int],
    base_cell: int,
    rewards: NDArray,
    cell_centers: NDArray,
    speed: float,
    time_budget: float,
) -> list[int]:
    """2-opt local search: reverse subsequences to shorten travel time.

    Frees up time budget for potentially inserting additional cells.
    """
    if len(route) < 3:
        return route

    improved = True
    best = list(route)
    best_time = _route_time(best, base_cell, cell_centers, speed)

    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                # Reverse segment [i+1 ... j]
                new_route = best[: i + 1] + best[i + 1 : j + 1][::-1] + best[j + 1 :]
                new_time = _route_time(new_route, base_cell, cell_centers, speed)
                if new_time < best_time - 1e-6 and new_time <= time_budget:
                    best = new_route
                    best_time = new_time
                    improved = True
                    break
            if improved:
                break

    return best


def or_opt(
    route: list[int],
    base_cell: int,
    rewards: NDArray,
    cell_centers: NDArray,
    speed: float,
    time_budget: float,
) -> list[int]:
    """Or-opt local search: relocate segments of 1-3 cells.

    Tries moving contiguous segments to other positions in the route.
    """
    if len(route) < 2:
        return route

    improved = True
    best = list(route)

    while improved:
        improved = False
        for seg_len in [1, 2, 3]:
            for i in range(len(best) - seg_len + 1):
                segment = best[i : i + seg_len]
                remaining = best[:i] + best[i + seg_len :]

                for j in range(len(remaining) + 1):
                    new_route = remaining[:j] + segment + remaining[j:]
                    new_time = _route_time(new_route, base_cell, cell_centers, speed)
                    if new_time <= time_budget:
                        old_reward = _route_reward(best, rewards)
                        new_reward = _route_reward(new_route, rewards)
                        old_time = _route_time(best, base_cell, cell_centers, speed)
                        # Accept if same reward but less time (frees budget)
                        # or more reward within budget
                        if new_reward > old_reward + 1e-6 or (
                            abs(new_reward - old_reward) < 1e-6
                            and new_time < old_time - 1e-6
                        ):
                            best = new_route
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break

    return best


def _try_insert_unvisited(
    route: list[int],
    base_cell: int,
    zone_cells: NDArray,
    rewards: NDArray,
    cell_centers: NDArray,
    speed: float,
    time_budget: float,
) -> list[int]:
    """After local search frees time, try inserting unvisited profitable cells."""
    visited = set(route)
    improved = True

    while improved:
        improved = False
        current_time = _route_time(route, base_cell, cell_centers, speed)

        best_ratio = 0.0
        best_pos = -1
        best_cell = -1

        for c in zone_cells:
            if c in visited or rewards[c] <= 0:
                continue
            # Try inserting at each position
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [c] + route[pos:]
                new_time = _route_time(new_route, base_cell, cell_centers, speed)
                if new_time <= time_budget:
                    added_time = new_time - current_time
                    ratio = rewards[c] / (added_time + 1e-6)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_pos = pos
                        best_cell = c

        if best_cell >= 0:
            route = route[:best_pos] + [best_cell] + route[best_pos:]
            visited.add(best_cell)
            improved = True

    return route


def solve_patrol_route(
    zone_cells: NDArray[np.intp],
    base_cell: int,
    rewards: NDArray[np.float64],
    cell_centers: NDArray[np.float64],
    speed_kmh: float = 4.0,
    time_budget: float = 8.0,
) -> PatrolRoute:
    """Solve team orienteering problem for one ranger team's zone.

    Algorithm: nearest-neighbor init → 2-opt → or-opt → insert unvisited.

    Args:
        zone_cells: Array of cell indices in this team's zone.
        base_cell: Starting/ending cell index (camp or base).
        rewards: (N,) array of per-cell reward (typically risk scores).
        cell_centers: (N, 2) array of cell coordinates.
        speed_kmh: Travel speed.
        time_budget: Maximum patrol duration in hours.

    Returns:
        PatrolRoute with optimized cell visit sequence.
    """
    route = nearest_neighbor_init(
        zone_cells, base_cell, rewards, cell_centers, speed_kmh, time_budget,
    )
    route = two_opt(route, base_cell, rewards, cell_centers, speed_kmh, time_budget)
    route = or_opt(route, base_cell, rewards, cell_centers, speed_kmh, time_budget)
    route = _try_insert_unvisited(
        route, base_cell, zone_cells, rewards, cell_centers, speed_kmh, time_budget,
    )

    time_used = _route_time(route, base_cell, cell_centers, speed_kmh)
    reward = _route_reward(route, rewards)

    return PatrolRoute(
        cells=route,
        reward=reward,
        time_used=time_used,
        time_budget=time_budget,
    )


def assign_zones(
    park_interior: NDArray[np.intp],
    base_cells: list[int],
    cell_centers: NDArray[np.float64],
) -> list[NDArray[np.intp]]:
    """Assign interior cells to nearest base (Voronoi partitioning).

    Args:
        park_interior: Array of interior cell indices.
        base_cells: List of base cell indices (one per team).
        cell_centers: (N, 2) coordinate array.

    Returns:
        List of arrays, one per base, containing assigned cell indices.
    """
    base_coords = cell_centers[base_cells]  # (B, 2)
    int_coords = cell_centers[park_interior]  # (M, 2)

    from scipy.spatial import distance as sp_dist
    dists = sp_dist.cdist(int_coords, base_coords)  # (M, B)
    assignments = np.argmin(dists, axis=1)  # (M,)

    zones = []
    for b in range(len(base_cells)):
        zone_mask = assignments == b
        zones.append(park_interior[zone_mask])

    return zones


def generate_route_pool(
    zone_cells: NDArray[np.intp],
    base_cell: int,
    rewards: NDArray[np.float64],
    cell_centers: NDArray[np.float64],
    speed_kmh: float = 4.0,
    time_budget: float = 8.0,
    pool_size: int = 5,
    seed: int = 42,
) -> list[PatrolRoute]:
    """Generate a pool of diverse high-quality routes for randomized patrols.

    Creates route variants by perturbing rewards and re-solving, producing
    a diverse set for the mini-Stackelberg randomization mechanism.

    Args:
        zone_cells: Cell indices in this team's zone.
        base_cell: Base cell index.
        rewards: (N,) per-cell reward array.
        cell_centers: (N, 2) coordinates.
        speed_kmh: Travel speed.
        time_budget: Shift duration in hours.
        pool_size: Number of route variants to generate.
        seed: Random seed.

    Returns:
        List of PatrolRoute objects (top-K by reward).
    """
    rng = np.random.default_rng(seed)
    routes = []

    # Base route (no perturbation)
    base_route = solve_patrol_route(
        zone_cells, base_cell, rewards, cell_centers, speed_kmh, time_budget,
    )
    routes.append(base_route)

    # Generate variants with perturbed rewards
    for _ in range(pool_size * 2):
        noise = rng.lognormal(0, 0.3, size=len(rewards))
        perturbed = rewards * noise
        route = solve_patrol_route(
            zone_cells, base_cell, perturbed, cell_centers, speed_kmh, time_budget,
        )
        routes.append(route)

    # De-duplicate by cell set and keep top-K by reward
    seen: set[frozenset[int]] = set()
    unique: list[PatrolRoute] = []
    for r in sorted(routes, key=lambda r: r.reward, reverse=True):
        key = frozenset(r.cells)
        if key not in seen:
            seen.add(key)
            unique.append(r)
        if len(unique) >= pool_size:
            break

    return unique
