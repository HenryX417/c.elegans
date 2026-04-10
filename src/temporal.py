"""Temporal simulation: 30-day stochastic model with randomized patrols.

Each day:
1. Threats arrive stochastically in cells with rates proportional to risk[i, t].
2. Rangers execute a route drawn randomly from a pool of precomputed routes.
3. A threat is "intercepted" if its cell is visited that day or covered by
   a sensor/camera/drone with success probability p[i, t].
4. Track PPI, intercepted threats, and per-species protection over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .park import Park
from .resources import (
    Allocation,
    compute_detection_prob,
    compute_effective_units,
)
from .risk import RiskSurface
from .routing import (
    PatrolRoute,
    assign_zones,
    generate_route_pool,
)


@dataclass
class DayResult:
    """Results for a single simulated day."""

    day: int
    threats_arrived: int
    threats_intercepted: int
    ppi: float
    ppi_by_species: dict[str, float]
    ppi_by_threat: dict[str, float]


@dataclass
class SimulationResult:
    """Full temporal simulation results.

    Attributes:
        days: List of DayResult objects.
        total_threats: Total threats over all days.
        total_intercepted: Total interceptions over all days.
        mean_ppi: Average daily PPI.
        ppi_timeseries: (D,) array of daily PPI values.
    """

    days: list[DayResult]
    total_threats: int = 0
    total_intercepted: int = 0
    mean_ppi: float = 0.0
    ppi_timeseries: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        if self.days:
            self.total_threats = sum(d.threats_arrived for d in self.days)
            self.total_intercepted = sum(d.threats_intercepted for d in self.days)
            self.ppi_timeseries = np.array([d.ppi for d in self.days])
            self.mean_ppi = float(self.ppi_timeseries.mean())


def _generate_threats(
    risk_surface: RiskSurface,
    interior: NDArray[np.intp],
    threat_rate: float,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Generate stochastic threat events for one day.

    Each cell-threat pair produces a Poisson number of events with rate
    proportional to its risk score. The threat_rate scales the overall
    expected number of events per day.

    Returns:
        List of (cell_index, threat_index) tuples.
    """
    rs = risk_surface
    events = []

    for ti in range(len(rs.threat_names)):
        # Rates proportional to risk_by_threat, normalized so total expected = threat_rate
        rates = rs.risk_by_threat[interior, ti].copy()
        total = rates.sum()
        if total < 1e-12:
            continue
        rates = rates * (threat_rate / len(rs.threat_names)) / total

        # Poisson draws per cell
        counts = rng.poisson(rates)
        for idx, count in enumerate(counts):
            if count > 0:
                for _ in range(count):
                    events.append((interior[idx], ti))

    return events


def simulate(
    allocation: Allocation,
    n_days: int = 30,
    threat_rate: float = 5.0,
    randomized_patrols: bool = True,
    route_pool_size: int = 5,
    seed: int = 42,
    verbose: bool = False,
) -> SimulationResult:
    """Run the temporal simulation.

    Args:
        allocation: Resource allocation to simulate.
        n_days: Number of days to simulate.
        threat_rate: Expected threats per day (scaled across all threat types).
        randomized_patrols: If True, sample from route pool each day (Stackelberg).
        route_pool_size: Number of route variants per team.
        seed: Random seed.
        verbose: Print daily summaries.

    Returns:
        SimulationResult with day-by-day tracking.
    """
    rng = np.random.default_rng(seed)
    park = allocation.park
    rs = allocation.risk_surface
    interior = park.interior_indices

    # Static detection from sensors/cameras/drones only (exclude mobile patrols)
    # Mobile resources (rangers, vehicles) contribute via patrol routes, not static detection
    mobile_types = {"ranger_foot_team", "vehicle_patrol"}
    static_units = allocation.units.copy()
    for k, rt in enumerate(allocation.resource_types):
        if rt.name in mobile_types:
            static_units[:, k] = 0
    effective = compute_effective_units(
        park, allocation.resource_types, static_units, allocation.kernel,
    )
    static_p = compute_detection_prob(effective, allocation.resource_types, rs.threat_names)

    # Identify ranger teams and generate patrol routes
    # Group units at the same cell into a single team (more realistic than
    # treating each unit individually, and avoids generating hundreds of routes)
    ranger_cells = []
    ranger_speeds = []

    for k, rt in enumerate(allocation.resource_types):
        if rt.name in mobile_types:
            placed = np.where(allocation.units[:, k] > 0)[0]
            speed = rt.extra.get("speed_kmh", 4.0)
            shift = rt.extra.get("shift_hours", 8.0)
            for cell in placed:
                ranger_cells.append(cell)
                ranger_speeds.append((speed, shift))

    # Cap patrol teams to keep routing tractable (max ~30 teams)
    max_teams = 30
    if len(ranger_cells) > max_teams:
        # Subsample: pick the max_teams cells with highest risk as team bases
        risks = rs.risk[ranger_cells]
        top_indices = np.argsort(risks)[-max_teams:]
        ranger_cells = [ranger_cells[i] for i in top_indices]
        ranger_speeds = [ranger_speeds[i] for i in top_indices]

    # Generate route pools for each ranger team
    route_pools: list[list[PatrolRoute]] = []
    if ranger_cells:
        zones = assign_zones(interior, ranger_cells, park.cell_centers)
        for team_idx, (base, zone) in enumerate(zip(ranger_cells, zones)):
            if len(zone) == 0:
                route_pools.append([])
                continue
            speed, shift = ranger_speeds[team_idx]
            pool = generate_route_pool(
                zone, base, rs.risk, park.cell_centers,
                speed_kmh=speed, time_budget=shift,
                pool_size=route_pool_size,
                seed=seed + team_idx,
            )
            route_pools.append(pool)

    days: list[DayResult] = []

    for day in range(n_days):
        # 1. Generate threats
        events = _generate_threats(rs, interior, threat_rate, rng)

        # 2. Select patrol routes for today
        visited_today: set[int] = set()
        for pool in route_pools:
            if not pool:
                continue
            if randomized_patrols:
                route = pool[rng.integers(len(pool))]
            else:
                route = pool[0]  # always use best route (deterministic)
            visited_today.update(route.cells)

        # 3. Determine interceptions
        intercepted = 0
        for cell_idx, threat_idx in events:
            # Mobile interception: cell visited today
            if cell_idx in visited_today:
                intercepted += 1
                continue

            # Static detection: sensor/camera/drone probability
            p = static_p[cell_idx, threat_idx]
            if rng.random() < p:
                intercepted += 1

        # 4. Compute daily PPI (fraction of risk covered)
        # Effective detection = static + mobile visit indicator
        daily_p = static_p.copy()
        for c in visited_today:
            if park.inside_mask[c]:
                daily_p[c, :] = np.clip(daily_p[c, :] + 0.8, 0.0, 1.0)  # strong boost from physical presence

        risk_t = rs.risk_by_threat[interior]
        total_risk = risk_t.sum()
        if total_risk > 1e-12:
            daily_ppi = float((risk_t * daily_p[interior]).sum() / total_risk)
        else:
            daily_ppi = 0.0

        # Per-species PPI
        ppi_by_species: dict[str, float] = {}
        for si, sname in enumerate(rs.species_names):
            den = rs.risk_st[interior, si, :].sum()
            if den > 1e-12:
                num = (rs.risk_st[interior, si, :] * daily_p[interior]).sum()
                ppi_by_species[sname] = float(num / den)
            else:
                ppi_by_species[sname] = 0.0

        # Per-threat PPI
        ppi_by_threat: dict[str, float] = {}
        for ti, tname in enumerate(rs.threat_names):
            den = risk_t[:, ti].sum()
            if den > 1e-12:
                num = (risk_t[:, ti] * daily_p[interior, ti]).sum()
                ppi_by_threat[tname] = float(num / den)
            else:
                ppi_by_threat[tname] = 0.0

        day_result = DayResult(
            day=day + 1,
            threats_arrived=len(events),
            threats_intercepted=intercepted,
            ppi=daily_ppi,
            ppi_by_species=ppi_by_species,
            ppi_by_threat=ppi_by_threat,
        )
        days.append(day_result)

        if verbose:
            print(f"  Day {day+1}: threats={len(events)}, intercepted={intercepted}, PPI={daily_ppi:.3f}")

    return SimulationResult(days=days)
