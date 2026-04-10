"""Smoke tests — one fast end-to-end test per layer.

Run with: python -m pytest tests/test_smoke.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
ETOSHA_CONFIG = ROOT / "configs" / "etosha.yaml"
RESOURCES_CONFIG = ROOT / "configs" / "resources.yaml"


class TestPark:
    """Layer 1: Park grid loading."""

    def test_load_etosha(self):
        from src.park import load_park
        park = load_park(ETOSHA_CONFIG, seed=42)
        assert park.name == "Etosha National Park"
        assert park.nx > 0 and park.ny > 0
        assert park.n_cells > 0
        assert park.n_cells <= park.nx * park.ny
        # Interior cells should be a subset
        assert park.inside_mask.sum() == park.n_cells

    def test_features_non_negative(self):
        from src.park import load_park
        park = load_park(ETOSHA_CONFIG, seed=42)
        assert (park.dist_fence >= 0).all()
        assert (park.dist_road >= 0).all()
        assert (park.dist_waterhole >= 0).all()
        assert (park.dist_camp >= 0).all()
        assert (park.animal_density >= 0).all()

    def test_waterholes_generated(self):
        from src.park import load_park
        park = load_park(ETOSHA_CONFIG, seed=42)
        assert len(park.waterholes) >= 12  # at least the anchors


class TestRisk:
    """Layer 1: Risk surface."""

    def test_risk_non_negative(self):
        from src.park import load_park
        from src.risk import build_risk_surface
        park = load_park(ETOSHA_CONFIG, seed=42)
        rs = build_risk_surface(park, season="dry")
        assert (rs.risk >= 0).all(), "Risk scores must be non-negative"
        assert (rs.risk_st >= 0).all(), "Per-species-threat risk must be non-negative"

    def test_risk_zero_outside(self):
        from src.park import load_park
        from src.risk import build_risk_surface
        park = load_park(ETOSHA_CONFIG, seed=42)
        rs = build_risk_surface(park, season="dry")
        assert (rs.risk[~park.inside_mask] == 0).all()

    def test_total_risk_positive(self):
        from src.park import load_park
        from src.risk import build_risk_surface
        park = load_park(ETOSHA_CONFIG, seed=42)
        rs = build_risk_surface(park, season="dry")
        assert rs.total_risk > 0


class TestAllocator:
    """Layer 2: Resource allocation."""

    def test_greedy_respects_budget(self):
        from src.park import load_park
        from src.risk import build_risk_surface
        from src.resources import load_resource_types, make_empty_allocation
        from src.allocator import greedy_allocate_fast

        park = load_park(ETOSHA_CONFIG, seed=42)
        rs = build_risk_surface(park, season="dry")
        resource_types, kernel = load_resource_types(RESOURCES_CONFIG)

        budget = 50.0
        caps = {"ranger_foot_team": 50}
        alloc = make_empty_allocation(park, rs, resource_types, budget, caps, kernel)
        alloc = greedy_allocate_fast(alloc)

        assert alloc.budget_used <= budget + 1e-6
        assert alloc.total_units().sum() > 0  # something was placed

    def test_ppi_in_range(self):
        from src.park import load_park
        from src.risk import build_risk_surface
        from src.resources import load_resource_types, make_empty_allocation
        from src.allocator import greedy_allocate_fast, compute_ppi

        park = load_park(ETOSHA_CONFIG, seed=42)
        rs = build_risk_surface(park, season="dry")
        resource_types, kernel = load_resource_types(RESOURCES_CONFIG)

        alloc = make_empty_allocation(park, rs, resource_types, 100.0,
                                       {"ranger_foot_team": 100}, kernel)
        alloc = greedy_allocate_fast(alloc)
        ppi_val = compute_ppi(alloc)

        assert 0.0 <= ppi_val <= 1.0, f"PPI must be in [0,1], got {ppi_val}"


class TestRouting:
    """Layer 3: Patrol routing."""

    def test_solve_route_valid(self):
        from src.routing import solve_patrol_route
        # Simple test: 10 cells in a line
        cells = np.arange(10)
        centers = np.column_stack([np.arange(10) * 5.0, np.zeros(10)])
        rewards = np.ones(10)
        base = 0

        route = solve_patrol_route(cells, base, rewards, centers,
                                    speed_kmh=4.0, time_budget=8.0)
        assert route.time_used <= route.time_budget + 1e-6
        assert route.reward >= 0
        assert len(route.cells) > 0
        # All visited cells should be in the zone
        assert all(c in cells for c in route.cells)

    def test_route_respects_time_budget(self):
        from src.routing import solve_patrol_route
        # Cells far apart — should limit visits
        cells = np.arange(20)
        centers = np.column_stack([np.arange(20) * 10.0, np.zeros(20)])
        rewards = np.ones(20)

        route = solve_patrol_route(cells, 0, rewards, centers,
                                    speed_kmh=4.0, time_budget=4.0)
        assert route.time_used <= 4.0 + 1e-6


class TestTemporal:
    """Layer 3: Temporal simulation."""

    def test_simulation_runs(self):
        from src.park import load_park
        from src.risk import build_risk_surface
        from src.resources import load_resource_types, make_empty_allocation
        from src.allocator import greedy_allocate_fast
        from src.temporal import simulate

        park = load_park(ETOSHA_CONFIG, seed=42)
        rs = build_risk_surface(park, season="dry")
        resource_types, kernel = load_resource_types(RESOURCES_CONFIG)

        alloc = make_empty_allocation(park, rs, resource_types, 50.0,
                                       {"ranger_foot_team": 50}, kernel)
        alloc = greedy_allocate_fast(alloc)

        sim = simulate(alloc, n_days=5, threat_rate=3.0, seed=42)
        assert len(sim.days) == 5
        assert sim.total_threats >= 0
        assert 0 <= sim.total_intercepted <= sim.total_threats
        assert all(0 <= d.ppi <= 1.0 for d in sim.days)


class TestScores:
    """Score functions."""

    def test_all_scores_compute(self):
        from src.park import load_park
        from src.risk import build_risk_surface
        from src.resources import load_resource_types, make_empty_allocation
        from src.allocator import greedy_allocate_fast
        from src.scores import (
            ppi, ppi_disaggregated, coverage_fraction,
            equity_index, disruption_score, robustness_score,
            response_time, cost_efficiency,
        )

        park = load_park(ETOSHA_CONFIG, seed=42)
        rs = build_risk_surface(park, season="dry")
        resource_types, kernel = load_resource_types(RESOURCES_CONFIG)

        alloc = make_empty_allocation(park, rs, resource_types, 50.0,
                                       {"ranger_foot_team": 50}, kernel)
        alloc = greedy_allocate_fast(alloc)

        assert 0 <= ppi(alloc) <= 1
        disagg = ppi_disaggregated(alloc)
        assert "by_species" in disagg
        assert "by_threat" in disagg
        assert 0 <= coverage_fraction(alloc) <= 1
        eq = equity_index(alloc)
        assert "min_mean_ratio" in eq
        assert "gini" in eq
        assert disruption_score(alloc) >= 0
        assert 0 <= robustness_score(alloc) <= 1
        assert response_time(alloc) >= 0
        ce = cost_efficiency(alloc)
        assert "overall" in ce


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
