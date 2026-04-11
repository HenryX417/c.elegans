"""Resource type definitions and allocation state.

Loads resource catalog from YAML, manages placement of resource units
on the park grid, and computes detection probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from scipy.spatial import distance as sp_dist

from .park import Park
from .risk import RiskSurface

# Module-level biome effects, populated by load_resource_types().
_biome_effects: dict[str, dict[str, float]] = {}


def get_biome_effects() -> dict[str, dict[str, float]]:
    """Return the currently loaded biome effects dict."""
    return _biome_effects


@dataclass
class ResourceType:
    """A type of conservation resource (ranger team, drone, etc.).

    Attributes:
        name: Resource type identifier.
        cost_per_unit: Normalized cost to deploy one unit.
        coverage_radius_km: Effective radius of influence.
        detection_alpha: Dict mapping threat_name -> detection coefficient.
        disruption_delta: Wildlife disruption per unit.
        terrain_modifier: Dict mapping terrain_name -> effectiveness multiplier.
        extra: Any additional config fields (speed, shift_hours, etc.).
    """

    name: str
    cost_per_unit: float
    coverage_radius_km: float
    detection_alpha: dict[str, float]
    disruption_delta: float
    terrain_modifier: dict[str, float]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Allocation:
    """Resource allocation state on the park grid.

    Attributes:
        park: The Park object.
        risk_surface: The RiskSurface used for this allocation.
        resource_types: List of available ResourceType objects.
        units: (N, K) int array — number of units of each resource type in each cell.
        budget_used: Total cost spent.
        budget_limit: Maximum budget.
        supply_caps: (K,) int array — max units per resource type.
        kernel: "gaussian" or "tophat" — coverage spreading method.
    """

    park: Park
    risk_surface: RiskSurface
    resource_types: list[ResourceType]
    units: NDArray[np.int32]
    budget_used: float
    budget_limit: float
    supply_caps: NDArray[np.int32]
    kernel: str = "gaussian"

    @property
    def K(self) -> int:
        """Number of resource types."""
        return len(self.resource_types)

    def total_units(self) -> NDArray[np.int32]:
        """Total units deployed per resource type."""
        return self.units.sum(axis=0)

    def total_cost(self) -> float:
        """Total cost of all deployed units."""
        costs = np.array([rt.cost_per_unit for rt in self.resource_types])
        return float((self.units.sum(axis=0) * costs).sum())

    def copy(self) -> Allocation:
        """Deep copy of the allocation (units array is copied)."""
        return Allocation(
            park=self.park,
            risk_surface=self.risk_surface,
            resource_types=self.resource_types,
            units=self.units.copy(),
            budget_used=self.budget_used,
            budget_limit=self.budget_limit,
            supply_caps=self.supply_caps.copy(),
            kernel=self.kernel,
        )


def load_resource_types(
    config_path: str | Path,
    park_config: dict[str, Any] | None = None,
) -> tuple[list[ResourceType], str]:
    """Load resource types from the catalog YAML.

    Args:
        config_path: Path to resources.yaml.
        park_config: Optional park config with per-park resource overrides.

    Returns:
        Tuple of (list of ResourceType, kernel_type string).
    """
    global _biome_effects

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    kernel = cfg.get("coverage_kernel", "gaussian")
    _biome_effects = cfg.get("biome_effects", {})

    park_overrides = {}
    if park_config and "resource_overrides" in park_config:
        park_overrides = park_config["resource_overrides"]

    types = []
    for name, rcfg in cfg["resource_types"].items():
        # Apply park-specific overrides if any
        if name in park_overrides:
            rcfg = {**rcfg, **park_overrides[name]}

        types.append(ResourceType(
            name=name,
            cost_per_unit=rcfg["cost_per_unit"],
            coverage_radius_km=rcfg["coverage_radius_km"],
            detection_alpha=rcfg["detection_alpha"],
            disruption_delta=rcfg["disruption_delta"],
            terrain_modifier=rcfg.get("terrain_modifier", {}),
            extra={k: v for k, v in rcfg.items()
                   if k not in {"cost_per_unit", "coverage_radius_km",
                                "detection_alpha", "disruption_delta",
                                "terrain_modifier"}},
        ))

    return types, kernel


def compute_effective_units(
    park: Park,
    resource_types: list[ResourceType],
    units: NDArray[np.int32],
    kernel: str = "gaussian",
) -> NDArray[np.float64]:
    """Compute effective resource presence per cell per resource type.

    Each placed unit spreads its influence to nearby cells via a coverage
    kernel (Gaussian or top-hat), modulated by terrain.

    Args:
        park: Park object.
        resource_types: List of resource types.
        units: (N, K) int array of placed units.
        kernel: "gaussian" or "tophat".

    Returns:
        (N, K) float array of effective units per cell per type.
    """
    N = len(park.cell_centers)
    K = len(resource_types)
    effective = np.zeros((N, K), dtype=np.float64)

    for k, rt in enumerate(resource_types):
        placed = np.where(units[:, k] > 0)[0]
        if len(placed) == 0:
            continue

        radius = rt.coverage_radius_km
        placed_coords = park.cell_centers[placed]
        placed_counts = units[placed, k].astype(np.float64)

        # Pairwise distances from all cells to placed cells
        dists = sp_dist.cdist(park.cell_centers, placed_coords)  # (N, P)

        if kernel == "gaussian":
            # Gaussian: influence decays with distance
            influence = np.exp(-0.5 * (dists / radius) ** 2)  # (N, P)
        else:
            # Top-hat: uniform within radius
            influence = (dists <= radius).astype(np.float64)

        # Sum influence from all placed units of this type
        effective[:, k] = influence @ placed_counts

        # Apply terrain modifier
        for ci, tname in enumerate(park.terrain_names):
            mask = park.terrain == ci
            mod = rt.terrain_modifier.get(tname, 1.0)
            effective[mask, k] *= mod

        # Apply biome_effects (patrol_speed_mult for mobile, drone_visibility for aerial)
        if _biome_effects:
            is_mobile = rt.name in ("ranger_foot_team", "vehicle_patrol")
            is_aerial = rt.name in ("drone", "aerial_overflight")
            if is_mobile or is_aerial:
                effect_key = "patrol_speed_mult" if is_mobile else "drone_visibility"
                for ci, tname in enumerate(park.terrain_names):
                    be = _biome_effects.get(tname, {})
                    mult = be.get(effect_key, 1.0)
                    if abs(mult - 1.0) > 1e-6:
                        mask = park.terrain == ci
                        effective[mask, k] *= mult

    return effective


def compute_detection_prob(
    effective_units: NDArray[np.float64],
    resource_types: list[ResourceType],
    threat_names: list[str],
) -> NDArray[np.float64]:
    """Compute detection probability per cell per threat type.

    p[i, t] = 1 - exp(-sum_k alpha[k,t] * effective_units[k,i])

    This is the standard exponential detection model: each unit independently
    contributes to detection, with diminishing returns (submodular).

    Args:
        effective_units: (N, K) array from compute_effective_units.
        resource_types: List of resource types.
        threat_names: List of threat type names.

    Returns:
        (N, T) array of detection probabilities in [0, 1].
    """
    N = effective_units.shape[0]
    T = len(threat_names)
    K = len(resource_types)

    # Build alpha matrix (K, T)
    alpha = np.zeros((K, T), dtype=np.float64)
    for k, rt in enumerate(resource_types):
        for ti, tname in enumerate(threat_names):
            alpha[k, ti] = rt.detection_alpha.get(tname, 0.0)

    # Exponent: (N, T) = effective_units (N, K) @ alpha (K, T)
    exponent = effective_units @ alpha

    # Detection probability: 1 - exp(-exponent)
    prob = 1.0 - np.exp(-exponent)

    return prob


def make_empty_allocation(
    park: Park,
    risk_surface: RiskSurface,
    resource_types: list[ResourceType],
    budget_limit: float,
    supply_caps: dict[str, int] | None = None,
    kernel: str = "gaussian",
) -> Allocation:
    """Create an empty allocation (no units placed).

    Args:
        park: Park object.
        risk_surface: RiskSurface to optimize against.
        resource_types: Available resource types.
        budget_limit: Maximum total cost.
        supply_caps: Dict of resource_name -> max units. If None, unlimited.
        kernel: Coverage kernel type.

    Returns:
        Allocation with zero units everywhere.
    """
    N = len(park.cell_centers)
    K = len(resource_types)
    units = np.zeros((N, K), dtype=np.int32)

    caps = np.full(K, 999999, dtype=np.int32)
    if supply_caps:
        for k, rt in enumerate(resource_types):
            if rt.name in supply_caps:
                caps[k] = supply_caps[rt.name]

    return Allocation(
        park=park,
        risk_surface=risk_surface,
        resource_types=resource_types,
        units=units,
        budget_used=0.0,
        budget_limit=budget_limit,
        supply_caps=caps,
        kernel=kernel,
    )
