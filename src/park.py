"""Park grid construction and spatial features.

Loads a park YAML config, builds a discretized grid, computes per-cell
spatial features (distances to fence, roads, waterholes, camps, terrain class).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from scipy.spatial import distance as sp_dist


@dataclass
class Park:
    """Discretized park grid with precomputed spatial features.

    Attributes:
        name: Human-readable park name.
        config: Full parsed YAML config dict.
        resolution: Grid cell size in km.
        nx, ny: Grid dimensions (number of cells along x, y).
        cell_centers: (N, 2) array of cell center coordinates in km.
        inside_mask: (N,) boolean — True if cell is inside park outline.
        pan_mask: (N,) boolean — True if cell is on the pan/special zone.
        terrain: (N,) int — terrain class index per cell.
        terrain_names: list of terrain class names.
        dist_fence: (N,) float — distance to nearest fence/boundary in km.
        dist_road: (N,) float — distance to nearest road segment in km.
        dist_waterhole: (N,) float — distance to nearest waterhole in km.
        dist_camp: (N,) float — distance to nearest camp in km.
        dist_pan_boundary: (N,) float — distance to nearest pan/lake edge in km.
        waterholes: (W, 2) array of waterhole coordinates.
        camps: list of camp dicts from config.
        animal_density: (N,) float — relative animal density per cell.
    """

    name: str
    config: dict[str, Any]
    resolution: float
    nx: int
    ny: int
    cell_centers: NDArray[np.float64]
    inside_mask: NDArray[np.bool_]
    pan_mask: NDArray[np.bool_]
    terrain: NDArray[np.int32]
    terrain_names: list[str]
    dist_fence: NDArray[np.float64]
    dist_road: NDArray[np.float64]
    dist_waterhole: NDArray[np.float64]
    dist_camp: NDArray[np.float64]
    dist_pan_boundary: NDArray[np.float64]
    waterholes: NDArray[np.float64]
    camps: list[dict[str, Any]]
    animal_density: NDArray[np.float64]
    n_cells: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_cells = int(self.inside_mask.sum())

    @property
    def interior_indices(self) -> NDArray[np.intp]:
        """Indices of cells that are inside the park."""
        return np.where(self.inside_mask)[0]

    @property
    def grid_shape(self) -> tuple[int, int]:
        """(ny, nx) shape for reshaping flat arrays to 2D grids."""
        return (self.ny, self.nx)

    def config_hash(self) -> str:
        """SHA-256 hash of the config for reproducibility tracking."""
        raw = json.dumps(self.config, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _point_in_polygon(points: NDArray, polygon: NDArray) -> NDArray[np.bool_]:
    """Ray-casting point-in-polygon test (vectorized).

    Args:
        points: (N, 2) array of query points.
        polygon: (M, 2) array of polygon vertices (closed or open).

    Returns:
        (N,) boolean array.
    """
    n = len(polygon)
    inside = np.zeros(len(points), dtype=bool)
    px, py = points[:, 0], points[:, 1]

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i, 0], polygon[i, 1]
        xj, yj = polygon[j, 0], polygon[j, 1]

        # Edge crosses the horizontal ray from point
        cond = ((yi > py) != (yj > py)) & (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi
        )
        inside ^= cond
        j = i

    return inside


def _dist_to_segments(points: NDArray, segments: list[list[list[float]]]) -> NDArray[np.float64]:
    """Minimum distance from each point to any line segment.

    Args:
        points: (N, 2) array.
        segments: list of [[x1,y1], [x2,y2]] pairs.

    Returns:
        (N,) array of minimum distances.
    """
    min_dist = np.full(len(points), np.inf)

    for seg in segments:
        a = np.array(seg[0], dtype=np.float64)
        b = np.array(seg[1], dtype=np.float64)
        ab = b - a
        ab_sq = np.dot(ab, ab)
        if ab_sq < 1e-12:
            d = np.linalg.norm(points - a, axis=1)
        else:
            # Project each point onto the line, clamp t to [0, 1]
            t = np.clip(((points - a) @ ab) / ab_sq, 0.0, 1.0)
            proj = a + np.outer(t, ab)
            d = np.linalg.norm(points - proj, axis=1)
        min_dist = np.minimum(min_dist, d)

    return min_dist


def _dist_to_polygon_boundary(points: NDArray, polygon: NDArray) -> NDArray[np.float64]:
    """Distance from each point to the nearest edge of a polygon."""
    segments = []
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        segments.append([polygon[i].tolist(), polygon[j].tolist()])
    return _dist_to_segments(points, segments)


def _generate_waterholes(
    anchors: list[dict], total: int, park_polygon: NDArray,
    inside_mask: NDArray, cell_centers: NDArray, seed: int
) -> NDArray[np.float64]:
    """Generate waterhole positions: use anchors + procedurally fill to total.

    Extra waterholes are scattered near existing anchors (within ~20 km)
    and must be inside the park outline.
    """
    rng = np.random.default_rng(seed)
    wh = np.array([[a["x"], a["y"]] for a in anchors], dtype=np.float64)

    remaining = total - len(wh)
    if remaining <= 0:
        return wh

    attempts = 0
    extra = []
    while len(extra) < remaining and attempts < remaining * 50:
        # Pick a random anchor and jitter
        anchor = wh[rng.integers(len(wh))]
        offset = rng.normal(0, 10, size=2)  # ~10 km std dev
        candidate = anchor + offset
        # Check inside park (approximate: check nearest cell)
        dists = np.linalg.norm(cell_centers - candidate, axis=1)
        nearest = np.argmin(dists)
        if inside_mask[nearest] and dists[nearest] < 5.0:
            extra.append(candidate)
        attempts += 1

    if extra:
        wh = np.vstack([wh, np.array(extra)])
    return wh


def _compute_terrain(
    cell_centers: NDArray, inside_mask: NDArray, pan_mask: NDArray,
    config: dict, rng: np.random.Generator
) -> tuple[NDArray[np.int32], list[str]]:
    """Assign terrain class to each cell.

    If config contains ``biome_polygons``, uses priority-ordered point-in-polygon
    assignment (first containing polygon wins).  Unmatched cells receive the
    ``default_biome``.  Falls back to the legacy probabilistic ``terrain_types``
    method when ``biome_polygons`` is absent.
    """
    biome_cfg = config.get("biome_polygons")

    if biome_cfg is not None:
        # --- Polygon-based terrain assignment ---
        default_biome = config.get("default_biome", "grassland")
        # Collect unique biome names in priority order, default last
        names: list[str] = []
        for entry in biome_cfg:
            bname = entry["name"]
            if bname not in names:
                names.append(bname)
        if default_biome not in names:
            names.append(default_biome)
        default_idx = names.index(default_biome)

        terrain = np.full(len(cell_centers), default_idx, dtype=np.int32)

        # Assign in reverse priority so earlier entries overwrite later ones
        for entry in reversed(biome_cfg):
            poly = np.array(entry["polygon"], dtype=np.float64)
            bname = entry["name"]
            idx = names.index(bname)
            mask = _point_in_polygon(cell_centers, poly) & inside_mask
            terrain[mask] = idx

        # Zero out exterior cells (they get index 0 but inside_mask filters them)
        return terrain, names

    # --- Legacy probabilistic method ---
    terrain_types = config.get("terrain_types", {})
    names_legacy = list(terrain_types.keys())
    fracs = np.array([terrain_types[n] for n in names_legacy], dtype=np.float64)

    pan_idx = names_legacy.index("salt_pan") if "salt_pan" in names_legacy else -1
    terrain = np.zeros(len(cell_centers), dtype=np.int32)

    if pan_idx >= 0:
        terrain[pan_mask & inside_mask] = pan_idx
        non_pan_fracs = fracs.copy()
        non_pan_fracs[pan_idx] = 0.0
        if non_pan_fracs.sum() > 0:
            non_pan_fracs /= non_pan_fracs.sum()
    else:
        non_pan_fracs = fracs / fracs.sum()

    non_pan_interior = inside_mask & ~pan_mask
    n_non_pan = non_pan_interior.sum()
    if n_non_pan > 0:
        assignments = rng.choice(len(names_legacy), size=n_non_pan, p=non_pan_fracs)
        terrain[non_pan_interior] = assignments

    return terrain, names_legacy


def _compute_animal_density(
    cell_centers: NDArray, inside_mask: NDArray, pan_mask: NDArray,
    waterholes: NDArray, config: dict
) -> NDArray[np.float64]:
    """Relative animal density per cell.

    Density peaks near waterholes, is very low on the pan, and is baseline elsewhere.
    """
    base = config.get("density_base", 1.0)
    wh_radius = config.get("density_waterhole_radius_km", 15.0)
    wh_peak = config.get("density_waterhole_peak", 3.0)
    pan_factor = config.get("density_pan_factor", 0.1)

    density = np.full(len(cell_centers), base)

    # Waterhole attraction: Gaussian kernel
    if len(waterholes) > 0:
        dists = sp_dist.cdist(cell_centers, waterholes)  # (N, W)
        min_wh_dist = dists.min(axis=1)
        wh_boost = (wh_peak - 1.0) * np.exp(-0.5 * (min_wh_dist / wh_radius) ** 2)
        density += wh_boost

    # Pan suppression
    density[pan_mask] *= pan_factor

    # Zero outside park
    density[~inside_mask] = 0.0

    return density


def load_park(config_path: str | Path, seed: int = 42) -> Park:
    """Load a park configuration and build the discretized grid.

    Args:
        config_path: Path to park YAML config file.
        seed: Random seed for procedural generation (waterholes, terrain).

    Returns:
        Fully initialized Park object.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    rng = np.random.default_rng(seed)
    res = config["grid_resolution_km"]
    bounds = config["bounds"]

    # Build grid
    xs = np.arange(bounds["x_min_km"] + res / 2, bounds["x_max_km"], res)
    ys = np.arange(bounds["y_min_km"] + res / 2, bounds["y_max_km"], res)
    nx, ny = len(xs), len(ys)
    xx, yy = np.meshgrid(xs, ys)
    cell_centers = np.column_stack([xx.ravel(), yy.ravel()])  # (N, 2)

    # Park outline
    outline = np.array(config["park_outline"], dtype=np.float64)
    inside_mask = _point_in_polygon(cell_centers, outline)

    # Pan mask
    pan_poly = np.array(config.get("pan_polygon", []), dtype=np.float64)
    if len(pan_poly) > 0:
        pan_mask = _point_in_polygon(cell_centers, pan_poly) & inside_mask
    else:
        pan_mask = np.zeros(len(cell_centers), dtype=bool)

    # Waterholes
    wh_anchors = config.get("waterhole_anchors", [])
    wh_total = config.get("waterhole_total", len(wh_anchors))
    wh_seed = config.get("waterhole_seed", seed)
    waterholes = _generate_waterholes(
        wh_anchors, wh_total, outline, inside_mask, cell_centers, wh_seed
    )

    # Distance features
    dist_fence = _dist_to_polygon_boundary(cell_centers, outline)
    road_segs = config.get("road_segments", [])
    dist_road = _dist_to_segments(cell_centers, road_segs) if road_segs else np.full(len(cell_centers), 50.0)

    if len(waterholes) > 0:
        dist_waterhole = sp_dist.cdist(cell_centers, waterholes).min(axis=1)
    else:
        dist_waterhole = np.full(len(cell_centers), 50.0)

    camp_list = config.get("camps", [])
    if camp_list:
        camp_coords = np.array([[c["x"], c["y"]] for c in camp_list], dtype=np.float64)
        dist_camp = sp_dist.cdist(cell_centers, camp_coords).min(axis=1)
    else:
        dist_camp = np.full(len(cell_centers), 50.0)

    # Distance to pan boundary
    if len(pan_poly) > 0:
        dist_pan_boundary = _dist_to_polygon_boundary(cell_centers, pan_poly)
    else:
        dist_pan_boundary = np.full(len(cell_centers), 999.0)

    # Terrain
    terrain, terrain_names = _compute_terrain(cell_centers, inside_mask, pan_mask, config, rng)

    # Animal density
    animal_density = _compute_animal_density(cell_centers, inside_mask, pan_mask, waterholes, config)

    return Park(
        name=config["park_name"],
        config=config,
        resolution=res,
        nx=nx,
        ny=ny,
        cell_centers=cell_centers,
        inside_mask=inside_mask,
        pan_mask=pan_mask,
        terrain=terrain,
        terrain_names=terrain_names,
        dist_fence=dist_fence,
        dist_road=dist_road,
        dist_waterhole=dist_waterhole,
        dist_camp=dist_camp,
        dist_pan_boundary=dist_pan_boundary,
        waterholes=waterholes,
        camps=camp_list,
        animal_density=animal_density,
    )
