"""Risk surface construction.

Computes per-cell, per-(species, threat) risk scores from spatial features
and config weights. Aggregates to a single risk score per cell.

risk[i, s, t] = sum_f w[f, s, t] * feature[i, f]
risk[i] = sum_{s,t} species_weight[s] * threat_weight[t] * risk[i, s, t]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .park import Park


# Feature names used in the risk model.  Order must match columns in the
# feature matrix built by _build_feature_matrix.
FEATURE_NAMES = [
    "fence_distance",
    "road_distance",
    "camp_distance",
    "waterhole_distance",
    "terrain_openness",
    "seasonal_dry",
]


@dataclass
class RiskSurface:
    """Precomputed risk surface for a park.

    Attributes:
        park: The underlying Park object.
        risk_st: (N, S, T) array — risk per cell, species, threat.
        risk_by_threat: (N, T) array — species-aggregated risk per cell per threat.
        risk_by_species: (N, S) array — threat-aggregated risk per cell per species.
        risk: (N,) array — fully aggregated risk per cell (0 outside park).
        species_names: list of species name strings.
        threat_names: list of threat name strings.
        species_weights: (S,) array.
        threat_weights: (T,) array.
    """

    park: Park
    risk_st: NDArray[np.float64]
    risk_by_threat: NDArray[np.float64]
    risk_by_species: NDArray[np.float64]
    risk: NDArray[np.float64]
    species_names: list[str]
    threat_names: list[str]
    species_weights: NDArray[np.float64]
    threat_weights: NDArray[np.float64]

    @property
    def total_risk(self) -> float:
        """Sum of risk across all interior cells."""
        return float(self.risk[self.park.inside_mask].sum())


def _build_feature_matrix(park: Park, season: str = "dry") -> NDArray[np.float64]:
    """Build (N, F) feature matrix for the risk model.

    Features are normalized to [0, 1] using min-max scaling over interior cells,
    then inverted where the weight convention is "closer = higher risk" (handled
    by negative weights in config, so we keep raw normalized distance here).

    The seasonal_dry feature is 1.0 in dry season, 0.0 in wet season (binary
    toggle; finer monthly resolution can be added later).
    """
    N = len(park.cell_centers)
    F = len(FEATURE_NAMES)
    features = np.zeros((N, F), dtype=np.float64)

    interior = park.inside_mask

    # Raw distance features (km) — will be normalized
    raw = {
        "fence_distance": park.dist_fence,
        "road_distance": park.dist_road,
        "camp_distance": park.dist_camp,
        "waterhole_distance": park.dist_waterhole,
    }

    for i, fname in enumerate(FEATURE_NAMES):
        if fname in raw:
            vals = raw[fname].copy()
            # Normalize over interior cells to [0, 1]
            lo = vals[interior].min()
            hi = vals[interior].max()
            if hi - lo > 1e-9:
                vals = (vals - lo) / (hi - lo)
            else:
                vals = np.zeros_like(vals)
            features[:, i] = vals

        elif fname == "terrain_openness":
            # Openness: pan and grassland are open (1.0); woodland is closed (0.0)
            openness_map: dict[str, float] = {
                "grassland": 0.8,
                "savanna_woodland": 0.4,
                "salt_pan": 1.0,
                "mopane_woodland": 0.2,
                "dolomite_hills": 0.3,
                "forest": 0.1,
                "rainforest": 0.05,
                "alpine": 0.6,
                "wetland": 0.5,
                "shrubland": 0.6,
                "woodland": 0.35,
            }
            for ci, tname in enumerate(park.terrain_names):
                mask = park.terrain == ci
                features[mask, i] = openness_map.get(tname, 0.5)

        elif fname == "seasonal_dry":
            features[:, i] = 1.0 if season == "dry" else 0.0

    return features


def build_risk_surface(
    park: Park,
    season: str = "dry",
    config_override: dict[str, Any] | None = None,
) -> RiskSurface:
    """Build the full risk surface from a park and its config.

    Args:
        park: Initialized Park object.
        season: "dry" or "wet".
        config_override: Optional dict to override species/threat config.

    Returns:
        RiskSurface with all precomputed arrays.
    """
    cfg = config_override if config_override else park.config

    species_cfg = cfg["species"]
    threats_cfg = cfg["threats"]

    species_names = list(species_cfg.keys())
    threat_names = list(threats_cfg.keys())
    S = len(species_names)
    T = len(threat_names)

    raw_species_weights = np.array(
        [species_cfg[s]["weight"] for s in species_names], dtype=np.float64
    )
    populations = np.array(
        [species_cfg[s].get("population", 1000) for s in species_names], dtype=np.float64
    )
    # Effective weight: w / sqrt(pop) — sqrt to avoid rare species completely
    # dominating while still biasing protection toward endangered species.
    species_weights = raw_species_weights / np.sqrt(populations)
    # Normalize so weights sum to original sum (preserves risk magnitude)
    species_weights *= raw_species_weights.sum() / (species_weights.sum() + 1e-12)

    threat_weights = np.array(
        [threats_cfg[t]["weight"] for t in threat_names], dtype=np.float64
    )

    features = _build_feature_matrix(park, season)
    N = len(park.cell_centers)

    # Build weight tensor (S, T, F) from config
    # Each threat defines feature weights; species modulate via habitat
    # preference (species near preferred habitat face higher threat there).
    risk_st = np.zeros((N, S, T), dtype=np.float64)

    for ti, tname in enumerate(threat_names):
        t_features = threats_cfg[tname]["features"]
        # Weight vector for this threat type
        w_t = np.array(
            [t_features.get(fname, 0.0) for fname in FEATURE_NAMES],
            dtype=np.float64,
        )
        # Base threat score per cell: features @ w_t
        # Negative weights mean "closer = higher risk" which works because
        # features are normalized distances: closer -> smaller value,
        # negative weight * small value = negative, so we negate:
        # Actually the convention is: weight is the coefficient directly.
        # fence_distance weight = -0.4 means risk DECREASES with distance
        # (i.e., cells close to fence have low distance -> low contribution
        # from this negative-weight term -> we want risk to be HIGH near fence).
        # So: risk = sum(w * feature). Near fence: feature~0, w=-0.4 -> contribution~0.
        # Far from fence: feature~1, w=-0.4 -> contribution=-0.4.
        # Net: near-fence cells have HIGHER (less negative) total risk. Correct.
        base = features @ w_t  # (N,)

        for si, sname in enumerate(species_names):
            sp = species_cfg[sname]
            # Species habitat modifier: species is more "at risk" in its preferred habitat
            habitat_pref = sp.get("habitat_preference", {})
            habitat_mod = np.ones(N, dtype=np.float64)
            for ci, tn in enumerate(park.terrain_names):
                mask = park.terrain == ci
                habitat_mod[mask] = 0.1 + 0.9 * habitat_pref.get(tn, 0.1)

            # Waterhole affinity: species near waterholes face higher threat near waterholes
            wh_aff = sp.get("waterhole_affinity", 0.5)
            # Boost risk near waterholes proportional to species' affinity
            wh_boost = wh_aff * np.exp(-park.dist_waterhole / 15.0)

            risk_st[:, si, ti] = (base + wh_boost) * habitat_mod

    # Apply biome_effects: poaching_concealment and wildfire_susceptibility
    from .resources import get_biome_effects
    biome_effects = get_biome_effects()
    if biome_effects:
        _threat_biome_key = {
            "poaching": "poaching_concealment",
            "illegal_logging": "poaching_concealment",
        }
        _fire_threats = {"wildfire"}
        for ti, tname in enumerate(threat_names):
            bkey = _threat_biome_key.get(tname)
            is_fire = tname in _fire_threats
            if bkey or is_fire:
                for ci, tn in enumerate(park.terrain_names):
                    be = biome_effects.get(tn, {})
                    mask = park.terrain == ci
                    if bkey:
                        mult = 0.5 + 0.5 * be.get(bkey, 0.5)
                        risk_st[mask, :, ti] *= mult
                    if is_fire:
                        mult = 0.5 + 0.5 * be.get("wildfire_susceptibility", 0.5)
                        risk_st[mask, :, ti] *= mult

    # Shift risk to be non-negative (per species-threat pair)
    for si in range(S):
        for ti in range(T):
            col = risk_st[:, si, ti]
            col_min = col[park.inside_mask].min() if park.inside_mask.any() else 0.0
            if col_min < 0:
                risk_st[:, si, ti] -= col_min

    # Seasonal pan risk modifiers
    if season == "wet":
        # Wet season: pan fills with water → zero risk on pan cells
        risk_st[park.pan_mask, :, :] = 0.0
    else:
        # Dry season: pan edge cells (within 5 km of pan boundary) get risk boost
        # Animals concentrate at drying pan edges
        pan_edge = (~park.pan_mask) & (park.dist_pan_boundary < 5.0) & park.inside_mask
        risk_st[pan_edge, :, :] *= 1.8

    # Zero out exterior cells
    risk_st[~park.inside_mask, :, :] = 0.0

    # Aggregate by species (summing over threats with threat weights)
    risk_by_species = np.zeros((N, S), dtype=np.float64)
    for si in range(S):
        risk_by_species[:, si] = (risk_st[:, si, :] * threat_weights[None, :]).sum(axis=1)

    # Aggregate by threat (summing over species with species weights)
    risk_by_threat = np.zeros((N, T), dtype=np.float64)
    for ti in range(T):
        risk_by_threat[:, ti] = (risk_st[:, :, ti] * species_weights[None, :]).sum(axis=1)

    # Fully aggregated risk per cell
    risk = np.zeros(N, dtype=np.float64)
    for si in range(S):
        for ti in range(T):
            risk += species_weights[si] * threat_weights[ti] * risk_st[:, si, ti]

    # Seasonal multiplier
    season_cfg = park.config.get("seasons", {}).get(season, {})
    season_mult = season_cfg.get("multiplier", 1.0)
    risk *= season_mult
    risk_by_species *= season_mult
    risk_by_threat *= season_mult
    risk_st *= season_mult

    return RiskSurface(
        park=park,
        risk_st=risk_st,
        risk_by_threat=risk_by_threat,
        risk_by_species=risk_by_species,
        risk=risk,
        species_names=species_names,
        threat_names=threat_names,
        species_weights=species_weights,
        threat_weights=threat_weights,
    )
