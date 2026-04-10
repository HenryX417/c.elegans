"""Figure generation with consistent style.

All figures save as both PNG (300 DPI) and PDF with stable filenames
for LaTeX \\includegraphics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from numpy.typing import NDArray

from .park import Park
from .resources import Allocation, compute_detection_prob, compute_effective_units
from .risk import RiskSurface

# Load our custom style
_STYLE_PATH = Path(__file__).parent.parent / "style.mplstyle"
if _STYLE_PATH.exists():
    plt.style.use(str(_STYLE_PATH))

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Custom colormaps
RISK_CMAP = LinearSegmentedColormap.from_list(
    "risk", ["#FFFFFF", "#FFF3B0", "#F18F01", "#C73E1D", "#3B1F2B"], N=256
)
ALLOC_CMAP = LinearSegmentedColormap.from_list(
    "alloc", ["#FFFFFF", "#B8E6D0", "#44BBA4", "#2E86AB", "#1B4965"], N=256
)


def _save(fig: plt.Figure, name: str) -> None:
    """Save figure as PNG and PDF."""
    png_path = OUTPUT_DIR / f"{name}.png"
    pdf_path = OUTPUT_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _grid_to_2d(park: Park, values: NDArray, fill: float = np.nan) -> NDArray:
    """Reshape flat (N,) array to (ny, nx) 2D grid, masking outside cells."""
    grid = np.full(park.ny * park.nx, fill)
    grid[:len(values)] = values
    grid[~park.inside_mask] = fill
    return grid.reshape(park.ny, park.nx)


def _draw_park_features(ax: plt.Axes, park: Park) -> None:
    """Overlay park outline, pan, waterholes, and camps on an axis."""
    outline = np.array(park.config["park_outline"])
    outline_closed = np.vstack([outline, outline[0]])
    ax.plot(outline_closed[:, 0], outline_closed[:, 1], "k-", linewidth=1.5, label="Park boundary")

    pan = park.config.get("pan_polygon")
    if pan:
        pan_arr = np.array(pan)
        pan_closed = np.vstack([pan_arr, pan_arr[0]])
        ax.plot(pan_closed[:, 0], pan_closed[:, 1], "k--", linewidth=1, alpha=0.5, label="Pan/special zone")

    # Waterholes
    if len(park.waterholes) > 0:
        ax.scatter(park.waterholes[:, 0], park.waterholes[:, 1],
                   c="#2E86AB", s=12, marker="o", alpha=0.7, zorder=5, label="Waterholes")

    # Camps
    for camp in park.camps:
        marker = "s" if camp.get("type") == "main" else "^"
        color = "#C73E1D" if camp.get("type") == "main" else "#F18F01"
        ax.plot(camp["x"], camp["y"], marker=marker, color=color,
                markersize=8, zorder=6)

    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Distance (km)")
    ax.set_aspect("equal")


def plot_risk_surface(park: Park, risk_surface: RiskSurface, name: str = "fig_etosha_risk_surface") -> None:
    """Heatmap of aggregate risk score overlaid on park geometry."""
    fig, ax = plt.subplots(figsize=(10, 6))

    grid = _grid_to_2d(park, risk_surface.risk)
    bounds = park.config["bounds"]
    extent = [bounds["x_min_km"], bounds["x_max_km"], bounds["y_min_km"], bounds["y_max_km"]]

    im = ax.imshow(grid, origin="lower", extent=extent, cmap=RISK_CMAP, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Aggregate Risk Score")

    _draw_park_features(ax, park)
    ax.set_title(f"{park.name} — Risk Surface")
    ax.legend(loc="upper right", fontsize=8)

    _save(fig, name)


def plot_allocation(
    allocation: Allocation,
    name: str = "fig_etosha_baseline_allocation",
    title_suffix: str = "Baseline Allocation",
) -> None:
    """Heatmap of total allocated resource intensity."""
    park = allocation.park
    fig, ax = plt.subplots(figsize=(10, 6))

    # Total resource intensity per cell (cost-weighted)
    intensity = np.zeros(len(park.cell_centers))
    for k, rt in enumerate(allocation.resource_types):
        intensity += allocation.units[:, k] * rt.cost_per_unit

    grid = _grid_to_2d(park, intensity)
    bounds = park.config["bounds"]
    extent = [bounds["x_min_km"], bounds["x_max_km"], bounds["y_min_km"], bounds["y_max_km"]]

    im = ax.imshow(grid, origin="lower", extent=extent, cmap=ALLOC_CMAP, aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Resource Intensity (cost-weighted)")

    _draw_park_features(ax, park)
    ax.set_title(f"{park.name} — {title_suffix}")
    ax.legend(loc="upper right", fontsize=8)

    _save(fig, name)


def plot_ppi_disaggregated(
    disagg: dict[str, dict[str, float]],
    name: str = "fig_etosha_ppi_disaggregated",
    park_name: str = "Etosha",
) -> None:
    """Bar chart of PPI by species and by threat type."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # By species
    species = list(disagg["by_species"].keys())
    species_ppi = [disagg["by_species"][s] for s in species]
    colors_s = ["#2E86AB", "#A23B72", "#F18F01", "#4B8F29"][:len(species)]
    ax1.barh(species, species_ppi, color=colors_s)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("PPI")
    ax1.set_title(f"{park_name} — PPI by Species")
    for i, v in enumerate(species_ppi):
        ax1.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=9)

    # By threat
    threats = list(disagg["by_threat"].keys())
    threat_ppi = [disagg["by_threat"][t] for t in threats]
    colors_t = ["#C73E1D", "#E94F37", "#F18F01", "#44BBA4"][:len(threats)]
    ax2.barh(threats, threat_ppi, color=colors_t)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("PPI")
    ax2.set_title(f"{park_name} — PPI by Threat Type")
    for i, v in enumerate(threat_ppi):
        ax2.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=9)

    fig.suptitle(f"{park_name} — Disaggregated Protection Index", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, name)


def plot_temporal(
    ppi_ts: NDArray,
    intercepted_ts: NDArray | None = None,
    name: str = "fig_temporal_30day",
    park_name: str = "Etosha",
) -> None:
    """PPI over time (30-day simulation)."""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    days = np.arange(1, len(ppi_ts) + 1)

    ax1.plot(days, ppi_ts, "o-", color="#2E86AB", linewidth=2, markersize=4, label="Daily PPI")
    ax1.axhline(ppi_ts.mean(), color="#A23B72", linestyle="--", alpha=0.7,
                label=f"Mean PPI = {ppi_ts.mean():.3f}")
    ax1.fill_between(days, ppi_ts.min(), ppi_ts, alpha=0.1, color="#2E86AB")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Park Protection Index")
    ax1.set_ylim(0, min(1.0, ppi_ts.max() * 1.3))
    ax1.legend(loc="lower right")
    ax1.set_title(f"{park_name} — 30-Day Protection Simulation")

    if intercepted_ts is not None:
        ax2 = ax1.twinx()
        ax2.bar(days, intercepted_ts, alpha=0.3, color="#F18F01", label="Intercepted threats")
        ax2.set_ylabel("Threats Intercepted")
        ax2.legend(loc="upper right")

    fig.tight_layout()
    _save(fig, name)


def plot_comparison_maps(
    alloc_before: Allocation,
    alloc_after: Allocation,
    name: str,
    title_before: str = "Before",
    title_after: str = "After",
    suptitle: str = "",
) -> None:
    """Side-by-side allocation maps for scenario comparison."""
    park = alloc_before.park
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    bounds = park.config["bounds"]
    extent = [bounds["x_min_km"], bounds["x_max_km"], bounds["y_min_km"], bounds["y_max_km"]]

    for ax, alloc, title in [(ax1, alloc_before, title_before), (ax2, alloc_after, title_after)]:
        intensity = np.zeros(len(park.cell_centers))
        for k, rt in enumerate(alloc.resource_types):
            intensity += alloc.units[:, k] * rt.cost_per_unit
        grid = _grid_to_2d(park, intensity)
        im = ax.imshow(grid, origin="lower", extent=extent, cmap=ALLOC_CMAP, aspect="equal")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Resource Intensity")
        _draw_park_features(ax, park)
        ax.set_title(title)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, name)


def plot_pareto(
    x_vals: NDArray,
    y_vals: NDArray,
    x_label: str,
    y_label: str,
    name: str,
    title: str = "",
    highlight_idx: int | None = None,
    highlight_label: str = "",
) -> None:
    """Pareto frontier plot."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x_vals, y_vals, "o-", color="#2E86AB", linewidth=2, markersize=6)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if highlight_idx is not None:
        ax.axvline(x_vals[highlight_idx], color="#C73E1D", linestyle="--", alpha=0.7)
        ax.plot(x_vals[highlight_idx], y_vals[highlight_idx], "s",
                color="#C73E1D", markersize=10, zorder=5, label=highlight_label)
        ax.legend()

    if title:
        ax.set_title(title)

    fig.tight_layout()
    _save(fig, name)


def plot_diminishing_returns(
    x_vals: NDArray,
    y_vals: NDArray,
    x_label: str,
    name: str,
    title: str = "",
    vline: float | None = None,
    vline_label: str = "",
) -> None:
    """Diminishing returns curve with optional vertical reference line."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x_vals, y_vals, "o-", color="#2E86AB", linewidth=2, markersize=5)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Park Protection Index (PPI)")
    ax.set_ylim(0, 1)

    if vline is not None:
        ax.axvline(vline, color="#C73E1D", linestyle="--", linewidth=2, alpha=0.8)
        # Find nearest y value
        nearest_idx = np.argmin(np.abs(x_vals - vline))
        ax.plot(vline, y_vals[nearest_idx], "s", color="#C73E1D",
                markersize=10, zorder=5, label=vline_label)
        ax.legend()

    if title:
        ax.set_title(title)

    fig.tight_layout()
    _save(fig, name)


def plot_robustness_comparison(
    det_score: float,
    rand_score: float,
    name: str = "fig_randomization_robustness",
    park_name: str = "Etosha",
) -> None:
    """Bar chart comparing deterministic vs randomized robustness."""
    fig, ax = plt.subplots(figsize=(6, 5))

    labels = ["Deterministic\nPatrol", "Randomized\nPatrol"]
    values = [det_score, rand_score]
    colors = ["#C73E1D", "#2E86AB"]

    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.set_ylabel("Robustness Score\n(Detection at adversarial cell)")
    ax.set_ylim(0, max(values) * 1.4)
    ax.set_title(f"{park_name} — Patrol Randomization Robustness")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold")

    gain = rand_score - det_score
    ax.text(0.5, 0.9, f"Robustness gain: +{gain:.3f} ({gain/det_score*100:.1f}%)" if det_score > 0 else "",
            transform=ax.transAxes, ha="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F4FD", alpha=0.8))

    fig.tight_layout()
    _save(fig, name)


def plot_equity_tradeoff(
    p_min_vals: NDArray,
    ppi_vals: NDArray,
    equity_vals: NDArray,
    name: str = "fig_equity_tradeoff",
    park_name: str = "Etosha",
) -> None:
    """PPI vs equity min/mean ratio as equity floor varies."""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "#2E86AB"
    color2 = "#A23B72"

    ax1.plot(p_min_vals, ppi_vals, "o-", color=color1, linewidth=2, label="PPI")
    ax1.set_xlabel("Equity Floor (p_min)")
    ax1.set_ylabel("PPI", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.plot(p_min_vals, equity_vals, "s--", color=color2, linewidth=2, label="Min/Mean Ratio")
    ax2.set_ylabel("Equity (Min/Mean Ratio)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title(f"{park_name} — PPI vs. Equity Tradeoff")
    fig.tight_layout()
    _save(fig, name)


def plot_staffing_inverse(
    targets: NDArray,
    rangers_needed: NDArray,
    name: str = "fig_staffing_inverse",
    park_name: str = "Etosha",
    current_rangers: int = 295,
) -> None:
    """Required rangers vs target PPI."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(targets, rangers_needed, "o-", color="#2E86AB", linewidth=2, markersize=6)
    ax.set_xlabel("Target PPI")
    ax.set_ylabel("Minimum Ranger Teams Required")
    ax.set_title(f"{park_name} — Staffing Requirements by Protection Target")

    ax.axhline(current_rangers, color="#C73E1D", linestyle="--", alpha=0.7,
               label=f"Current staffing ({current_rangers} rangers)")
    ax.legend()

    fig.tight_layout()
    _save(fig, name)


def plot_adaptation_comparison(
    park_names: list[str],
    metrics: dict[str, list[float]],
    name: str = "fig_adaptation_comparison",
) -> None:
    """Grouped bar chart comparing key metrics across parks."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_names = list(metrics.keys())
    n_parks = len(park_names)
    n_metrics = len(metric_names)
    x = np.arange(n_metrics)
    width = 0.25

    colors = ["#2E86AB", "#A23B72", "#F18F01"]
    for i, (pname, color) in enumerate(zip(park_names, colors)):
        values = [metrics[m][i] for m in metric_names]
        offset = (i - n_parks / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=pname, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Cross-Park Adaptation Comparison")
    ax.legend()

    fig.tight_layout()
    _save(fig, name)
