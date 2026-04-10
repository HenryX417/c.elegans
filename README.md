# Wildlife Protection Optimization Model — IM²C 2026

**Protecting Wildlife at Scale** — Resource allocation model for national park conservation.

## Quick Start

```bash
pip install -r requirements.txt
bash scripts/run_all.sh
```

All figures land in `outputs/figures/`, tables in `outputs/tables/`, raw JSON in `outputs/results/`.

## Reproducing Individual Figures

```bash
# Etosha baseline (risk surface, allocation, disaggregated PPI, temporal sim)
python scripts/run_etosha.py

# All scenario/sensitivity sweeps (dry season, wildfire, equity, Pareto, etc.)
python scripts/run_sensitivity.py

# Adaptation parks (Yellowstone, Sierra del Divisor, comparison)
python scripts/run_adaptations.py
```

## Figure → Command Map

| Figure | Script |
|--------|--------|
| `fig_etosha_risk_surface.png` | `run_etosha.py` |
| `fig_etosha_baseline_allocation.png` | `run_etosha.py` |
| `fig_etosha_ppi_disaggregated.png` | `run_etosha.py` |
| `fig_temporal_30day.png` | `run_etosha.py` |
| `fig_randomization_robustness.png` | `run_sensitivity.py` |
| `fig_dry_season_shift.png` | `run_sensitivity.py` |
| `fig_wildfire_reallocation.png` | `run_sensitivity.py` |
| `fig_tech_human_pareto.png` | `run_sensitivity.py` |
| `fig_diminishing_returns_budget.png` | `run_sensitivity.py` |
| `fig_diminishing_returns_rangers.png` | `run_sensitivity.py` |
| `fig_equity_tradeoff.png` | `run_sensitivity.py` |
| `fig_disruption_pareto.png` | `run_sensitivity.py` |
| `fig_staffing_inverse.png` | `run_sensitivity.py` |
| `fig_yellowstone_risk_surface.png` | `run_adaptations.py` |
| `fig_yellowstone_allocation.png` | `run_adaptations.py` |
| `fig_sierra_risk_surface.png` | `run_adaptations.py` |
| `fig_sierra_allocation.png` | `run_adaptations.py` |
| `fig_adaptation_comparison.png` | `run_adaptations.py` |

## Architecture

Three-layer model:
1. **Risk Surface** — grid-based risk scoring per (species, threat) pair
2. **Resource Allocation** — greedy submodular optimizer with LP upper bound
3. **Routing & Temporal** — team orienteering + 30-day stochastic simulation

All parameters live in YAML configs under `configs/`. Every run is seeded for reproducibility.

## Requirements

Python 3.11+. Dependencies: NumPy, SciPy, Matplotlib, Pandas, PuLP, NetworkX, PyYAML.
