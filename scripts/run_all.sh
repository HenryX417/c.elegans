#!/usr/bin/env bash
# Regenerate every figure and table for the IM²C 2026 paper.
# Usage: bash scripts/run_all.sh
#
# Prerequisites: pip install -r requirements.txt
# Expected runtime: < 10 minutes on a laptop

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=============================================="
echo " IM²C 2026 — Wildlife Protection Model"
echo " Regenerating all figures and tables"
echo "=============================================="
echo ""

# Create output directories
mkdir -p outputs/figures outputs/tables outputs/results

# Step 1: Etosha baseline
echo "[Step 1/3] Etosha baseline + temporal simulation..."
python scripts/run_etosha.py
echo ""

# Step 2: Sensitivity & scenario analysis
echo "[Step 2/3] Sensitivity & scenario analysis..."
python scripts/run_sensitivity.py
echo ""

# Step 3: Adaptation parks
echo "[Step 3/3] Adaptation parks (Yellowstone, Sierra del Divisor)..."
python scripts/run_adaptations.py
echo ""

# Verify all expected figures exist
echo "=============================================="
echo " Verifying outputs..."
echo "=============================================="

EXPECTED_FIGURES=(
    "fig_etosha_risk_surface"
    "fig_etosha_baseline_allocation"
    "fig_etosha_ppi_disaggregated"
    "fig_temporal_30day"
    "fig_randomization_robustness"
    "fig_dry_season_shift"
    "fig_wildfire_reallocation"
    "fig_tech_human_pareto"
    "fig_diminishing_returns_budget"
    "fig_diminishing_returns_rangers"
    "fig_equity_tradeoff"
    "fig_disruption_pareto"
    "fig_staffing_inverse"
    "fig_yellowstone_risk_surface"
    "fig_yellowstone_allocation"
    "fig_sierra_risk_surface"
    "fig_sierra_allocation"
    "fig_adaptation_comparison"
)

EXPECTED_TABLES=(
    "table_disaggregated_ppi"
    "table_scenario_comparison"
    "table_sensitivity_summary"
    "table_adaptation_parameters"
)

missing=0
for fig in "${EXPECTED_FIGURES[@]}"; do
    if [ -f "outputs/figures/${fig}.png" ]; then
        echo "  [OK] ${fig}.png"
    else
        echo "  [MISSING] ${fig}.png"
        missing=$((missing + 1))
    fi
done

for tbl in "${EXPECTED_TABLES[@]}"; do
    if [ -f "outputs/tables/${tbl}.csv" ]; then
        echo "  [OK] ${tbl}.csv"
    else
        echo "  [MISSING] ${tbl}.csv"
        missing=$((missing + 1))
    fi
done

echo ""
if [ $missing -eq 0 ]; then
    echo "All ${#EXPECTED_FIGURES[@]} figures and ${#EXPECTED_TABLES[@]} tables generated successfully!"
else
    echo "WARNING: $missing expected outputs are missing."
    exit 1
fi

echo ""
echo "Output locations:"
echo "  Figures: outputs/figures/"
echo "  Tables:  outputs/tables/"
echo "  Results: outputs/results/"
echo ""
echo "Done!"
