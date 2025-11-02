#!/bin/bash
# Master Script: Run Phase 02 and Phase 03 Sequentially
#
# 1. Runs Phase 02 (vintage validation) - 6 experiments
# 2. Waits for completion and summarizes results
# 3. Runs Phase 03 (distribution loss) - 9 experiments
# 4. Summarizes final results

echo "========================================================================"
echo "MASTER SCRIPT: PHASE 02 & 03"
echo "========================================================================"
echo "Total experiments: 15"
echo "Started at: $(date)"
echo ""

# ============================================================================
# PHASE 02: VINTAGE VALIDATION
# ============================================================================

echo "========================================================================"
echo "RUNNING PHASE 02: VINTAGE VALIDATION (6 experiments)"
echo "========================================================================"
echo ""

bash run_phase_02.sh

if [ $? -ne 0 ]; then
    echo "ERROR: Phase 02 failed"
    exit 1
fi

echo ""
echo "Phase 02 complete. Summarizing results..."
echo ""

# Summarize Phase 02 - save to phase-specific directory
python summarize_experiments.py \
    --phase 02_vintage_baseline \
    --output-dir experiments/02_vintage_baseline \
    --evaluated-only

echo ""
echo "Press Enter to continue to Phase 03, or Ctrl+C to stop..."
read

# ============================================================================
# PHASE 03: DISTRIBUTION LOSS
# ============================================================================

echo ""
echo "========================================================================"
echo "RUNNING PHASE 03: DISTRIBUTION LOSS (9 experiments)"
echo "========================================================================"
echo ""

bash run_phase_03.sh

if [ $? -ne 0 ]; then
    echo "ERROR: Phase 03 failed"
    exit 1
fi

echo ""
echo "Phase 03 complete. Summarizing results..."
echo ""

# Summarize Phase 03 - save to phase-specific directory
python summarize_experiments.py \
    --phase 03_distribution_loss \
    --output-dir experiments/03_distribution_loss \
    --evaluated-only

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo "========================================================================"
echo "ALL PHASES COMPLETE"
echo "========================================================================"
echo "Completed at: $(date)"
echo ""
echo "Results:"
echo "  Phase 02: experiments/02_vintage_baseline/"
echo "  Phase 03: experiments/03_distribution_loss/"
echo ""
echo "Logs:"
echo "  Phase 02: logs/phase_02/"
echo "  Phase 03: logs/phase_03/"
echo ""
