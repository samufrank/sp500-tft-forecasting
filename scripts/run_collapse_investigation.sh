#!/bin/bash

# Master script for systematic collapse investigation
# Runs all three analysis approaches
#
# Usage: bash scripts/run_collapse_investigation.sh [phase]
#   phase = all|1|2|3  (default: all)

PHASE=${1:-all}

echo "========================================================================="
echo "COLLAPSE INVESTIGATION FRAMEWORK"
echo "========================================================================="
echo ""
echo "This will run systematic experiments to understand why TFT models"
echo "collapse to constant predictions for hidden sizes > 18."
echo ""
echo "Three-pronged approach:"
echo "  1. Training Dynamics Monitoring"
echo "  2. Capacity Threshold Analysis"  
echo "  3. Architecture Ablation (design only)"
echo ""

# ============================================================================
# PHASE 1: Training Dynamics Monitoring
# ============================================================================

if [[ "$PHASE" == "all" ]] || [[ "$PHASE" == "1" ]]; then
    echo "========================================================================="
    echo "PHASE 1: Training Dynamics Monitoring"
    echo "========================================================================="
    echo ""
    echo "Running baseline experiments with detailed collapse monitoring..."
    echo ""
    
    # Quick test on working vs collapsed
    python train/train_with_monitoring.py \
        --experiment-name dynamics_h16_working \
        --hidden-size 16 \
        --learning-rate 0.0005 \
        --dropout 0.15 \
        --max-encoder-length 20 \
        --max-epochs 50 \
        --monitor-every-n-epochs 1
    
    python train/train_with_monitoring.py \
        --experiment-name dynamics_h24_collapsed \
        --hidden-size 24 \
        --learning-rate 0.0005 \
        --dropout 0.15 \
        --max-encoder-length 20 \
        --max-epochs 50 \
        --monitor-every-n-epochs 1
    
    echo ""
    echo "Phase 1 complete. Check:"
    echo "  experiments/dynamics_*/collapse_monitoring/"
    echo ""
fi

# ============================================================================
# PHASE 2: Capacity Analysis
# ============================================================================

if [[ "$PHASE" == "all" ]] || [[ "$PHASE" == "2" ]]; then
    echo "========================================================================="
    echo "PHASE 2: Capacity Threshold Analysis"
    echo "========================================================================="
    echo ""
    echo "Running systematic capacity sweep..."
    echo "This will take several hours."
    echo ""
    
    read -p "Continue with full capacity sweep? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        bash scripts/sweep_capacity_analysis.sh
        
        echo ""
        echo "Analyzing results..."
        python scripts/analyze_capacity_sweep.py
        
        echo ""
        echo "Phase 2 complete. Check:"
        echo "  experiments/capacity_analysis_*.png"
        echo "  experiments/capacity_analysis_results.csv"
    else
        echo "Skipping Phase 2."
    fi
    echo ""
fi

# ============================================================================
# PHASE 3: Architecture Ablation
# ============================================================================

if [[ "$PHASE" == "all" ]] || [[ "$PHASE" == "3" ]]; then
    echo "========================================================================="
    echo "PHASE 3: Architecture Ablation"
    echo "========================================================================="
    echo ""
    echo "This phase requires implementing simplified TFT variants."
    echo "See scripts/sweep_architecture_ablation.sh for design."
    echo ""
    
    read -p "Run preliminary architectural variations? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        bash scripts/sweep_architecture_ablation.sh
        echo ""
        echo "Phase 3 (preliminary) complete."
    else
        echo "Skipping Phase 3."
    fi
    echo ""
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo "========================================================================="
echo "INVESTIGATION SUMMARY"
echo "========================================================================="
echo ""
echo "Completed phases: $PHASE"
echo ""
echo "Generated:"
echo "  - Monitoring logs: experiments/*/collapse_monitoring/"
echo "  - Capacity plots: experiments/capacity_analysis_*.png"
echo "  - Results CSV: experiments/capacity_analysis_results.csv"
echo ""
echo "Next:"
echo "  1. Review monitoring logs to see when collapse occurs"
echo "  2. Check capacity_threshold_analysis.png for critical size"
echo "  3. Examine gradient flow and VSN activity in collapsed models"
echo "  4. (Optional) Implement architecture ablations for deeper analysis"
echo ""
echo "Document all findings in sections on:"
echo "  - Training dynamics (when/how collapse happens)"
echo "  - Capacity constraints (parameter/sample ratio threshold)"
echo "  - Architectural factors (if ablations completed)"
echo ""
echo "========================================================================="

