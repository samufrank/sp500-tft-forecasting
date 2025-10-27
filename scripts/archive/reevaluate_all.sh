#!/bin/bash
#
# Re-evaluate all experiments that have been previously evaluated.
#
# Usage:
#   ./reevaluate_all.sh                    # Re-evaluate all phases
#   ./reevaluate_all.sh 00_baseline_exploration  # Re-evaluate specific phase
#

#set -e  # pytorch doesn't guarantee clean exists

RED=''
GREEN=''
YELLOW=''
NC=''

# Configuration
EXPERIMENTS_DIR="experiments"
EVALUATE_SCRIPT="train/evaluate_tft.py"

# Check if evaluate script exists
if [ ! -f "$EVALUATE_SCRIPT" ]; then
    echo -e "${RED}ERROR: $EVALUATE_SCRIPT not found in current directory${NC}"
    echo "Please run this script from your project root"
    exit 1
fi

# Determine which phases to process
if [ $# -eq 0 ]; then
    # No arguments - process all phases
    PHASES=$(find "$EXPERIMENTS_DIR" -mindepth 1 -maxdepth 1 -type d -name "*_*" | sort)
    echo -e "${GREEN}Processing all phases${NC}"
else
    # Specific phase provided
    PHASES="$EXPERIMENTS_DIR/$1"
    if [ ! -d "$PHASES" ]; then
        echo -e "${RED}ERROR: Phase directory not found: $PHASES${NC}"
        exit 1
    fi
    echo -e "${GREEN}Processing phase: $1${NC}"
fi

# Counters
total_found=0
total_success=0
total_failed=0
total_skipped=0

# Log file for batch run
batch_log="reevaluation_batch_$(date +%Y%m%d_%H%M%S).log"
echo "Batch re-evaluation started at $(date)" > "$batch_log"
echo "========================================" >> "$batch_log"

# Function to re-evaluate a single experiment
reevaluate_experiment() {
    local exp_path=$1
    local exp_name=$(basename "$exp_path")
    local phase=$(basename $(dirname "$exp_path"))
    local full_exp_name="$phase/$exp_name"
    
    echo -e "${YELLOW}Processing: $full_exp_name${NC}"
    echo "Processing: $full_exp_name" >> "$batch_log"
    
    # Run evaluation
    if python "$EVALUATE_SCRIPT" --experiment-name "$full_exp_name" 2>&1 | tee -a "$batch_log"; then
        echo -e "${GREEN}  ✓ Success${NC}"
        echo "  ✓ Success" >> "$batch_log"
	echo ""
        ((total_success++))
    else
        echo -e "${RED}  ✗ Failed${NC}"
        echo "  ✗ Failed" >> "$batch_log"
	echo ""
        ((total_failed++))
    fi
    echo "" >> "$batch_log"
}

# Process each phase
for phase_dir in $PHASES; do
    phase_name=$(basename "$phase_dir")
    echo ""
    echo "========================================================================"
    echo "PHASE: $phase_name"
    echo "========================================================================"
    echo ""
    echo "========================================================================"  >> "$batch_log"
    echo "PHASE: $phase_name" >> "$batch_log"
    echo "========================================================================" >> "$batch_log"
    
    # Find all experiments with evaluation directories
    for exp_dir in "$phase_dir"/*; do
        if [ ! -d "$exp_dir" ]; then
            continue
        fi
        
        eval_dir="$exp_dir/evaluation"
        
        if [ -d "$eval_dir" ]; then
            # Has evaluation directory - re-evaluate
            ((total_found++))
            reevaluate_experiment "$exp_dir"
        else
            # No evaluation directory - skip
            exp_name=$(basename "$exp_dir")
            echo -e "${YELLOW}Skipping: $phase_name/$exp_name (no previous evaluation)${NC}"
            ((total_skipped++))
        fi
    done
done

# Summary
echo ""
echo "========================================================================"
echo "BATCH RE-EVALUATION SUMMARY"
echo "========================================================================"
echo "Total experiments with evaluations: $total_found"
echo "Successfully re-evaluated:          $total_success"
echo "Failed:                             $total_failed"
echo "Skipped (no previous evaluation):   $total_skipped"
echo ""
echo "Batch log saved to: $batch_log"
echo "Individual logs saved in: experiments/<phase>/<exp>/evaluation/evaluation_*.log"
echo "========================================================================"

# Write summary to batch log
echo "" >> "$batch_log"
echo "========================================" >> "$batch_log"
echo "SUMMARY" >> "$batch_log"
echo "========================================" >> "$batch_log"
echo "Total experiments with evaluations: $total_found" >> "$batch_log"
echo "Successfully re-evaluated:          $total_success" >> "$batch_log"
echo "Failed:                             $total_failed" >> "$batch_log"
echo "Skipped (no previous evaluation):   $total_skipped" >> "$batch_log"
echo "Batch completed at $(date)" >> "$batch_log"

# Exit with error if any failed
if [ $total_failed -gt 0 ]; then
    exit 1
fi

exit 0
