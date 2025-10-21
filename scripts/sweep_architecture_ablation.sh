#!/bin/bash

# Architecture ablation sweep: Which TFT components cause collapse?
#
# This requires creating modified TFT variants - placeholder for now
# Would need to implement simplified TFT versions in models/
#
# Run with: bash scripts/sweep_architecture_ablation.sh

echo "Architecture ablation sweep at $(date)"
echo ""
echo "Goal: Identify which TFT components contribute to collapse instability"
echo ""

# ============================================================================
# NOTE: This requires implementing simplified TFT variants
# ============================================================================
echo "IMPLEMENTATION NOTE:"
echo "This sweep requires creating modified TFT model variants."
echo "Need to implement in models/:"
echo "  - tft_no_attention.py (LSTM encoder-decoder only)"
echo "  - tft_simple_vsn.py (simplified variable selection)"
echo "  - tft_no_static.py (disable static context enrichment)"
echo "  - tft_minimal.py (minimal GRN complexity)"
echo ""
echo "For now, documenting the experimental design..."
echo ""

# ============================================================================
# EXPERIMENTAL DESIGN
# ============================================================================

cat << 'EOF'
ARCHITECTURE ABLATION EXPERIMENTS
==================================

Test each variant with h=16 (works) and h=24 (collapses baseline)

1. BASELINE TFT (full architecture)
   - Already have: capacity_baseline_h16, capacity_baseline_h24

2. NO ATTENTION
   - Remove multi-head attention
   - Test if attention mechanism causes instability
   - Commands (once implemented):
     python train/train_tft_ablation.py --variant no_attention --hidden-size 16
     python train/train_tft_ablation.py --variant no_attention --hidden-size 24

3. SIMPLIFIED VARIABLE SELECTION
   - Replace VSN with simple linear projection
   - Test if VSN gating causes collapse
   - Commands:
     python train/train_tft_ablation.py --variant simple_vsn --hidden-size 16
     python train/train_tft_ablation.py --variant simple_vsn --hidden-size 24

4. NO STATIC ENRICHMENT
   - Disable all static context pathways
   - Test if static processing is problematic
   - Commands:
     python train/train_tft_ablation.py --variant no_static --hidden-size 16
     python train/train_tft_ablation.py --variant no_static --hidden-size 24

5. MINIMAL GRN
   - Replace GRN with simple feedforward
   - Test if GRN complexity causes issues
   - Commands:
     python train/train_tft_ablation.py --variant minimal_grn --hidden-size 16
     python train/train_tft_ablation.py --variant minimal_grn --hidden-size 24

6. LSTM-ONLY BASELINE
   - Pure LSTM encoder-decoder, no TFT components
   - Compare to standard seq2seq
   - Commands:
     python train/train_tft_ablation.py --variant lstm_only --hidden-size 16
     python train/train_tft_ablation.py --variant lstm_only --hidden-size 24

EXPECTED OUTCOMES
=================

If collapse is due to:
- Attention: variants 2,6 will work at h=24
- Variable Selection: variant 3 will work at h=24
- Static enrichment: variant 4 will work at h=24
- GRN complexity: variant 5 will work at h=24
- Fundamental capacity: all variants collapse at h=24

IMPLEMENTATION PRIORITY
=======================

1. LSTM-only baseline (easiest - can use standard PyTorch LSTM)
2. Simplified VSN (likely culprit based on gating hypothesis)
3. No attention (test attention hypothesis)
4. Others if needed

EOF

echo ""
echo "To implement this sweep:"
echo "  1. Create simplified model variants in models/"
echo "  2. Create train_tft_ablation.py that can load variants"
echo "  3. Run this script to test all variants"
echo "  4. Analyze with analyze_ablation_sweep.py"
echo ""
echo "Estimated implementation time: 1-2 days"
echo ""

# ============================================================================
# QUICK TEST: Can we at least test different TFT hyperparams?
# ============================================================================
echo "Running quick architectural variations with existing TFT..."
echo ""

# Test with different attention heads (architectural parameter)
for attn_heads in 1 2 4 8; do
    python train/train_with_monitoring.py \
        --experiment-name ablation_h24_attn${attn_heads} \
        --hidden-size 24 \
        --attention-heads ${attn_heads} \
        --learning-rate 0.0005 \
        --dropout 0.15 \
        --max-epochs 50 \
        --monitor-every-n-epochs 2
done

# Test with different hidden_continuous_size (affects GRN complexity)
for hcs in 4 8 16; do
    python train/train_with_monitoring.py \
        --experiment-name ablation_h24_hcs${hcs} \
        --hidden-size 24 \
        --hidden-continuous-size ${hcs} \
        --learning-rate 0.0005 \
        --dropout 0.15 \
        --max-epochs 50 \
        --monitor-every-n-epochs 2
done

echo ""
echo "Quick architectural tests completed at $(date)"
echo "These test different TFT hyperparameters, not full ablations."
echo ""
echo "For complete ablation study, implement the variants described above."

