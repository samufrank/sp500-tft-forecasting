"""
Run this to see exactly what baseline uses for encoder features.

This will answer the question: Does baseline include encoder_length in encoder VSN or not?

Usage:
    python diagnose_baseline_features.py

Expected output will show:
1. What's in training_dataset.reals (all features)
2. What's in training_dataset.time_varying_reals_encoder (encoder VSN inputs)
3. What's in model encoder VSN configuration
4. Actual parameter counts in encoder VSN
"""

# The debug code I added to train_tft.py will print this info.
# Just run your baseline with max_epochs=1 to see the output:

# cd /mnt/project
# python train_tft.py --experiment-name debug_features --max-epochs 1 --batch-size 64

# Look for the section:
# ================================================================================
# BASELINE FEATURE INSPECTION (before model creation)
# ================================================================================

# This will show you:
# - training_dataset.reals: [...] (Count: X)
# - training_dataset.time_varying_reals_encoder: [...] (Count: Y)
# - Encoder VSN num_inputs: Z

# KEY QUESTIONS TO ANSWER:
# 1. Is 'encoder_length' in training_dataset.reals? 
#    → YES expected (6 total features)
#
# 2. Is 'encoder_length' in training_dataset.time_varying_reals_encoder?
#    → If YES: baseline DOES use it in encoder VSN (I was wrong)
#    → If NO: baseline excludes it from encoder VSN (I was right)
#
# 3. What is encoder VSN num_inputs?
#    → If 5: confirms encoder_length is excluded
#    → If 6: confirms encoder_length is included
#
# 4. What is flattened_grn.fc1.in_features?
#    → Previous Claude found 400 params = 5 inputs × 16 hidden × 5
#    → This confirms 5 inputs to encoder VSN

print(__doc__)