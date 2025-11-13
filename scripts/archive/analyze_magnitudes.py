#!/usr/bin/env python3
"""
Parse debug output to identify where magnitudes explode.

Usage:
    python train/train_tft_custom.py --debug 2>&1 | tee debug.log
    python analyze_magnitudes.py debug.log
"""

import re
import sys
from collections import defaultdict

def parse_magnitude_line(line):
    """Extract component name and std from debug line."""
    # Match patterns like "  After encoder VSN: mean=X, std=Y"
    match = re.search(r'  (.+?):\s+mean=([-\d.]+),\s+std=([-\d.]+)', line)
    if match:
        component = match.group(1).strip()
        std = float(match.group(3))
        return component, std
    return None, None

def analyze_log(filename):
    """Analyze magnitude tracking from log file."""
    
    # Track stds by epoch and batch
    epoch_data = defaultdict(lambda: defaultdict(list))
    current_epoch = None
    current_batch = None
    
    with open(filename, 'r') as f:
        for line in f:
            # Track epoch
            if '[TRAINING_STEP] Epoch' in line:
                match = re.search(r'Epoch (\d+), Batch (\d+)', line)
                if match:
                    current_epoch = int(match.group(1))
                    current_batch = int(match.group(2))
            
            # Parse magnitude lines
            if current_epoch is not None and '  ' in line and 'mean=' in line and 'std=' in line:
                component, std = parse_magnitude_line(line)
                if component and std is not None:
                    epoch_data[current_epoch][component].append(std)
    
    return epoch_data

def print_summary(epoch_data):
    """Print summary showing where magnitudes grow."""
    
    print("="*80)
    print("MAGNITUDE EXPLOSION ANALYSIS")
    print("="*80)
    
    for epoch in sorted(epoch_data.keys()):
        print(f"\nEPOCH {epoch}:")
        print("-"*80)
        
        # Get component order (from first batch)
        components = list(epoch_data[epoch].keys())
        
        # Calculate stats for each component across all batches in epoch
        for component in components:
            stds = epoch_data[epoch][component]
            if stds:
                avg_std = sum(stds) / len(stds)
                min_std = min(stds)
                max_std = max(stds)
                
                # Flag if exploding (std > 0.5 is suspicious)
                flag = " ⚠️ EXPLODING!" if avg_std > 0.5 else ""
                
                print(f"  {component:45s}: avg={avg_std:7.4f}  "
                      f"min={min_std:7.4f}  max={max_std:7.4f}{flag}")
    
    print("\n" + "="*80)
    print("CRITICAL TRANSITIONS (where std jumps >2x):")
    print("="*80)
    
    # Find biggest jumps between consecutive components
    for epoch in sorted(epoch_data.keys()):
        components = list(epoch_data[epoch].keys())
        print(f"\nEPOCH {epoch}:")
        
        for i in range(len(components) - 1):
            curr_comp = components[i]
            next_comp = components[i + 1]
            
            curr_std = sum(epoch_data[epoch][curr_comp]) / len(epoch_data[epoch][curr_comp])
            next_std = sum(epoch_data[epoch][next_comp]) / len(epoch_data[epoch][next_comp])
            
            if next_std > curr_std * 2 and next_std > 0.1:  # 2x jump and significant magnitude
                ratio = next_std / curr_std if curr_std > 0 else float('inf')
                print(f"  {curr_comp:40s} ({curr_std:.4f})")
                print(f"  └─> {next_comp:37s} ({next_std:.4f})  "
                      f"[{ratio:.1f}x jump] 🔥")

def print_component_progression(epoch_data, epoch=0):
    """Show std progression through network for one epoch."""
    print("\n" + "="*80)
    print(f"COMPONENT-BY-COMPONENT STD PROGRESSION (Epoch {epoch}):")
    print("="*80)
    
    if epoch not in epoch_data:
        print(f"No data for epoch {epoch}")
        return
    
    components = list(epoch_data[epoch].keys())
    for i, component in enumerate(components):
        stds = epoch_data[epoch][component]
        avg_std = sum(stds) / len(stds) if stds else 0
        
        # Visual bar
        bar_length = int(avg_std * 20)  # Scale for visibility
        bar = "█" * min(bar_length, 60)
        
        print(f"{i:2d}. {component:40s} {avg_std:7.4f} {bar}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python analyze_magnitudes.py debug.log")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        epoch_data = analyze_log(filename)
        
        if not epoch_data:
            print("No magnitude tracking data found in log file.")
            print("Make sure you ran with --debug flag.")
            sys.exit(1)
        
        print_summary(epoch_data)
        print_component_progression(epoch_data, epoch=0)
        
        # Also show last epoch if multiple
        if len(epoch_data) > 1:
            last_epoch = max(epoch_data.keys())
            print_component_progression(epoch_data, epoch=last_epoch)
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing log: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
