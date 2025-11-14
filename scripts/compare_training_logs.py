"""
Compare training logs from train_tft.py vs train_tft_custom.py side-by-side.

Extracts and compares:
- Training/validation loss per epoch
- Collapse monitor metrics (prediction diversity, gradient norms, VSN stats, attention entropy)
- Learning rate changes

Usage:
    python compare_training_logs.py test_collapse_old.log test_collapse_new.log
"""

import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path


def parse_collapse_monitor_block(lines, start_idx):
    """
    Parse a single CollapseMonitor epoch block.
    
    Returns dict with metrics and the index after this block.
    """
    metrics = {}
    idx = start_idx
    
    # Extract epoch number from header
    match = re.search(r'\[CollapseMonitor\] Epoch (\d+)', lines[idx])
    if match:
        metrics['epoch'] = int(match.group(1))
    
    idx += 1
    
    # Parse until we hit the "Saved to:" line or next CollapseMonitor block
    while idx < len(lines):
        line = lines[idx]
        
        # End of block
        if 'Saved to:' in line or '[CollapseMonitor]' in line:
            break
            
        # Prediction diversity metrics
        if 'Pred std:' in line:
            # Format: "  Pred std: 0.002114, range: 0.010168, mean: 0.047687"
            std_match = re.search(r'Pred std: ([\d.]+)', line)
            range_match = re.search(r'range: ([\d.]+)', line)
            mean_match = re.search(r'mean: ([\d.]+)', line)
            if std_match:
                metrics['pred_std'] = float(std_match.group(1))
            if range_match:
                metrics['pred_range'] = float(range_match.group(1))
            if mean_match:
                metrics['pred_mean'] = float(mean_match.group(1))
        
        # Positive/negative/unique predictions
        elif 'Pos:' in line and 'Neg:' in line:
            # Format: "  Pos: 100.0%, Neg: 0.0%, Unique: 1123"
            pos_match = re.search(r'Pos: ([\d.]+)%', line)
            neg_match = re.search(r'Neg: ([\d.]+)%', line)
            unique_match = re.search(r'Unique: (\d+)', line)
            if pos_match:
                metrics['pct_positive'] = float(pos_match.group(1))
            if neg_match:
                metrics['pct_negative'] = float(neg_match.group(1))
            if unique_match:
                metrics['num_unique'] = int(unique_match.group(1))
        
        # Gradient norms
        elif 'lstm_encoder:' in line:
            match = re.search(r'lstm_encoder: ([\d.]+)', line)
            if match:
                metrics['grad_lstm_encoder'] = float(match.group(1))
        elif 'lstm_decoder:' in line:
            match = re.search(r'lstm_decoder: ([\d.]+)', line)
            if match:
                metrics['grad_lstm_decoder'] = float(match.group(1))
        elif 'output_layer:' in line:
            match = re.search(r'output_layer: ([\d.]+)', line)
            if match:
                metrics['grad_output_layer'] = float(match.group(1))
        
        # VSN output std
        elif 'encoder:' in line and 'VSN output std' in lines[idx-1]:
            # Previous line was "VSN output std:", this line is "  encoder: 0.123456"
            match = re.search(r'encoder: ([\d.]+)', line)
            if match:
                metrics['vsn_encoder_std'] = float(match.group(1))
            elif 'no data captured' in line:
                metrics['vsn_encoder_std'] = None
        
        # Attention entropy
        elif 'Attention entropy:' in line:
            match = re.search(r'Attention entropy: ([\d.]+)', line)
            if match:
                metrics['attention_entropy'] = float(match.group(1))
            elif 'no data captured' in line:
                metrics['attention_entropy'] = None
        
        idx += 1
    
    return metrics, idx


def parse_log_file(filepath):
    """Parse entire log file and extract metrics."""
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'collapse_metrics': []
    }
    
    i = 0
    current_epoch = None
    current_train_loss = None
    current_val_loss = None
    
    while i < len(lines):
        line = lines[i]
        
        # Training/validation loss - simpler format
        # Format: "Epoch 0: val_loss=0.868480920791626"
        # Format: "Epoch 0: train_loss=0.5701186060905457"
        epoch_match = re.search(r'Epoch (\d+):', line)
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
            
            # Check for train_loss
            train_match = re.search(r'train_loss=([\d.]+)', line)
            if train_match:
                # We have a complete epoch (train loss marks end of epoch)
                current_train_loss = float(train_match.group(1))
                
                # Save the complete epoch data if we have val_loss too
                if current_val_loss is not None:
                    data['epochs'].append(epoch_num)
                    data['train_loss'].append(current_train_loss)
                    data['val_loss'].append(current_val_loss)
                
                # Reset for next epoch
                current_train_loss = None
                current_val_loss = None
                current_epoch = None
            
            # Check for val_loss
            val_match = re.search(r'val_loss=([\d.]+)', line)
            if val_match:
                # Store val_loss (may see it multiple times per epoch, keep last)
                current_val_loss = float(val_match.group(1))
                current_epoch = epoch_num
        
        # Collapse monitor block
        elif '[CollapseMonitor] Epoch' in line:
            metrics, next_idx = parse_collapse_monitor_block(lines, i)
            data['collapse_metrics'].append(metrics)
            i = next_idx - 1  # Will be incremented at end of loop
        
        i += 1
    
    return data


def create_comparison_table(baseline_data, custom_data):
    """Create side-by-side comparison DataFrame."""
    
    # Training/validation loss comparison
    loss_df = pd.DataFrame({
        'epoch': baseline_data['epochs'],
        'baseline_train': baseline_data['train_loss'],
        'custom_train': custom_data['train_loss'],
        'baseline_val': baseline_data['val_loss'],
        'custom_val': custom_data['val_loss'],
    })
    
    # Compute differences
    loss_df['train_diff'] = loss_df['custom_train'] - loss_df['baseline_train']
    loss_df['val_diff'] = loss_df['custom_val'] - loss_df['baseline_val']
    
    # Collapse monitor metrics comparison
    baseline_collapse = pd.DataFrame(baseline_data['collapse_metrics'])
    custom_collapse = pd.DataFrame(custom_data['collapse_metrics'])
    
    if not baseline_collapse.empty and not custom_collapse.empty:
        # Merge on epoch
        collapse_df = baseline_collapse.merge(
            custom_collapse, 
            on='epoch', 
            suffixes=('_baseline', '_custom'),
            how='outer'
        )
    else:
        collapse_df = pd.DataFrame()
    
    return loss_df, collapse_df


def print_loss_comparison(loss_df):
    """Print formatted loss comparison table."""
    print("\n" + "="*100)
    print("TRAINING & VALIDATION LOSS COMPARISON")
    print("="*100)
    print(f"{'Epoch':<6} {'Baseline Train':<15} {'Custom Train':<15} {'Diff':<10} {'Baseline Val':<15} {'Custom Val':<15} {'Diff':<10}")
    print("-"*100)
    
    for _, row in loss_df.iterrows():
        epoch = int(row['epoch'])
        bt = row['baseline_train']
        ct = row['custom_train']
        td = row['train_diff']
        bv = row['baseline_val']
        cv = row['custom_val']
        vd = row['val_diff']
        
        # Flag significant differences
        train_diff_str = f"{td:+.6f}"
        val_diff_str = f"{vd:+.6f}"
        
        if abs(td) > 0.01:
            train_diff_str = f"*{train_diff_str}"
        if abs(vd) > 0.01:
            val_diff_str = f"*{val_diff_str}"
        
        print(f"{epoch:<6} {bt:<15.6f} {ct:<15.6f} {train_diff_str:<10} {bv:<15.6f} {cv:<15.6f} {val_diff_str:<10}")
    
    # Summary statistics
    print("-"*100)
    print(f"{'MEAN':<6} {loss_df['baseline_train'].mean():<15.6f} {loss_df['custom_train'].mean():<15.6f} "
          f"{loss_df['train_diff'].mean():+.6f}    {loss_df['baseline_val'].mean():<15.6f} "
          f"{loss_df['custom_val'].mean():<15.6f} {loss_df['val_diff'].mean():+.6f}")
    print(f"{'STD':<6} {loss_df['baseline_train'].std():<15.6f} {loss_df['custom_train'].std():<15.6f} "
          f"{loss_df['train_diff'].std():+.6f}    {loss_df['baseline_val'].std():<15.6f} "
          f"{loss_df['custom_val'].std():<15.6f} {loss_df['val_diff'].std():+.6f}")


def print_collapse_comparison(collapse_df):
    """Print formatted collapse metrics comparison."""
    if collapse_df.empty:
        print("\n[WARNING] No collapse monitor data found in logs")
        return
    
    print("\n" + "="*120)
    print("COLLAPSE MONITOR METRICS COMPARISON")
    print("="*120)
    
    # Metrics to compare (skip epoch as it's the index)
    metrics = [col.replace('_baseline', '').replace('_custom', '') 
               for col in collapse_df.columns if col.endswith('_baseline')]
    
    for metric in metrics:
        baseline_col = f"{metric}_baseline"
        custom_col = f"{metric}_custom"
        
        if baseline_col not in collapse_df.columns or custom_col not in collapse_df.columns:
            continue
        
        print(f"\n{metric.upper().replace('_', ' ')}:")
        print(f"{'Epoch':<8} {'Baseline':<20} {'Custom':<20} {'Diff':<20} {'% Diff':<15}")
        print("-"*100)
        
        for _, row in collapse_df.iterrows():
            epoch = int(row['epoch'])
            baseline_val = row[baseline_col]
            custom_val = row[custom_col]
            
            # Handle None values
            if baseline_val is None or custom_val is None:
                baseline_str = 'None' if baseline_val is None else f"{baseline_val:.6f}"
                custom_str = 'None' if custom_val is None else f"{custom_val:.6f}"
                print(f"{epoch:<8} {baseline_str:<20} {custom_str:<20} {'N/A':<20} {'N/A':<15}")
            else:
                diff = custom_val - baseline_val
                pct_diff = (diff / baseline_val * 100) if baseline_val != 0 else 0
                
                # Flag significant differences
                flag = ""
                if abs(pct_diff) > 10:
                    flag = "*** "
                elif abs(pct_diff) > 5:
                    flag = "**  "
                
                print(f"{epoch:<8} {baseline_val:<20.6f} {custom_val:<20.6f} {flag}{diff:<+19.6f} {pct_diff:+.2f}%")
        
        # Summary stats for this metric
        baseline_vals = collapse_df[baseline_col].dropna()
        custom_vals = collapse_df[custom_col].dropna()
        
        if not baseline_vals.empty and not custom_vals.empty:
            print("-"*100)
            print(f"{'MEAN':<8} {baseline_vals.mean():<20.6f} {custom_vals.mean():<20.6f} "
                  f"{(custom_vals.mean() - baseline_vals.mean()):<+20.6f} "
                  f"{((custom_vals.mean() - baseline_vals.mean()) / baseline_vals.mean() * 100):+.2f}%")
            print(f"{'STD':<8} {baseline_vals.std():<20.6f} {custom_vals.std():<20.6f}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_training_logs.py <baseline_log> <custom_log>")
        sys.exit(1)
    
    baseline_log = Path(sys.argv[1])
    custom_log = Path(sys.argv[2])
    
    if not baseline_log.exists():
        print(f"Error: {baseline_log} not found")
        sys.exit(1)
    if not custom_log.exists():
        print(f"Error: {custom_log} not found")
        sys.exit(1)
    
    print(f"Parsing baseline log: {baseline_log}")
    baseline_data = parse_log_file(baseline_log)
    
    print(f"Parsing custom log: {custom_log}")
    custom_data = parse_log_file(custom_log)
    
    print(f"\nBaseline: {len(baseline_data['epochs'])} epochs, {len(baseline_data['collapse_metrics'])} collapse monitor entries")
    print(f"Custom:   {len(custom_data['epochs'])} epochs, {len(custom_data['collapse_metrics'])} collapse monitor entries")
    
    # Create comparison tables
    loss_df, collapse_df = create_comparison_table(baseline_data, custom_data)
    
    # Print comparisons
    print_loss_comparison(loss_df)
    print_collapse_comparison(collapse_df)
    
    # Key findings summary
    print("\n" + "="*120)
    print("KEY FINDINGS")
    print("="*120)
    
    # Loss convergence
    if not loss_df.empty:
        final_epoch = int(loss_df['epoch'].iloc[-1])
        final_train_baseline = loss_df['baseline_train'].iloc[-1]
        final_train_custom = loss_df['custom_train'].iloc[-1]
        final_train_diff = loss_df['train_diff'].iloc[-1]
        final_val_baseline = loss_df['baseline_val'].iloc[-1]
        final_val_custom = loss_df['custom_val'].iloc[-1]
        final_val_diff = loss_df['val_diff'].iloc[-1]
        
        print(f"\n1. FINAL LOSS (Epoch {final_epoch}):")
        print(f"   Train: Baseline={final_train_baseline:.6f}, Custom={final_train_custom:.6f}, Diff={final_train_diff:+.6f}")
        print(f"   Val:   Baseline={final_val_baseline:.6f}, Custom={final_val_custom:.6f}, Diff={final_val_diff:+.6f}")
    else:
        print("\n1. FINAL LOSS:")
        print("   [No loss data found in logs]")
    
    # Attention entropy comparison (most relevant to +280 param investigation)
    if 'attention_entropy_baseline' in collapse_df.columns and 'attention_entropy_custom' in collapse_df.columns:
        attn_baseline = collapse_df['attention_entropy_baseline'].dropna()
        attn_custom = collapse_df['attention_entropy_custom'].dropna()
        
        if not attn_baseline.empty and not attn_custom.empty:
            print(f"\n2. ATTENTION ENTROPY (relates to +280 param gap):")
            print(f"   Baseline: Mean={attn_baseline.mean():.6f}, Std={attn_baseline.std():.6f}")
            print(f"   Custom:   Mean={attn_custom.mean():.6f}, Std={attn_custom.std():.6f}")
            print(f"   Difference: {(attn_custom.mean() - attn_baseline.mean()):+.6f} ({((attn_custom.mean() - attn_baseline.mean()) / attn_baseline.mean() * 100):+.2f}%)")
            
            if abs((attn_custom.mean() - attn_baseline.mean()) / attn_baseline.mean()) > 0.05:
                print(f"   [SIGNIFICANT DIFFERENCE] May indicate different attention mechanism behavior")
    
    # VSN activity
    if 'vsn_encoder_std_baseline' in collapse_df.columns and 'vsn_encoder_std_custom' in collapse_df.columns:
        vsn_baseline = collapse_df['vsn_encoder_std_baseline'].dropna()
        vsn_custom = collapse_df['vsn_encoder_std_custom'].dropna()
        
        if not vsn_baseline.empty and not vsn_custom.empty:
            print(f"\n3. VSN ENCODER ACTIVITY:")
            print(f"   Baseline: Mean={vsn_baseline.mean():.6f}, Std={vsn_baseline.std():.6f}")
            print(f"   Custom:   Mean={vsn_custom.mean():.6f}, Std={vsn_custom.std():.6f}")
            print(f"   Difference: {(vsn_custom.mean() - vsn_baseline.mean()):+.6f} ({((vsn_custom.mean() - vsn_baseline.mean()) / vsn_baseline.mean() * 100):+.2f}%)")
    
    print("\n" + "="*120)


if __name__ == '__main__':
    main()
