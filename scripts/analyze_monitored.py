"""
Analyze collapse monitoring data from experiments.

IMPORTANT - WORKFLOW AND PURPOSE:
==================================
This script analyzes TRAINING DYNAMICS captured during model training via the
CollapseMonitor callback. It shows HOW and WHEN collapse emerges during training
by examining validation set predictions, gradients, and attention patterns.

This is complementary to, NOT a replacement for, evaluate_tft.py:

WORKFLOW:
---------
1. train_tft.py          → Trains model, CollapseMonitor logs training dynamics
2. evaluate_tft.py       → Evaluates final model on TEST SET (unseen data)
3. analyze_monitored.py  → Analyzes TRAINING DYNAMICS to understand collapse

WHAT EACH TOOL DOES:
--------------------
- train_tft.py:
  * Trains TFT model
  * CollapseMonitor callback logs validation predictions, gradients, VSN weights
  * Saves: experiments/{name}/collapse_monitoring/*.json
  
- evaluate_tft.py:
  * Loads trained checkpoint
  * Generates predictions on TEST SET
  * Computes final metrics (accuracy, Sharpe, AUC-ROC)
  * Creates quality diagnostic plots for test period
  * Saves: experiments/{name}/evaluation/*
  * Answers: "How good is the final model?"
  
- analyze_monitored.py (this script):
  * Loads collapse_monitoring/*.json (from training)
  * Compares training dynamics between experiments
  * Shows when/how collapse emerged during training
  * Analyzes gradient flow and feature importance
  * Answers: "What happened during training to cause collapse?"

KEY DISTINCTION:
----------------
- CollapseMonitor tracks VALIDATION SET during training (epochs 0-N)
- evaluate_tft.py analyzes TEST SET after training (final model performance)
- Both are important: training dynamics explain WHY final test behavior occurs

USAGE MODES:
------------
1. Pair comparison: Compare two specific experiments (e.g., baseline vs staleness)
2. Group analysis: Analyze multiple experiments matching a pattern

Examples:
    # Compare two experiments
    python analyze_monitored.py --mode pair \\
        --experiments baseline_h16_monitored staleness_h16_monitored \\
        --output results/monitoring_comparison.png
    
    # Analyze all monitored experiments
    python analyze_monitored.py --mode group \\
        --pattern "*_monitored" \\
        --output results/monitoring_group_analysis.png
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import seaborn as sns


# ============================================================================
# Data Loading
# ============================================================================

def load_monitoring_data(exp_name):
    """Load collapse monitoring data for an experiment."""
    monitor_path = Path(f'experiments/{exp_name}/collapse_monitoring/collapse_monitor_latest.json')
    
    if not monitor_path.exists():
        print(f"Warning: No monitoring data found for {exp_name}")
        return None
        
    with open(monitor_path, 'r') as f:
        data = json.load(f)
    
    return data


def load_experiment_config(exp_name):
    """Load experiment configuration."""
    config_path = Path(f'experiments/{exp_name}/config.json')
    
    if not config_path.exists():
        return None
        
    with open(config_path, 'r') as f:
        return json.load(f)


# ============================================================================
# Analysis Functions
# ============================================================================

def compute_unidirectional_threshold(pct_positive, threshold=98):
    """Detect when model becomes unidirectional during training."""
    epochs_unidirectional = [e for e, p in enumerate(pct_positive) 
                             if p >= threshold or p <= (100 - threshold)]
    
    if epochs_unidirectional:
        return epochs_unidirectional[0]
    return None


def extract_vsn_feature_importance(monitor_data, feature_name):
    """Extract VSN output statistics for a specific feature."""
    vsn_outputs = monitor_data.get('vsn_output_std', {})
    
    if feature_name in vsn_outputs:
        return vsn_outputs[feature_name]
    
    # Try partial match (e.g., 'days_since' matches 'days_since_CPI_update')
    for key in vsn_outputs.keys():
        if feature_name in key:
            return vsn_outputs[key]
    
    return None


def compute_gradient_divergence(baseline_grads, staleness_grads):
    """Compare gradient magnitudes between experiments."""
    divergence = {}
    
    for layer_name in baseline_grads.keys():
        if layer_name in staleness_grads:
            base_vals = np.array(baseline_grads[layer_name])
            stale_vals = np.array(staleness_grads[layer_name])
            
            # Compute ratio of average gradient norms
            if len(base_vals) > 0 and len(stale_vals) > 0:
                divergence[layer_name] = np.mean(stale_vals) / (np.mean(base_vals) + 1e-10)
    
    return divergence


# ============================================================================
# Pair Comparison Mode
# ============================================================================

def compare_pair(exp1_name, exp2_name, output_path):
    """Compare two experiments in detail."""
    print(f"\n{'='*80}")
    print(f"TRAINING DYNAMICS COMPARISON: {exp1_name} vs {exp2_name}")
    print(f"{'='*80}")
    print("\nNOTE: These metrics reflect VALIDATION SET predictions during training,")
    print("      NOT final test set performance.")
    print("\n      For test set results, see: experiments/{exp_name}/evaluation/")
    print(f"{'='*80}\n")
    
    # Load data
    data1 = load_monitoring_data(exp1_name)
    data2 = load_monitoring_data(exp2_name)
    config1 = load_experiment_config(exp1_name)
    config2 = load_experiment_config(exp2_name)
    
    if data1 is None or data2 is None:
        print("Error: Could not load monitoring data for both experiments")
        return
    
    # Convert to DataFrames
    df1 = pd.DataFrame({
        'epoch': data1['epoch'],
        'pred_std': data1['prediction_std'],
        'pred_mean': data1['prediction_mean'],
        'pct_positive': data1['pct_positive'],
        'num_unique': data1['num_unique_predictions'],
        'attention_entropy': data1.get('attention_entropy', [None]*len(data1['epoch']))
    })
    
    df2 = pd.DataFrame({
        'epoch': data2['epoch'],
        'pred_std': data2['prediction_std'],
        'pred_mean': data2['prediction_mean'],
        'pct_positive': data2['pct_positive'],
        'num_unique': data2['num_unique_predictions'],
        'attention_entropy': data2.get('attention_entropy', [None]*len(data2['epoch']))
    })
    
    # Detect key events
    unidirectional_epoch_1 = compute_unidirectional_threshold(df1['pct_positive'])
    unidirectional_epoch_2 = compute_unidirectional_threshold(df2['pct_positive'])
    
    # Print summary statistics
    print("VALIDATION SET - TRAINING DYNAMICS SUMMARY:")
    print("-" * 80)
    print(f"{'Metric':<30} {exp1_name[:20]:>20} {exp2_name[:20]:>20}")
    print("-" * 80)
    print(f"{'Mean pred_std':<30} {df1['pred_std'].mean():>20.4f} {df2['pred_std'].mean():>20.4f}")
    print(f"{'Max pred_std':<30} {df1['pred_std'].max():>20.4f} {df2['pred_std'].max():>20.4f}")
    print(f"{'Mean pct_positive':<30} {df1['pct_positive'].mean():>20.1f}% {df2['pct_positive'].mean():>20.1f}%")
    print(f"{'Std pct_positive':<30} {df1['pct_positive'].std():>20.1f}% {df2['pct_positive'].std():>20.1f}%")
    print(f"{'Epochs at 100% pos':<30} {(df1['pct_positive'] >= 98).sum():>20} {(df2['pct_positive'] >= 98).sum():>20}")
    print(f"{'Unidirectional at epoch':<30} {str(unidirectional_epoch_1):>20} {str(unidirectional_epoch_2):>20}")
    print("-" * 80)
    print("\nINTERPRETATION:")
    print("  - Unidirectional = Model predicting >98% same direction on validation set")
    print("  - This shows WHEN collapse emerged during training")
    print("  - See evaluate_tft.py output for final test set performance")
    print("-" * 80)
    
    # VSN analysis (if staleness features present)
    vsn_staleness_1 = extract_vsn_feature_importance(data1, 'days_since')
    vsn_staleness_2 = extract_vsn_feature_importance(data2, 'days_since')
    
    if vsn_staleness_2 is not None:
        print(f"\nVSN STALENESS FEATURE IMPORTANCE (final training epoch):")
        print(f"  {exp2_name}: std = {vsn_staleness_2[-1] if vsn_staleness_2 else 'N/A'}")
        print("  (Higher std = feature has more variable importance across samples)")
    
    # Create comparison plots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Prediction diversity metrics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df1['epoch'], df1['pred_std'], 'o-', label=exp1_name, alpha=0.7)
    ax1.plot(df2['epoch'], df2['pred_std'], 's-', label=exp2_name, alpha=0.7)
    ax1.axhline(0.05, color='red', linestyle='--', alpha=0.3, label='Collapse threshold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Prediction Std Dev')
    ax1.set_title('Prediction Diversity\n(Validation Set)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df1['epoch'], df1['pct_positive'], 'o-', label=exp1_name, alpha=0.7)
    ax2.plot(df2['epoch'], df2['pct_positive'], 's-', label=exp2_name, alpha=0.7)
    ax2.axhline(98, color='red', linestyle='--', alpha=0.3, label='Unidirectional threshold')
    ax2.axhline(2, color='red', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('% Positive Predictions')
    ax2.set_title('Directional Bias\n(Validation Set)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(df1['epoch'], df1['num_unique'], 'o-', label=exp1_name, alpha=0.7)
    ax3.plot(df2['epoch'], df2['num_unique'], 's-', label=exp2_name, alpha=0.7)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Unique Predictions')
    ax3.set_title('Prediction Variety\n(Validation Set)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # Log scale to better show when predictions become identical (value = 1)
    ax3.set_yscale('log')
    ax3.set_ylim([0.9, max(df1['num_unique'].max(), df2['num_unique'].max()) * 1.1])
    
    # Row 2: Prediction magnitude
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(df1['epoch'], df1['pred_mean'], 'o-', label=exp1_name, alpha=0.7)
    ax4.plot(df2['epoch'], df2['pred_mean'], 's-', label=exp2_name, alpha=0.7)
    ax4.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Mean Prediction')
    ax4.set_title('Prediction Mean (Bias)\n(Validation Set)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Attention entropy
    ax5 = fig.add_subplot(gs[1, 1])
    if df1['attention_entropy'].notna().any():
        ax5.plot(df1['epoch'], df1['attention_entropy'], 'o-', label=exp1_name, alpha=0.7)
    if df2['attention_entropy'].notna().any():
        ax5.plot(df2['epoch'], df2['attention_entropy'], 's-', label=exp2_name, alpha=0.7)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Attention Entropy')
    ax5.set_title('Attention Concentration\n(Validation Set)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # VSN staleness feature (if available)
    ax6 = fig.add_subplot(gs[1, 2])
    if vsn_staleness_2 is not None and len(vsn_staleness_2) > 0:
        ax6.plot(data2['epoch'], vsn_staleness_2, 's-', label=exp2_name, alpha=0.7, color='C1')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('VSN Output Std')
        ax6.set_title('Staleness Feature Importance\n(Variable Selection Network)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No staleness\nfeature data', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Staleness Feature Importance\n(Not available)')
    
    # Row 3: Gradient analysis
    ax7 = fig.add_subplot(gs[2, :])
    
    # Compare gradient norms for key layers
    grad_norms_1 = data1.get('gradient_norms', {})
    grad_norms_2 = data2.get('gradient_norms', {})
    
    # Find common layers and plot a subset (sorted for deterministic ordering)
    common_layers = sorted(set(grad_norms_1.keys()) & set(grad_norms_2.keys()))
    interesting_layers = [l for l in common_layers if any(x in l.lower() 
                          for x in ['variable_selection', 'gate', 'attention', 'output'])][:5]
    
    if interesting_layers:
        x_pos = np.arange(len(interesting_layers))
        width = 0.35
        
        # Get final epoch gradient norms
        vals_1 = [np.mean(grad_norms_1[layer][-1:]) if len(grad_norms_1[layer]) > 0 else 0 
                  for layer in interesting_layers]
        vals_2 = [np.mean(grad_norms_2[layer][-1:]) if len(grad_norms_2[layer]) > 0 else 0 
                  for layer in interesting_layers]
        
        ax7.bar(x_pos - width/2, vals_1, width, label=exp1_name, alpha=0.7)
        ax7.bar(x_pos + width/2, vals_2, width, label=exp2_name, alpha=0.7)
        ax7.set_xlabel('Layer')
        ax7.set_ylabel('Gradient Norm (final epoch)')
        ax7.set_title('Gradient Flow Comparison (Final Training Epoch)')
        ax7.set_xticks(x_pos)
        ax7.set_xticklabels([l.split('.')[-1][:20] for l in interesting_layers], 
                            rotation=45, ha='right')
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')
    else:
        ax7.text(0.5, 0.5, 'No gradient data available', 
                ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Gradient Flow Comparison')
    
    plt.suptitle(f'Training Dynamics Comparison: {exp1_name} vs {exp2_name}\n' +
                 '(Validation Set During Training - NOT Test Set)', 
                 fontsize=14, fontweight='bold')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot: {output_path}")
    plt.close()
    
    return df1, df2


# ============================================================================
# Group Analysis Mode
# ============================================================================

def analyze_group(pattern, output_path):
    """Analyze multiple experiments matching a pattern."""
    print(f"\n{'='*80}")
    print(f"GROUP ANALYSIS: pattern='{pattern}'")
    print(f"{'='*80}")
    print("\nNOTE: These metrics reflect VALIDATION SET predictions during training,")
    print("      NOT final test set performance.")
    print(f"{'='*80}\n")
    
    experiments_dir = Path('experiments')
    
    # Find matching experiments
    matching_exps = []
    for exp_dir in sorted(experiments_dir.glob(pattern)):
        exp_name = exp_dir.name
        monitor_path = exp_dir / 'collapse_monitoring' / 'collapse_monitor_latest.json'
        
        if monitor_path.exists():
            matching_exps.append(exp_name)
    
    if len(matching_exps) == 0:
        print(f"No experiments found matching pattern: {pattern}")
        return
    
    print(f"Found {len(matching_exps)} experiments with monitoring data:\n")
    
    # Collect summary statistics
    results = []
    for exp_name in matching_exps:
        data = load_monitoring_data(exp_name)
        config = load_experiment_config(exp_name)
        
        if data is None or config is None:
            continue
        
        final_idx = -1
        unidirectional_epoch = compute_unidirectional_threshold(data['pct_positive'])
        
        result = {
            'experiment': exp_name,
            'hidden_size': config['architecture']['hidden_size'],
            'dropout': config['architecture']['dropout'],
            'has_staleness': 'staleness' in config.get('features', {}).get('staleness', []),
            'final_pred_std': data['prediction_std'][final_idx],
            'final_pct_positive': data['pct_positive'][final_idx],
            'final_num_unique': data['num_unique_predictions'][final_idx],
            'unidirectional_epoch': unidirectional_epoch,
            'is_unidirectional': unidirectional_epoch is not None,
            'total_epochs': len(data['epoch'])
        }
        
        results.append(result)
        
        # Print summary line
        print(f"  {exp_name[:40]:<40} "
              f"std={result['final_pred_std']:.4f}  "
              f"pos={result['final_pct_positive']:>5.1f}%  "
              f"unidirectional={'YES' if result['is_unidirectional'] else 'NO'}")
    
    df = pd.DataFrame(results)
    
    # Generate group visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Final std by experiment
    ax = axes[0, 0]
    colors = ['red' if u else 'green' for u in df['is_unidirectional']]
    ax.bar(range(len(df)), df['final_pred_std'], color=colors, alpha=0.7)
    ax.axhline(0.05, color='orange', linestyle='--', label='Collapse threshold')
    ax.set_xlabel('Experiment Index')
    ax.set_ylabel('Final Prediction Std')
    ax.set_title('Prediction Diversity\n(Red = Unidirectional on Validation)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Directional bias
    ax = axes[0, 1]
    ax.bar(range(len(df)), df['final_pct_positive'], color=colors, alpha=0.7)
    ax.axhline(98, color='orange', linestyle='--', label='Unidirectional threshold')
    ax.set_xlabel('Experiment Index')
    ax.set_ylabel('% Positive Predictions')
    ax.set_title('Directional Bias (Validation Set)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Staleness vs baseline comparison
    if 'has_staleness' in df.columns:
        ax = axes[1, 0]
        staleness_df = df[df['has_staleness']]
        baseline_df = df[~df['has_staleness']]
        
        if len(staleness_df) > 0 and len(baseline_df) > 0:
            x_pos = np.arange(2)
            means = [baseline_df['final_pct_positive'].mean(), 
                    staleness_df['final_pct_positive'].mean()]
            stds = [baseline_df['final_pct_positive'].std(), 
                   staleness_df['final_pct_positive'].std()]
            
            ax.bar(x_pos, means, yerr=stds, alpha=0.7, capsize=10)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(['Baseline', 'Staleness'])
            ax.set_ylabel('% Positive Predictions')
            ax.set_title('Staleness Effect on Directional Bias\n(Validation Set)')
            ax.axhline(98, color='red', linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary table as text
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"Total experiments: {len(df)}\n"
    summary_text += f"Unidirectional: {df['is_unidirectional'].sum()} ({df['is_unidirectional'].mean()*100:.1f}%)\n\n"
    summary_text += f"Avg final std: {df['final_pred_std'].mean():.4f}\n"
    summary_text += f"Avg pct positive: {df['final_pct_positive'].mean():.1f}%\n\n"
    
    if 'has_staleness' in df.columns:
        baseline_count = (~df['has_staleness']).sum()
        staleness_count = df['has_staleness'].sum()
        summary_text += f"Baseline exps: {baseline_count}\n"
        summary_text += f"Staleness exps: {staleness_count}\n"
    
    summary_text += "\nNOTE: Validation set metrics\n"
    summary_text += "See evaluate_tft.py for test results"
    
    ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
           verticalalignment='center', transform=ax.transAxes)
    
    plt.suptitle(f'Training Dynamics - Group Analysis: {pattern}\n(Validation Set During Training)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved group analysis: {output_path}")
    plt.close()
    
    # Save results CSV
    csv_path = output_path.replace('.png', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved results table: {csv_path}")
    
    return df


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze collapse monitoring data from training dynamics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WORKFLOW REMINDER:
------------------
1. train_tft.py          → Trains model, logs training dynamics
2. evaluate_tft.py       → Evaluates final model on test set
3. analyze_monitored.py  → Analyzes training dynamics (this script)

This script analyzes VALIDATION SET behavior during training.
For final TEST SET performance, see evaluate_tft.py outputs.
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['pair', 'group'],
                       help='Analysis mode: pair (compare 2 experiments) or group (analyze multiple)')
    
    parser.add_argument('--experiments', type=str, nargs='+',
                       help='Experiment names (required for pair mode, exactly 2 names)')
    
    parser.add_argument('--pattern', type=str,
                       help='Glob pattern to match experiments (required for group mode)')
    
    parser.add_argument('--output', type=str, default='results/monitoring_analysis.png',
                       help='Output path for plots')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'pair':
        if not args.experiments or len(args.experiments) != 2:
            print("Error: --mode pair requires exactly 2 experiment names via --experiments")
            return
        
        compare_pair(args.experiments[0], args.experiments[1], args.output)
    
    elif args.mode == 'group':
        if not args.pattern:
            print("Error: --mode group requires --pattern")
            return
        
        analyze_group(args.pattern, args.output)


if __name__ == '__main__':
    main()
