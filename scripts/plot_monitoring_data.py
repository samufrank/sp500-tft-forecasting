"""
Plotting utility for collapse monitoring data.

Creates publication-quality figures from monitoring logs.

Usage:
    python scripts/plot_monitoring_data.py monitor_h16 monitor_h24
    python scripts/plot_monitoring_data.py --output-dir figures/ monitor_*
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


def load_monitoring_log(experiment_name):
    """Load collapse monitoring data."""
    log_path = Path(f'experiments/{experiment_name}/collapse_monitoring/collapse_monitor_latest.json')
    
    if not log_path.exists():
        return None
    
    with open(log_path, 'r') as f:
        return json.load(f)


def plot_prediction_diversity(experiments, output_path='prediction_diversity.png'):
    """Plot prediction std over training for multiple experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(experiments)))
    
    for idx, exp_name in enumerate(experiments):
        data = load_monitoring_log(exp_name)
        if data is None:
            print(f"Warning: No data for {exp_name}")
            continue
        
        epochs = data['epoch']
        pred_std = data['prediction_std']
        pred_range = data['prediction_range']
        
        # Determine if collapsed
        collapsed = pred_std[-1] < 0.05
        linestyle = '--' if collapsed else '-'
        marker = 'x' if collapsed else 'o'
        
        label = exp_name.replace('capacity_', '').replace('monitor_', '')
        
        # Plot std
        ax1.plot(epochs, pred_std, linestyle=linestyle, marker=marker,
                markersize=4, alpha=0.7, color=colors[idx], label=label)
        
        # Plot range
        ax2.plot(epochs, pred_range, linestyle=linestyle, marker=marker,
                markersize=4, alpha=0.7, color=colors[idx], label=label)
    
    # Formatting
    ax1.axhline(y=0.05, color='red', linestyle=':', linewidth=2, 
               label='Collapse threshold', alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Prediction Std Dev', fontsize=12)
    ax1.set_title('Prediction Diversity Over Training', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Prediction Range', fontsize=12)
    ax2.set_title('Prediction Range Over Training', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_gradient_flow(experiments, output_path='gradient_flow.png'):
    """Plot gradient norms over training."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Key layers to track
    layer_groups = {
        'LSTM Encoder': 'lstm_encoder',
        'LSTM Decoder': 'lstm_decoder',
        'Attention': 'multihead_attn',
        'Output Layer': 'output_layer',
    }
    
    for idx, (group_name, layer_key) in enumerate(layer_groups.items()):
        ax = axes[idx]
        
        for exp_name in experiments:
            data = load_monitoring_log(exp_name)
            if data is None:
                continue
            
            epochs = data['epoch']
            
            # Find all gradient norms for this layer group
            matching_layers = [k for k in data['gradient_norms'].keys() 
                             if layer_key in k]
            
            if not matching_layers:
                continue
            
            # Average across all params in this layer
            avg_norms = []
            for epoch_idx in range(len(epochs)):
                norms = [data['gradient_norms'][layer][epoch_idx] 
                        for layer in matching_layers
                        if epoch_idx < len(data['gradient_norms'][layer])]
                if norms:
                    avg_norms.append(np.mean(norms))
                else:
                    avg_norms.append(np.nan)
            
            # Determine if collapsed
            collapsed = data['prediction_std'][-1] < 0.05
            linestyle = '--' if collapsed else '-'
            
            label = exp_name.replace('capacity_', '').replace('monitor_', '')
            ax.plot(epochs[:len(avg_norms)], avg_norms, linestyle=linestyle,
                   alpha=0.7, label=label)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title(f'{group_name}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.suptitle('Gradient Flow Analysis', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_sign_distribution(experiments, output_path='sign_distribution.png'):
    """Plot evolution of prediction sign distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for exp_name in experiments:
        data = load_monitoring_log(exp_name)
        if data is None:
            continue
        
        epochs = data['epoch']
        pct_pos = data['pct_positive']
        pct_neg = data['pct_negative']
        
        collapsed = data['prediction_std'][-1] < 0.05
        linestyle = '--' if collapsed else '-'
        marker = 'x' if collapsed else 'o'
        
        label = exp_name.replace('capacity_', '').replace('monitor_', '')
        
        ax1.plot(epochs, pct_pos, linestyle=linestyle, marker=marker,
                markersize=4, alpha=0.7, label=label)
        ax2.plot(epochs, pct_neg, linestyle=linestyle, marker=marker,
                markersize=4, alpha=0.7, label=label)
    
    # Reference lines
    ax1.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax1.axhline(y=95, color='red', linestyle=':', alpha=0.5, 
               label='Collapse threshold')
    ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=95, color='red', linestyle=':', alpha=0.5,
               label='Collapse threshold')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('% Positive Predictions', fontsize=12)
    ax1.set_title('Positive Predictions Over Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('% Negative Predictions', fontsize=12)
    ax2.set_title('Negative Predictions Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_comprehensive_summary(experiments, output_path='comprehensive_summary.png'):
    """Create a comprehensive 6-panel summary figure."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Prediction std
    ax2 = fig.add_subplot(gs[0, 1])  # Prediction range
    ax3 = fig.add_subplot(gs[0, 2])  # Num unique
    ax4 = fig.add_subplot(gs[1, 0])  # % Positive
    ax5 = fig.add_subplot(gs[1, 1])  # % Negative  
    ax6 = fig.add_subplot(gs[1, 2])  # Summary stats
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(experiments)))
    
    summary_data = []
    
    for idx, exp_name in enumerate(experiments):
        data = load_monitoring_log(exp_name)
        if data is None:
            continue
        
        epochs = data['epoch']
        collapsed = data['prediction_std'][-1] < 0.05
        linestyle = '--' if collapsed else '-'
        marker = 'x' if collapsed else 'o'
        label = exp_name.replace('capacity_', '').replace('monitor_', '')[:20]
        
        # Plot all metrics
        ax1.plot(epochs, data['prediction_std'], linestyle=linestyle, 
                marker=marker, markersize=3, alpha=0.7, color=colors[idx], label=label)
        ax2.plot(epochs, data['prediction_range'], linestyle=linestyle,
                marker=marker, markersize=3, alpha=0.7, color=colors[idx])
        ax3.plot(epochs, data['num_unique_predictions'], linestyle=linestyle,
                marker=marker, markersize=3, alpha=0.7, color=colors[idx])
        ax4.plot(epochs, data['pct_positive'], linestyle=linestyle,
                marker=marker, markersize=3, alpha=0.7, color=colors[idx])
        ax5.plot(epochs, data['pct_negative'], linestyle=linestyle,
                marker=marker, markersize=3, alpha=0.7, color=colors[idx])
        
        # Collect summary data
        summary_data.append({
            'experiment': label,
            'final_std': data['prediction_std'][-1],
            'collapsed': collapsed,
            'collapse_epoch': next((i for i, s in enumerate(data['prediction_std']) 
                                  if s < 0.05), None) if collapsed else None
        })
    
    # Format subplots
    ax1.axhline(y=0.05, color='red', linestyle=':', alpha=0.7)
    ax1.set_ylabel('Prediction Std', fontsize=10)
    ax1.set_title('Prediction Diversity', fontweight='bold')
    ax1.legend(fontsize=7, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.set_ylabel('Prediction Range', fontsize=10)
    ax2.set_title('Value Range', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    ax3.set_ylabel('Unique Predictions', fontsize=10)
    ax3.set_title('Prediction Uniqueness', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    ax4.axhline(y=95, color='red', linestyle=':', alpha=0.7)
    ax4.set_xlabel('Epoch', fontsize=10)
    ax4.set_ylabel('% Positive', fontsize=10)
    ax4.set_title('Positive Predictions', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    ax5.axhline(y=95, color='red', linestyle=':', alpha=0.7)
    ax5.set_xlabel('Epoch', fontsize=10)
    ax5.set_ylabel('% Negative', fontsize=10)
    ax5.set_title('Negative Predictions', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Summary table
    ax6.axis('off')
    ax6.text(0.5, 0.95, 'Collapse Summary', ha='center', va='top',
            fontsize=12, fontweight='bold', transform=ax6.transAxes)
    
    table_text = "Experiment          Final Std  Status\n"
    table_text += "-" * 45 + "\n"
    for item in summary_data:
        status = "COLLAPSED" if item['collapsed'] else "Working"
        table_text += f"{item['experiment']:<20} {item['final_std']:.4f}  {status}\n"
    
    ax6.text(0.1, 0.85, table_text, ha='left', va='top',
            fontsize=8, family='monospace', transform=ax6.transAxes)
    
    fig.suptitle('Collapse Investigation - Comprehensive Summary', 
                fontsize=16, fontweight='bold')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot collapse monitoring data')
    parser.add_argument('experiments', nargs='+', help='Experiment names')
    parser.add_argument('--output-dir', type=str, default='experiments',
                       help='Directory for output plots')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Generate comprehensive summary figure')
    
    args = parser.parse_args()
    
    # Expand wildcards
    experiments = []
    for pattern in args.experiments:
        if '*' in pattern:
            exp_dir = Path('experiments')
            matches = [p.name for p in exp_dir.glob(pattern) if p.is_dir()]
            experiments.extend(sorted(matches))
        else:
            experiments.append(pattern)
    
    if not experiments:
        print("No experiments found")
        return
    
    print(f"Plotting {len(experiments)} experiments...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    if args.comprehensive:
        plot_comprehensive_summary(experiments, 
                                   output_dir / 'collapse_comprehensive.png')
    else:
        plot_prediction_diversity(experiments, 
                                 output_dir / 'collapse_diversity.png')
        plot_sign_distribution(experiments, 
                             output_dir / 'collapse_signs.png')
        plot_gradient_flow(experiments, 
                         output_dir / 'collapse_gradients.png')
    
    print("\nDone! Check output in:", output_dir)


if __name__ == '__main__':
    main()

