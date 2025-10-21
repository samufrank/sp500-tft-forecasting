#!/usr/bin/env python3
"""
Summarize all experiments into CSV files for analysis.

Usage:
    python scripts/summarize_experiments.py --all
    python scripts/summarize_experiments.py --group sweep2
    python scripts/summarize_experiments.py exp001 exp002
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
from datetime import datetime


def infer_group(exp_name):
    """Infer experiment group from name."""
    if exp_name.startswith('sweep2'):
        return 'sweep2'
    elif exp_name.startswith('sweep'):
        return 'sweep1'
    elif exp_name.startswith('capacity'):
        return 'capacity'
    elif exp_name.startswith('exp'):
        return 'early_exp'
    else:
        return 'other'


def extract_experiment_data(exp_path):
    """Extract data from one experiment directory."""
    exp_name = exp_path.name
    
    data = {
        'experiment_name': exp_name,
        'group': infer_group(exp_name),
    }
    
    # Read config.json
    config_path = exp_path / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        # Extract key hyperparameters
        data['hidden_size'] = config.get('architecture', {}).get('hidden_size')
        data['dropout'] = config.get('architecture', {}).get('dropout')
        data['learning_rate'] = config.get('training', {}).get('learning_rate')
        data['encoder_length'] = config.get('architecture', {}).get('max_encoder_length')
        data['attention_heads'] = config.get('architecture', {}).get('attention_head_size')
        data['batch_size'] = config.get('training', {}).get('batch_size')
        data['created_at'] = config.get('created_at')
    
    # Read final_metrics.json
    final_path = exp_path / 'final_metrics.json'
    if final_path.exists():
        with open(final_path) as f:
            final = json.load(f)
        
        data['best_val_loss'] = final.get('best_val_loss')
        data['total_epochs'] = final.get('total_epochs')
        data['early_stopped'] = final.get('early_stopped')
    
    # Read evaluation_metrics.json if exists
    eval_path = exp_path / 'evaluation' / 'evaluation_metrics.json'
    if eval_path.exists():
        with open(eval_path) as f:
            eval_data = json.load(f)
        
        # Statistical metrics
        stat = eval_data.get('statistical_metrics', {})
        data['test_mse'] = stat.get('mse')
        data['test_rmse'] = stat.get('rmse')
        data['test_mae'] = stat.get('mae')
        data['test_r2'] = stat.get('r2')
        
        # Financial metrics
        fin = eval_data.get('financial_metrics', {})
        data['dir_acc'] = fin.get('directional_accuracy')
        data['sharpe_ratio'] = fin.get('sharpe_ratio')
        data['max_drawdown'] = fin.get('max_drawdown')
        data['total_return'] = fin.get('total_return')
        data['num_trades'] = fin.get('num_trades')
        data['precision'] = fin.get('precision')
        data['recall'] = fin.get('recall')
        data['f1_score'] = fin.get('f1_score')
        data['auc_roc'] = fin.get('auc_roc')
        data['alpha'] = fin.get('alpha')
        
        data['evaluated'] = True
    else:
        data['evaluated'] = False
    
    # Read collapse diagnosis if exists (from diagnose_existing_models.py)
    diagnosis_path = exp_path / 'collapse_diagnosis.json'
    if diagnosis_path.exists():
        with open(diagnosis_path) as f:
            diag = json.load(f)
        
        data['collapsed'] = diag.get('collapsed')
        data['collapse_type'] = diag.get('collapse_type')
        # Optionally extract prediction stats if needed
        if 'predictions' in diag and diag['predictions']:
            data['pred_std'] = diag['predictions'].get('std')
            data['pred_mean'] = diag['predictions'].get('mean')
    else:
        # No diagnosis available
        data['collapsed'] = None
        data['collapse_type'] = None
        
    """
    # TODO: implement real time collapse monitoring metrics
    if (collapse_monitoring/collapse_monitor_latest.json exists):
        read that, extract final pred_std and collapsed flag
    elif (collapse_diagnosis.json exists):
        read that
    else:
        collapsed = None
    """
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Summarize experiment results')
    parser.add_argument('experiments', nargs='*', help='Specific experiment names')
    parser.add_argument('--all', action='store_true', help='Process all experiments')
    parser.add_argument('--group', type=str, help='Process specific group (sweep2, capacity, etc)')
    parser.add_argument('--output', type=str, default='experiments_summary.csv', 
                       help='Output CSV filename')
    parser.add_argument('--evaluated-only', action='store_true',
                       help='Only include experiments with evaluation results')
    args = parser.parse_args()
    
    exp_dir = Path('experiments')
    if not exp_dir.exists():
        print(f"Error: {exp_dir} directory not found")
        return
    
    # Determine which experiments to process
    if args.all:
        exp_paths = [p for p in exp_dir.iterdir() if p.is_dir()]
    elif args.group:
        exp_paths = [p for p in exp_dir.iterdir() if p.is_dir() and infer_group(p.name) == args.group]
    elif args.experiments:
        exp_paths = [exp_dir / name for name in args.experiments]
    else:
        print("Error: Must specify --all, --group, or experiment names")
        return
    
    # Extract data from each experiment
    print(f"\nScanning {len(exp_paths)} experiments...")
    results = []
    for exp_path in sorted(exp_paths):
        if not exp_path.exists():
            print(f"Warning: {exp_path.name} not found, skipping")
            continue
        
        try:
            data = extract_experiment_data(exp_path)
            
            # Skip if --evaluated-only and no evaluation
            if args.evaluated_only and not data.get('evaluated'):
                continue
            
            results.append(data)
            status = " evaluated" if data.get('evaluated') else " not evaluated"
            collapsed = " (COLLAPSED)" if data.get('collapsed') else ""
            print(f"  {exp_path.name}: {status}{collapsed}")
        except Exception as e:
            print(f"  {exp_path.name}: ERROR - {e}")
    
    if not results:
        print("\nNo experiments processed")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by group, then by experiment name
    df = df.sort_values(['group', 'experiment_name'])
    
    # Save full results
    df.to_csv(args.output, index=False)
    print(f"\n Saved {len(df)} experiments to {args.output}")
    
    # Also create a summary for Claude
    summary_cols = [
        'experiment_name', 'group', 'hidden_size', 'dropout', 'learning_rate',
        'best_val_loss', 'test_mse', 'sharpe_ratio', 'dir_acc', 'collapsed'
    ]
    # Only include columns that exist
    summary_cols = [c for c in summary_cols if c in df.columns]
    df_summary = df[summary_cols]
    
    summary_file = args.output.replace('.csv', '_for_claude.csv')
    df_summary.to_csv(summary_file, index=False)
    print(f" Saved Claude-friendly summary to {summary_file}")
    
    # Print quick stats
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total experiments: {len(df)}")
    print(f"\nBy group:")
    for group, count in df['group'].value_counts().items():
        print(f"  {group}: {count}")
    
    if 'evaluated' in df.columns:
        evaluated = df['evaluated'].sum()
        print(f"\nEvaluated: {evaluated}/{len(df)}")
    
    if 'collapsed' in df.columns and df['collapsed'].notna().any():
        collapsed = df[df['collapsed'] == True]
        print(f"Collapsed: {len(collapsed)}/{len(df)}")
    
    # Best models
    if 'sharpe_ratio' in df.columns:
        best_by_sharpe = df.nlargest(3, 'sharpe_ratio')[['experiment_name', 'sharpe_ratio', 'dir_acc']]
        print(f"\nTop 3 by Sharpe Ratio:")
        for _, row in best_by_sharpe.iterrows():
            print(f"  {row['experiment_name']}: Sharpe={row['sharpe_ratio']:.4f}, Acc={row['dir_acc']:.2%}")


if __name__ == '__main__':
    main()

