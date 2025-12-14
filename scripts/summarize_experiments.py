#!/usr/bin/env python3
"""
Summarize all experiments into CSV files for analysis.
Enhanced version incorporating collapse detection and ranking from compare_evaluations.py

Supports both old (single evaluation) and new (per-checkpoint) folder structures.

Outputs:
    - experiments_summary.csv: Full details for all experiments
    - experiments_summary_key_metrics.csv: Condensed version with essential metrics
    - working_models.csv: Only non-collapsed evaluated models (with --split-working)
    - collapsed_models.csv: Only collapsed models (with --split-working)

Usage:
    python scripts/summarize_experiments.py --all  # All phases
    python scripts/summarize_experiments.py --phase 00_baseline_exploration
    python scripts/summarize_experiments.py --phase 01_staleness_features
    python scripts/summarize_experiments.py --split-working  # Also output working/collapsed CSVs
    python scripts/summarize_experiments.py 00_baseline_exploration/exp001 01_staleness_features/exp002
    
    # Checkpoint selection for multi-checkpoint experiments:
    python scripts/summarize_experiments.py --all --best-by dir_acc --min-epoch 20
    python scripts/summarize_experiments.py --all --best-by sharpe --min-epoch 15
    python scripts/summarize_experiments.py --all --best-by healthy_pct --no-require-diverse
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime


def is_rolling_experiment(exp_path):
    """Check if this is a rolling evaluation experiment."""
    rolling_csv = exp_path / 'rolling_results_full.csv'
    has_folds = len(list(exp_path.glob('fold_*'))) > 0
    return rolling_csv.exists() or has_folds


def parse_json_column(val):
    """Parse a JSON string column, handling both string and dict inputs."""
    if pd.isna(val):
        return {}
    if isinstance(val, dict):
        return val
    try:
        # Handle single quotes (Python dict repr) vs double quotes (JSON)
        import ast
        return ast.literal_eval(val)
    except:
        try:
            return json.loads(val)
        except:
            return {}


def extract_rolling_experiment_data(exp_path, verbose=False):
    """
    Extract aggregated data from a rolling evaluation experiment.
    
    Parses rolling_results_full.csv and computes cross-fold statistics.
    """
    exp_name = exp_path.name
    parent_name = exp_path.parent.name
    phase = parent_name if parent_name != 'experiments' else None
    
    data = {
        'experiment_name': exp_name,
        'phase': phase,
        'evaluation_type': 'rolling',
    }
    
    # Read config.json for hyperparameters (from parent or first fold)
    config_path = exp_path / 'config.json'
    if not config_path.exists():
        # Try first fold
        fold_dirs = sorted(exp_path.glob('fold_*'))
        if fold_dirs:
            config_path = fold_dirs[0] / 'config.json'
    
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        data['hidden_size'] = config.get('architecture', {}).get('hidden_size')
        data['dropout'] = config.get('architecture', {}).get('dropout')
        data['learning_rate'] = config.get('training', {}).get('learning_rate')
        data['encoder_length'] = config.get('architecture', {}).get('max_encoder_length')
        data['attention_heads'] = config.get('architecture', {}).get('attention_head_size')
        data['batch_size'] = config.get('training', {}).get('batch_size')
    
    # Load rolling results
    rolling_csv = exp_path / 'rolling_results_full.csv'
    if not rolling_csv.exists():
        data['evaluated'] = False
        return data
    
    try:
        df = pd.read_csv(rolling_csv)
    except Exception as e:
        if verbose:
            print(f"  Failed to read {rolling_csv}: {e}")
        data['evaluated'] = False
        return data
    
    if len(df) == 0:
        data['evaluated'] = False
        return data
    
    data['evaluated'] = True
    data['n_folds'] = len(df)
    data['checkpoint_selected'] = 'varies_by_fold'
    
    # Parse JSON columns and extract metrics per fold
    fold_metrics = {
        'dir_acc': [],
        'sharpe': [],
        'total_return': [],
        'max_drawdown': [],
        'auc_roc': [],
        'alpha': [],
        'mse': [],
        'rmse': [],
        'mae': [],
        'r2': [],
        'healthy_pct': [],
        'degraded_pct': [],
        'unidirectional_pct': [],
        'weak_collapse_pct': [],
        'strong_collapse_pct': [],
        'pred_std': [],
        'pct_positive': [],
        'pct_negative': [],
        'num_negative': [],
    }
    
    for _, row in df.iterrows():
        # Parse financial metrics
        fin = parse_json_column(row.get('financial_metrics', {}))
        if fin:
            if 'directional_accuracy' in fin:
                fold_metrics['dir_acc'].append(fin['directional_accuracy'])
            if 'sharpe_ratio' in fin:
                fold_metrics['sharpe'].append(fin['sharpe_ratio'])
            if 'total_return' in fin:
                fold_metrics['total_return'].append(fin['total_return'])
            if 'max_drawdown' in fin:
                fold_metrics['max_drawdown'].append(fin['max_drawdown'])
            if 'auc_roc' in fin:
                fold_metrics['auc_roc'].append(fin['auc_roc'])
            if 'alpha' in fin:
                fold_metrics['alpha'].append(fin['alpha'])
        
        # Parse statistical metrics
        stat = parse_json_column(row.get('statistical_metrics', {}))
        if stat:
            if 'mse' in stat:
                fold_metrics['mse'].append(stat['mse'])
            if 'rmse' in stat:
                fold_metrics['rmse'].append(stat['rmse'])
            if 'mae' in stat:
                fold_metrics['mae'].append(stat['mae'])
            if 'r2' in stat:
                fold_metrics['r2'].append(stat['r2'])
        
        # Parse mode stats
        mode = parse_json_column(row.get('mode_stats', {}))
        if mode:
            if 'healthy_pct' in mode:
                fold_metrics['healthy_pct'].append(mode['healthy_pct'])
            if 'degraded_pct' in mode:
                fold_metrics['degraded_pct'].append(mode['degraded_pct'])
            if 'unidirectional_pct' in mode:
                fold_metrics['unidirectional_pct'].append(mode['unidirectional_pct'])
            if 'weak_collapse_pct' in mode:
                fold_metrics['weak_collapse_pct'].append(mode['weak_collapse_pct'])
            if 'strong_collapse_pct' in mode:
                fold_metrics['strong_collapse_pct'].append(mode['strong_collapse_pct'])
        
        # Parse prediction stats
        pred = parse_json_column(row.get('prediction_stats', {}))
        if pred:
            if 'std' in pred:
                fold_metrics['pred_std'].append(pred['std'])
            if 'pct_positive' in pred:
                fold_metrics['pct_positive'].append(pred['pct_positive'])
            if 'pct_negative' in pred:
                fold_metrics['pct_negative'].append(pred['pct_negative'])
                # Also compute num_negative from pct and num_predictions
                pct_neg = pred.get('pct_negative', 0)
                n_pred = pred.get('num_predictions', 0)
                fold_metrics['num_negative'].append(pct_neg * n_pred / 100 if n_pred > 0 else 0)
    
    # Compute aggregated statistics (mean and std across folds)
    # Use primary column names directly (mean value)
    # Only keep _std for key metrics where variability matters
    metric_to_primary = {
        'dir_acc': 'dir_acc',
        'sharpe': 'sharpe_ratio',
        'auc_roc': 'auc_roc',
        'alpha': 'alpha',
        'total_return': 'total_return',
        'max_drawdown': 'max_drawdown',
        'mse': 'test_mse',
        'rmse': 'test_rmse',
        'mae': 'test_mae',
        'r2': 'test_r2',
        'healthy_pct': 'healthy_pct',
        'degraded_pct': 'degraded_pct',
        'unidirectional_pct': 'unidirectional_pct',
        'weak_collapse_pct': 'weak_collapse_pct',
        'strong_collapse_pct': 'strong_collapse_pct',
        'pred_std': 'pred_std',
        'pct_positive': 'pct_positive',
        'pct_negative': 'pct_negative',
        'num_negative': 'num_negative',
    }
    
    # Only include _std for these key metrics (others are noise)
    key_std_metrics = ['dir_acc', 'sharpe', 'healthy_pct']
    
    for metric, values in fold_metrics.items():
        if values:
            arr = np.array(values)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            
            # Store as primary column name
            primary_col = metric_to_primary.get(metric)
            if primary_col:
                data[primary_col] = mean_val
            
            # Only store _std for key metrics
            if metric in key_std_metrics:
                data[f'{metric}_std'] = std_val
    
    # Compute derived metrics
    if 'healthy_pct' in data:
        data['problematic_pct'] = 100 - data['healthy_pct']
    
    # Collapse flags based on mean values
    data['has_any_collapse'] = (data.get('weak_collapse_pct', 0) + data.get('strong_collapse_pct', 0)) > 0
    data['has_strong_collapse'] = data.get('strong_collapse_pct', 0) > 0
    data['has_degradation'] = data.get('degraded_pct', 0) > 0
    data['has_unidirectional'] = data.get('unidirectional_pct', 0) > 0
    
    # Check if mostly collapsed (all folds have num_negative == 0)
    if fold_metrics['num_negative']:
        data['collapsed'] = all(n == 0 for n in fold_metrics['num_negative'])
    else:
        data['collapsed'] = None
    
    # Extract fold years for reference
    if 'test_year' in df.columns:
        years = sorted(df['test_year'].unique())
        data['fold_years'] = f"{min(years)}-{max(years)}"
    
    # Extract epoch range from checkpoint names
    epochs_used = []
    if 'checkpoint_used' in df.columns:
        for ckpt in df['checkpoint_used'].dropna():
            import re
            match = re.search(r'epoch[=_](\d+)', str(ckpt))
            if match:
                epochs_used.append(int(match.group(1)))
    
    if epochs_used:
        data['checkpoint_epoch_min'] = min(epochs_used)
        data['checkpoint_epoch_max'] = max(epochs_used)
        data['checkpoint_epoch'] = np.mean(epochs_used)  # For compatibility
    
    # Explicitly mark fields that don't apply to rolling experiments
    data['best_val_loss'] = 'N/A (rolling)'
    data['total_epochs'] = 'N/A (rolling)'
    data['early_stopped'] = 'N/A (rolling)'
    
    # Ensure all standard columns exist (with NaN if not populated)
    # This prevents blank cells from being ambiguous
    expected_numeric_cols = [
        'dir_acc', 'sharpe_ratio', 'auc_roc', 'alpha', 'total_return', 'max_drawdown',
        'test_mse', 'test_rmse', 'test_mae', 'test_r2',
        'healthy_pct', 'degraded_pct', 'unidirectional_pct', 'weak_collapse_pct', 'strong_collapse_pct',
        'pred_std', 'pct_positive', 'pct_negative', 'num_negative',
        'problematic_pct',
    ]
    for col in expected_numeric_cols:
        if col not in data or data[col] is None:
            data[col] = np.nan
    
    # Boolean columns - only set to NaN if completely missing
    expected_bool_cols = [
        'has_any_collapse', 'has_strong_collapse', 'has_degradation', 'has_unidirectional',
    ]
    for col in expected_bool_cols:
        if col not in data:
            data[col] = np.nan
    
    if verbose:
        print(f"  {exp_name}: rolling ({data['n_folds']} folds, years {data.get('fold_years', 'N/A')})")
    
    return data


def select_best_checkpoint(comparison_csv_path, best_by='dir_acc', min_epoch=None, 
                           require_diverse=True, verbose=False):
    """
    Select the best checkpoint from checkpoint_comparison.csv.
    
    Parameters
    ----------
    comparison_csv_path : Path
        Path to checkpoint_comparison.csv
    best_by : str
        Column to sort by for selection (dir_acc, sharpe, healthy_pct, num_negative, etc.)
    min_epoch : int or None
        Minimum epoch threshold (exclude early unconverged checkpoints)
    require_diverse : bool
        If True, require num_negative > 0 (exclude collapsed checkpoints)
    verbose : bool
        Print warnings about filtering
        
    Returns
    -------
    tuple: (best_row as dict, checkpoint_name, warning_message or None)
    """
    try:
        df = pd.read_csv(comparison_csv_path)
    except Exception as e:
        return None, None, f"Failed to read {comparison_csv_path}: {e}"
    
    if len(df) == 0:
        return None, None, "Empty checkpoint_comparison.csv"
    
    original_count = len(df)
    warnings = []
    
    # Epoch filter
    if min_epoch is not None and 'epoch' in df.columns:
        df_filtered = df[df['epoch'] >= min_epoch]
        if len(df_filtered) == 0:
            warnings.append(f"No checkpoints >= epoch {min_epoch}, using all {original_count}")
            df_filtered = df
        elif verbose and len(df_filtered) < original_count:
            print(f"  Epoch filter: {original_count} -> {len(df_filtered)}")
        df = df_filtered
    
    # Diversity filter (require some negative predictions)
    if require_diverse and 'num_negative' in df.columns:
        df_diverse = df[df['num_negative'] > 0]
        if len(df_diverse) == 0:
            warnings.append(f"No diverse checkpoints (num_negative > 0), using {len(df)} filtered")
        else:
            if verbose and len(df_diverse) < len(df):
                print(f"  Diversity filter: {len(df)} -> {len(df_diverse)}")
            df = df_diverse
    
    # Check if best_by column exists
    if best_by not in df.columns:
        # Try common alternatives
        alternatives = {
            'dir_acc': ['directional_accuracy', 'test_dir_acc'],
            'sharpe': ['sharpe_ratio', 'test_sharpe'],
        }
        found = False
        for alt in alternatives.get(best_by, []):
            if alt in df.columns:
                best_by = alt
                found = True
                break
        if not found:
            return None, None, f"Column '{best_by}' not found. Available: {df.columns.tolist()}"
    
    # Determine sort order (lower is better for some metrics)
    ascending = best_by in ['mse', 'mae', 'max_drawdown', 'val_loss', 
                            'weak_collapse_pct', 'strong_collapse_pct', 
                            'degraded_pct', 'unidirectional_pct']
    
    # Sort and select best
    df_sorted = df.sort_values(best_by, ascending=ascending, na_position='last')
    best_row = df_sorted.iloc[0]
    
    # Extract checkpoint name
    # Try multiple possible column names
    checkpoint_name = None
    for col_name in ['checkpoint', 'ckpt_name', 'checkpoint_name', 'ckpt']:
        if col_name in best_row.index:
            checkpoint_name = best_row[col_name]
            break
    
    if verbose:
        print(f"    CSV columns: {df.columns.tolist()}")
        print(f"    checkpoint_name extracted: {checkpoint_name}")
    
    warning_msg = "; ".join(warnings) if warnings else None
    
    return best_row.to_dict(), checkpoint_name, warning_msg


def load_evaluation_metrics(eval_json_path):
    """Load evaluation metrics from JSON file."""
    try:
        with open(eval_json_path) as f:
            return json.load(f)
    except Exception as e:
        return None


def parse_evaluation_log(log_path):
    """
    Parse evaluation log file to extract metrics not in JSON.
    
    Returns dict with extracted values, or empty dict if parsing fails.
    """
    metrics = {}
    if not log_path.exists():
        return metrics
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        import re
        
        # Financial/classification metrics patterns
        financial_patterns = {
            'precision': r'Precision:\s+([0-9.]+)',
            'recall': r'Recall:\s+([0-9.]+)',
            'f1_score': r'F1 Score:\s+([0-9.]+)',
            'hit_rate': r'Hit Rate:\s+([0-9.]+)',
            'num_trades': r'Number of Trades:\s+([0-9]+)',
            'auc_roc': r'AUC-ROC:\s+([0-9.]+)',
            'alpha': r'Excess Return:\s+([0-9.-]+)',
        }
        
        for key, pattern in financial_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                val = match.group(1)
                if key == 'num_trades':
                    metrics[key] = int(val)
                else:
                    metrics[key] = float(val)
        
        # Handle percentage values (divide by 100 for consistency)
        if 'hit_rate' in metrics and metrics['hit_rate'] > 1:
            metrics['hit_rate'] = metrics['hit_rate'] / 100
        
        # Prediction statistics patterns
        pred_patterns = {
            'pred_min': r'Prediction statistics:.*?Min:\s+([0-9.-]+)',
            'pred_max': r'Prediction statistics:.*?Max:\s+([0-9.-]+)',
            'pred_mean': r'Prediction statistics:.*?Mean:\s+([0-9.-]+)',
            'pred_std': r'Prediction statistics:.*?Std:\s+([0-9.-]+)',
            'num_unique': r'Unique values:\s+([0-9]+)',
        }
        
        for key, pattern in pred_patterns.items():
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                val = match.group(1)
                if key == 'num_unique':
                    metrics[key] = int(val)
                else:
                    metrics[key] = float(val)
        
        # Negative predictions: "Negative predictions: 0/1282"
        neg_match = re.search(r'Negative predictions:\s+(\d+)/(\d+)', content)
        if neg_match:
            num_negative = int(neg_match.group(1))
            total = int(neg_match.group(2))
            metrics['num_negative'] = num_negative
            metrics['num_positive'] = total - num_negative
            metrics['pct_negative'] = (num_negative / total * 100) if total > 0 else 0
            metrics['pct_positive'] = 100 - metrics['pct_negative']
        
        # Mode distribution patterns: "HEALTHY:            29 days (  2.3%)" or "HEALTHY:             5 weeks (  1.9%)"
        mode_patterns = {
            'healthy_days': r'HEALTHY:\s+(\d+)\s+(?:days|weeks)',
            'degraded_days': r'DEGRADED:\s+(\d+)\s+(?:days|weeks)',
            'unidirectional_days': r'UNIDIRECTIONAL:\s+(\d+)\s+(?:days|weeks)',
            'weak_collapse_days': r'WEAK_COLLAPSE:\s+(\d+)\s+(?:days|weeks)',
            'strong_collapse_days': r'STRONG_COLLAPSE:\s+(\d+)\s+(?:days|weeks)',
        }
        
        for key, pattern in mode_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metrics[key] = int(match.group(1))
        
        return metrics
    except Exception as e:
        return {}


def extract_epoch_from_checkpoint(checkpoint_path_or_name):
    """
    Extract epoch number from checkpoint filename.
    
    Handles formats like:
    - tft-epoch=epoch=05-sharpe=val_sharpe=557.2863.ckpt
    - tft-epoch_epoch_05-sharpe_val_sharpe_557_2863 (folder name variant)
    - tft-epoch=42-val_loss=0.1234.ckpt
    - epoch_42.ckpt
    """
    import re
    
    if checkpoint_path_or_name is None:
        return None
    
    name = str(checkpoint_path_or_name)
    
    # Try various patterns (order matters - more specific first)
    patterns = [
        r'epoch[=_]epoch[=_](\d+)',  # epoch=epoch=05 or epoch_epoch_05
        r'epoch[=_](\d+)',           # epoch=42 or epoch_42
        r'epoch(\d+)',               # epoch42
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return None


def find_checkpoint_epoch_from_log(eval_dir):
    """
    Find the checkpoint epoch from evaluation log by parsing checkpoint path.
    """
    log_files = list(eval_dir.glob('evaluation_*.log'))
    if not log_files:
        return None
    
    try:
        with open(log_files[0], 'r') as f:
            content = f.read()
        
        import re
        # Look for checkpoint loading line
        # Handles both: tft-epoch=epoch=05-... and tft-epoch_epoch_05-...
        match = re.search(r'(tft-epoch[^\s/\\]+\.ckpt)', content)
        if match:
            return extract_epoch_from_checkpoint(match.group(1))
        
        # Also check for folder name pattern (used in evaluation path)
        # e.g., evaluation/tft-epoch_epoch_05-sharpe_val_sharpe_557_2863/
        match = re.search(r'tft-epoch[_=]epoch[_=](\d+)', content)
        if match:
            return int(match.group(1))
        
        # Fallback: look for any epoch pattern
        match = re.search(r'epoch[=_](\d+)', content)
        if match:
            return int(match.group(1))
            
    except Exception:
        pass
    
    return None


def extract_experiment_data(exp_path, best_by='dir_acc', min_epoch=None, 
                           require_diverse=True, verbose=False):
    """Extract data from one experiment directory."""
    exp_name = exp_path.name
    
    # Determine phase from parent directory
    parent_name = exp_path.parent.name
    phase = parent_name if parent_name != 'experiments' else None
    
    data = {
        'experiment_name': exp_name,
        'phase': phase,
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
    
    # =========================================================================
    # EVALUATION METRICS - Handle both old and new folder structures
    # =========================================================================
    eval_dir = exp_path / 'evaluation'
    eval_data = None
    checkpoint_selected = None
    checkpoint_epoch = None
    selection_warning = None
    
    if eval_dir.exists():
        # Check for new structure: checkpoint_comparison.csv
        comparison_csv = eval_dir / 'checkpoint_comparison.csv'
        direct_eval = eval_dir / 'evaluation_metrics.json'
        
        if comparison_csv.exists():
            # NEW STRUCTURE: Multiple checkpoints evaluated
            best_row, checkpoint_selected, selection_warning = select_best_checkpoint(
                comparison_csv,
                best_by=best_by,
                min_epoch=min_epoch,
                require_diverse=require_diverse,
                verbose=verbose
            )
            
            if verbose:
                print(f"    checkpoint_selected: {checkpoint_selected}")
            
            if best_row is not None:
                checkpoint_epoch = best_row.get('epoch')
                
                # Try to load full metrics from checkpoint subfolder
                if checkpoint_selected:
                    # Folder names use _ instead of = and . (filesystem restriction)
                    # Also remove .ckpt extension if present
                    folder_name = checkpoint_selected.replace('.ckpt', '').replace('=', '_').replace('.', '_')
                    ckpt_eval_dir = eval_dir / folder_name
                    
                    # If exact folder not found, try to find by epoch prefix
                    if not ckpt_eval_dir.exists() and checkpoint_epoch is not None:
                        # Try different epoch formats (padded and unpadded)
                        for epoch_str in [f"{checkpoint_epoch:02d}", f"{checkpoint_epoch}"]:
                            epoch_prefix = f"tft-epoch_epoch_{epoch_str}"
                            matching_dirs = [d for d in eval_dir.iterdir() 
                                            if d.is_dir() and d.name.startswith(epoch_prefix)]
                            if matching_dirs:
                                ckpt_eval_dir = matching_dirs[0]
                                if verbose:
                                    print(f"    Found by prefix: {ckpt_eval_dir.name}")
                                break
                    
                    ckpt_eval_path = ckpt_eval_dir / 'evaluation_metrics.json'
                    
                    if verbose:
                        print(f"    Looking for: {ckpt_eval_path}")
                        print(f"    Exists: {ckpt_eval_path.exists()}")
                    
                    if ckpt_eval_path.exists():
                        eval_data = load_evaluation_metrics(ckpt_eval_path)
                    
                    # Also try to parse evaluation log for additional metrics (precision, recall, etc.)
                    log_files = list(ckpt_eval_dir.glob('evaluation_*.log')) if ckpt_eval_dir.exists() else []
                    if verbose:
                        print(f"    Log files found: {len(log_files)}")
                    
                    if log_files:
                        log_metrics = parse_evaluation_log(log_files[0])
                        if verbose:
                            print(f"    Log metrics parsed: {list(log_metrics.keys()) if log_metrics else 'None'}")
                        if log_metrics:
                            # Merge into eval_data financial_metrics
                            if eval_data is None:
                                eval_data = {'financial_metrics': {}, 'statistical_metrics': {}, 'mode_stats': {}}
                            if 'financial_metrics' not in eval_data:
                                eval_data['financial_metrics'] = {}
                            if 'mode_stats' not in eval_data:
                                eval_data['mode_stats'] = {}
                            
                            # Separate metrics by type
                            financial_keys = ['precision', 'recall', 'f1_score', 'hit_rate', 'num_trades', 'auc_roc', 'alpha']
                            mode_keys = ['healthy_days', 'degraded_days', 'unidirectional_days', 'weak_collapse_days', 'strong_collapse_days']
                            pred_keys = ['pred_min', 'pred_max', 'pred_mean', 'pred_std', 'num_unique', 'num_negative', 'num_positive', 'pct_negative', 'pct_positive']
                            
                            for key, val in log_metrics.items():
                                if key in financial_keys:
                                    existing = eval_data['financial_metrics'].get(key)
                                    if existing is None:
                                        eval_data['financial_metrics'][key] = val
                                        if verbose:
                                            print(f"      Added financial {key}={val} from log")
                                elif key in mode_keys:
                                    existing = eval_data['mode_stats'].get(key)
                                    if existing is None:
                                        eval_data['mode_stats'][key] = val
                                        if verbose:
                                            print(f"      Added mode_stat {key}={val} from log")
                                elif key in pred_keys:
                                    # Store prediction stats directly in data dict
                                    if key not in data or data[key] is None:
                                        data[key] = val
                                        if verbose:
                                            print(f"      Added pred_stat {key}={val} from log")
                
                # If subfolder doesn't exist or load failed, use CSV row data directly
                if eval_data is None:
                    # Map CSV columns to expected structure
                    eval_data = {
                        'statistical_metrics': {
                            'mse': best_row.get('mse'),
                            'mae': best_row.get('mae'),
                            'rmse': np.sqrt(best_row.get('mse', 0)) if best_row.get('mse') else None,
                            'r2': best_row.get('r2'),
                        },
                        'financial_metrics': {
                            'directional_accuracy': best_row.get('dir_acc'),
                            'sharpe_ratio': best_row.get('sharpe'),
                            'max_drawdown': best_row.get('max_drawdown'),
                            'total_return': best_row.get('total_return'),
                        },
                        'mode_stats': {
                            'healthy_pct': best_row.get('healthy_pct'),
                            'degraded_pct': best_row.get('degraded_pct'),
                            'unidirectional_pct': best_row.get('unidirectional_pct'),
                            'weak_collapse_pct': best_row.get('weak_collapse_pct'),
                            'strong_collapse_pct': best_row.get('strong_collapse_pct'),
                        },
                        '_from_csv': True,  # Flag that this came from CSV, not full JSON
                    }
                    # Also store prediction stats from CSV
                    data['pred_std'] = best_row.get('pred_std')
                    data['pred_mean'] = best_row.get('pred_mean')
                    data['pct_positive'] = best_row.get('pct_positive')
                    data['num_negative'] = best_row.get('num_negative')
        
        elif direct_eval.exists():
            # OLD STRUCTURE: Single evaluation in evaluation/evaluation_metrics.json
            eval_data = load_evaluation_metrics(direct_eval)
            checkpoint_selected = 'default'
            
            # Try to extract checkpoint epoch from evaluation log
            checkpoint_epoch = find_checkpoint_epoch_from_log(eval_dir)
            
            # Fallback: try to find epoch from checkpoints folder (best checkpoint)
            if checkpoint_epoch is None:
                checkpoints_dir = exp_path / 'checkpoints'
                if checkpoints_dir.exists():
                    # Look for checkpoint files and extract epoch from best one
                    ckpt_files = list(checkpoints_dir.glob('*.ckpt'))
                    if ckpt_files:
                        # Try to find "best" or "valloss" checkpoint first
                        for ckpt in ckpt_files:
                            if 'valloss' in ckpt.name.lower() or 'val_loss' in ckpt.name.lower():
                                checkpoint_epoch = extract_epoch_from_checkpoint(ckpt.name)
                                if checkpoint_epoch is not None:
                                    break
                        # If not found, just use the first checkpoint with an epoch
                        if checkpoint_epoch is None:
                            for ckpt in ckpt_files:
                                checkpoint_epoch = extract_epoch_from_checkpoint(ckpt.name)
                                if checkpoint_epoch is not None:
                                    break
            
            if verbose:
                print(f"    OLD structure, checkpoint_epoch: {checkpoint_epoch}")
            
            # Also try to parse evaluation log for additional metrics
            log_files = list(eval_dir.glob('evaluation_*.log'))
            if verbose:
                print(f"    Log files found: {len(log_files)}")
            
            if log_files:
                log_metrics = parse_evaluation_log(log_files[0])
                if verbose:
                    print(f"    Log metrics parsed: {list(log_metrics.keys()) if log_metrics else 'None'}")
                if log_metrics and eval_data:
                    if 'financial_metrics' not in eval_data:
                        eval_data['financial_metrics'] = {}
                    if 'mode_stats' not in eval_data:
                        eval_data['mode_stats'] = {}
                    
                    # Separate metrics by type
                    financial_keys = ['precision', 'recall', 'f1_score', 'hit_rate', 'num_trades', 'auc_roc', 'alpha']
                    mode_keys = ['healthy_days', 'degraded_days', 'unidirectional_days', 'weak_collapse_days', 'strong_collapse_days']
                    pred_keys = ['pred_min', 'pred_max', 'pred_mean', 'pred_std', 'num_unique', 'num_negative', 'num_positive', 'pct_negative', 'pct_positive']
                    
                    for key, val in log_metrics.items():
                        if key in financial_keys:
                            existing = eval_data['financial_metrics'].get(key)
                            if existing is None:
                                eval_data['financial_metrics'][key] = val
                                if verbose:
                                    print(f"      Added financial {key}={val} from log")
                        elif key in mode_keys:
                            existing = eval_data['mode_stats'].get(key)
                            if existing is None:
                                eval_data['mode_stats'][key] = val
                                if verbose:
                                    print(f"      Added mode_stat {key}={val} from log")
                        elif key in pred_keys:
                            # Store prediction stats directly in data dict
                            if key not in data or data[key] is None:
                                data[key] = val
                                if verbose:
                                    print(f"      Added pred_stat {key}={val} from log")
    
    # Store checkpoint selection info
    data['checkpoint_selected'] = checkpoint_selected
    data['checkpoint_epoch'] = checkpoint_epoch
    if selection_warning:
        data['selection_warning'] = selection_warning
    
    # Process evaluation data if we have it
    if eval_data:
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
        data['hit_rate'] = fin.get('hit_rate')
        
        # Mode statistics (from 5-mode quality detection system)
        if 'mode_stats' in eval_data:
            mode_stats = eval_data['mode_stats']
            data['healthy_pct'] = mode_stats.get('healthy_pct')
            data['degraded_pct'] = mode_stats.get('degraded_pct')
            data['unidirectional_pct'] = mode_stats.get('unidirectional_pct')
            data['weak_collapse_pct'] = mode_stats.get('weak_collapse_pct')
            data['strong_collapse_pct'] = mode_stats.get('strong_collapse_pct')
            data['healthy_days'] = mode_stats.get('healthy_days')
            data['degraded_days'] = mode_stats.get('degraded_days')
            data['unidirectional_days'] = mode_stats.get('unidirectional_days')
            data['weak_collapse_days'] = mode_stats.get('weak_collapse_days')
            data['strong_collapse_days'] = mode_stats.get('strong_collapse_days')
            
            # Compute boolean quality flags from mode percentages
            data['has_any_collapse'] = (mode_stats.get('weak_collapse_pct', 0) + mode_stats.get('strong_collapse_pct', 0)) > 0
            data['has_strong_collapse'] = mode_stats.get('strong_collapse_pct', 0) > 0
            data['has_degradation'] = mode_stats.get('degraded_pct', 0) > 0
            data['has_unidirectional'] = mode_stats.get('unidirectional_pct', 0) > 0
            data['problematic_pct'] = (mode_stats.get('degraded_pct', 0) + 
                                       mode_stats.get('unidirectional_pct', 0) +
                                       mode_stats.get('weak_collapse_pct', 0) + 
                                       mode_stats.get('strong_collapse_pct', 0))
        
        # Enhanced collapse detection from confusion matrix (if available in full JSON)
        if 'confusion_matrix' in fin:
            conf = fin['confusion_matrix']
            # conf_matrix = [[TN, FP], [FN, TP]]
            predicted_down = conf[0][0] + conf[1][0]  # TN + FN
            predicted_up = conf[0][1] + conf[1][1]    # FP + TP
            total_predictions = predicted_down + predicted_up
            
            data['predicted_down'] = predicted_down
            data['predicted_up'] = predicted_up
            data['pct_down_predictions'] = predicted_down / total_predictions if total_predictions > 0 else 0
            data['pct_up_predictions'] = predicted_up / total_predictions if total_predictions > 0 else 0
        
        data['evaluated'] = True
    else:
        data['evaluated'] = False
    
    # Read collapse diagnosis if exists (from diagnose_existing_models.py)
    diagnosis_path = exp_path / 'collapse_diagnosis.json'
    pred_stats_loaded = 'pred_std' in data and data['pred_std'] is not None
    diag_collapsed = None
    
    if diagnosis_path.exists():
        with open(diagnosis_path) as f:
            diag = json.load(f)
        
        # Use diagnosis collapse flag if available
        diag_collapsed = diag.get('collapsed')
        
        # Extract prediction stats from diagnosis (if not already loaded from CSV)
        if not pred_stats_loaded and 'predictions' in diag and diag['predictions']:
            pred = diag['predictions']
            data['pred_std'] = pred.get('std')
            data['pred_mean'] = pred.get('mean')
            data['pred_min'] = pred.get('min')
            data['pred_max'] = pred.get('max')
            data['num_unique'] = pred.get('num_unique')
            data['num_positive'] = pred.get('num_positive')
            data['num_negative'] = pred.get('num_negative')
            data['num_zero'] = pred.get('num_zero')
            data['pct_positive'] = pred.get('pct_positive')
            data['pct_negative'] = pred.get('pct_negative')
            pred_stats_loaded = True
    
    # Fallback: read from predictions.csv if we still don't have pred stats
    if not pred_stats_loaded:
        # Try selected checkpoint's predictions.csv first
        if checkpoint_selected and checkpoint_selected != 'default':
            pred_csv = exp_path / 'evaluation' / checkpoint_selected / 'predictions.csv'
        else:
            pred_csv = exp_path / 'evaluation' / 'predictions.csv'
        
        if pred_csv.exists():
            pred_df = pd.read_csv(pred_csv)
            if 'Predicted' in pred_df.columns:
                preds = pred_df['Predicted'].values
                
                data['pred_std'] = float(preds.std())
                data['pred_mean'] = float(preds.mean())
                data['pred_min'] = float(preds.min())
                data['pred_max'] = float(preds.max())
                data['num_unique'] = int(len(np.unique(preds.round(6))))
                data['num_positive'] = int((preds > 0).sum())
                data['num_negative'] = int((preds < 0).sum())
                data['num_zero'] = int((preds == 0).sum())
                data['pct_positive'] = float((preds > 0).mean() * 100)
                data['pct_negative'] = float((preds < 0).mean() * 100)
    
    # Determine final collapse status using multiple indicators
    # Priority: confusion matrix > diagnosis > pred_std
    if data.get('evaluated') and 'pct_down_predictions' in data:
        # Use confusion matrix if available (most reliable)
        data['collapsed'] = (
            data['pct_down_predictions'] == 0 or 
            data['pct_up_predictions'] == 0 or
            data.get('pred_std', 1.0) < 0.05
        )
    elif diag_collapsed is not None:
        # Use diagnosis if available
        data['collapsed'] = diag_collapsed
    elif 'pred_std' in data and data['pred_std'] is not None:
        # Fallback to pred_std
        data['collapsed'] = data['pred_std'] < 0.05
    else:
        # Unknown
        data['collapsed'] = None
    
    # Ensure all expected columns exist (with NaN if not populated)
    # This prevents blank cells from being ambiguous
    # Note: Be careful not to overwrite False with NaN for boolean columns
    expected_numeric_cols = [
        'dir_acc', 'sharpe_ratio', 'auc_roc', 'alpha', 'total_return', 'max_drawdown',
        'test_mse', 'test_rmse', 'test_mae', 'test_r2',
        'healthy_pct', 'degraded_pct', 'unidirectional_pct', 'weak_collapse_pct', 'strong_collapse_pct',
        'pred_std', 'pct_positive', 'pct_negative', 'num_negative',
        'problematic_pct', 'checkpoint_epoch',
        'precision', 'recall', 'f1_score', 'hit_rate', 'num_trades',
    ]
    for col in expected_numeric_cols:
        if col not in data or data[col] is None:
            data[col] = np.nan
    
    # Boolean columns - only set to NaN if completely missing
    expected_bool_cols = [
        'has_any_collapse', 'has_strong_collapse', 'has_degradation', 'has_unidirectional',
    ]
    for col in expected_bool_cols:
        if col not in data:
            data[col] = np.nan
    
    # String columns
    if 'checkpoint_selected' not in data or data['checkpoint_selected'] is None:
        data['checkpoint_selected'] = np.nan
        
    return data


def compute_composite_score(df):
    """Compute composite score for ranking models."""
    if 'auc_roc' in df.columns and 'dir_acc' in df.columns:
        # Balance discriminative power (AUC) with accuracy
        # AUC already [0,1], dir_acc is [0,1], center at 0.5
        df['composite_score'] = df['auc_roc'] + (df['dir_acc'] - 0.5)
    return df


def print_working_vs_collapsed_summary(df):
    """Print analysis of working vs collapsed models."""
    if 'collapsed' not in df.columns or not df['collapsed'].notna().any():
        return
    
    evaluated = df[df['evaluated'] == True].copy()
    if len(evaluated) == 0:
        return
    
    working = evaluated[evaluated['collapsed'] == False].copy()
    collapsed = evaluated[evaluated['collapsed'] == True].copy()
    
    print(f"\n{'='*80}")
    print("WORKING vs COLLAPSED ANALYSIS")
    print(f"{'='*80}")
    print(f"Total evaluated: {len(evaluated)}")
    print(f"Working models (varied predictions): {len(working)}")
    print(f"Collapsed models (constant predictions): {len(collapsed)}")
    
    if len(working) > 0:
        print(f"\n{'='*80}")
        print("TOP WORKING MODELS")
        print(f"{'='*80}")
        
        # Add composite score
        working = compute_composite_score(working)
        
        display_cols = ['experiment_name', 'auc_roc', 'dir_acc', 'sharpe_ratio', 
                        'pct_down_predictions', 'num_trades']
        display_cols = [c for c in display_cols if c in working.columns]
        
        if 'composite_score' in working.columns:
            print("\nTop 5 by Composite Score (AUC + (dir_acc - 0.5)):")
            top = working.nlargest(5, 'composite_score')
            for col in ['auc_roc', 'dir_acc', 'sharpe_ratio', 'composite_score']:
                if col in top.columns:
                    print(f"\n{col}:")
                    for _, row in top.iterrows():
                        print(f"  {row['experiment_name']}: {row[col]:.4f}")
        
        if 'auc_roc' in working.columns:
            print("\nTop 5 by AUC-ROC (Discriminative Power):")
            top = working.nlargest(5, 'auc_roc')[display_cols]
            print(top.to_string(index=False))
        
        if 'dir_acc' in working.columns:
            print("\nTop 5 by Directional Accuracy:")
            top = working.nlargest(5, 'dir_acc')[display_cols]
            print(top.to_string(index=False))
        
        if 'sharpe_ratio' in working.columns:
            print("\nTop 5 by Sharpe Ratio:")
            top = working.nlargest(5, 'sharpe_ratio')[display_cols]
            print(top.to_string(index=False))
    
    if len(collapsed) > 0:
        print(f"\n{'='*80}")
        print("COLLAPSED MODELS (Reference)")
        print(f"{'='*80}")
        print(f"\nShowing first 5 collapsed models:")
        display_cols = ['experiment_name', 'dir_acc', 'sharpe_ratio', 'pct_down_predictions']
        display_cols = [c for c in display_cols if c in collapsed.columns]
        print(collapsed[display_cols].head().to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description='Summarize experiment results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('experiments', nargs='*', help='Specific experiment names')
    parser.add_argument('--all', action='store_true', help='Process all experiments across all phases')
    parser.add_argument('--phase', type=str, 
                       help='Process specific phase directory (e.g., 00_baseline_exploration)')
    parser.add_argument('--output', type=str, default='experiments_summary.csv', 
                       help='Output CSV filename')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save output files (default: current directory)')
    parser.add_argument('--evaluated-only', action='store_true',
                       help='Only include experiments with evaluation results')
    parser.add_argument('--split-working', action='store_true',
                       help='Also output working_models.csv and collapsed_models.csv')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed processing info')
    
    # Checkpoint selection arguments
    parser.add_argument('--best-by', type=str, default='dir_acc',
                       help='Metric to use for selecting best checkpoint '
                            '(dir_acc, sharpe, healthy_pct, num_negative, pred_std, etc.)')
    parser.add_argument('--min-epoch', type=int, default=None,
                       help='Minimum epoch for checkpoint selection (exclude early unconverged)')
    parser.add_argument('--no-require-diverse', action='store_true',
                       help='Disable diversity filter (allow collapsed checkpoints with num_negative=0)')
    parser.add_argument('--skip-rolling', action='store_true',
                       help='Skip rolling evaluation experiments (only include standard train/test)')
    parser.add_argument('--skip-phases', nargs='*', default=['03_distribution_loss', '05b_regime_with_penalties', '07c_regime_attention_ablations'],
                       help='List of phases to skip (default: 03_distribution_loss, 05b_regime_with_penalties, 07c_regime_attention_ablations)')
    parser.add_argument('--include-all-phases', action='store_true',
                       help='Include all phases (override --skip-phases)')
    
    args = parser.parse_args()
    
    exp_dir = Path('experiments')
    if not exp_dir.exists():
        print(f"Error: {exp_dir} directory not found")
        return
    
    # Determine which experiments to process
    if args.all:
        # Search all phase directories (00_, 01_, 02_, etc.)
        exp_paths = []
        for phase_dir in sorted(exp_dir.iterdir()):
            if phase_dir.is_dir() and phase_dir.name[0].isdigit():  # Phase directories start with digit
                exp_paths.extend([p for p in phase_dir.iterdir() if p.is_dir()])
    elif args.phase:
        # Process specific phase directory
        phase_dir = exp_dir / args.phase
        if not phase_dir.exists():
            print(f"Error: Phase directory {phase_dir} not found")
            return
        exp_paths = [p for p in phase_dir.iterdir() if p.is_dir()]
    elif args.experiments:
        exp_paths = [exp_dir / name for name in args.experiments]
    else:
        print("Error: Must specify --all, --phase, or experiment names")
        return
    
    # Create output directory if specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path('.')
    
    # Print checkpoint selection settings
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARIZATION")
    print(f"{'='*80}")
    print(f"\nCheckpoint selection settings:")
    print(f"  Best by: {args.best_by}")
    print(f"  Min epoch: {args.min_epoch if args.min_epoch else 'None (all epochs)'}")
    print(f"  Require diverse: {not args.no_require_diverse}")
    print(f"  Skip rolling: {args.skip_rolling}")
    
    # Handle phase skipping
    skip_phases = [] if args.include_all_phases else (args.skip_phases or [])
    if skip_phases:
        print(f"  Skip phases: {skip_phases}")
        # Filter out experiments from skipped phases
        exp_paths = [p for p in exp_paths if p.parent.name not in skip_phases]
    
    # Extract data from each experiment
    print(f"\nScanning {len(exp_paths)} experiments...")
    results = []
    rolling_count = 0
    standard_count = 0
    skipped_rolling = 0
    
    for exp_path in sorted(exp_paths):
        if not exp_path.exists():
            print(f"Warning: {exp_path.name} not found, skipping")
            continue
        
        # Skip archive folders
        if exp_path.name == 'archive' or 'archive' in exp_path.parts:
            if args.verbose:
                print(f"  {exp_path.name}: skipped (archive)")
            continue
        
        try:
            # Check if this is a rolling experiment
            if is_rolling_experiment(exp_path):
                if args.skip_rolling:
                    skipped_rolling += 1
                    if args.verbose:
                        print(f"  {exp_path.name}: skipped (rolling)")
                    continue
                data = extract_rolling_experiment_data(exp_path, verbose=args.verbose)
                rolling_count += 1
            else:
                data = extract_experiment_data(
                    exp_path,
                    best_by=args.best_by,
                    min_epoch=args.min_epoch,
                    require_diverse=not args.no_require_diverse,
                    verbose=args.verbose
                )
                if data:
                    data['evaluation_type'] = 'standard'
                standard_count += 1
            
            if data is None:
                continue
            
            # Skip if --evaluated-only and no evaluation
            if args.evaluated_only and not data.get('evaluated'):
                continue
            
            results.append(data)
            
            if args.verbose:
                eval_type = data.get('evaluation_type', 'standard')
                status = "✓ evaluated" if data.get('evaluated') else "  not evaluated"
                collapsed = " (COLLAPSED)" if data.get('collapsed') else ""
                if eval_type == 'rolling':
                    n_folds = data.get('n_folds', '?')
                    print(f"  {exp_path.name}: {status} [rolling, {n_folds} folds]{collapsed}")
                else:
                    ckpt_info = f" [ckpt: {data.get('checkpoint_selected', 'N/A')}]" if data.get('checkpoint_selected') else ""
                    print(f"  {exp_path.name}: {status}{collapsed}{ckpt_info}")
                    
        except Exception as e:
            print(f"  {exp_path.name}: ERROR - {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    if not results:
        print("\nNo experiments processed")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Add composite score for all evaluated models
    df = compute_composite_score(df)
    
    # Sort by phase, then experiment name
    df = df.sort_values(['phase', 'experiment_name'])
    
    # Prepare output paths
    output_path = output_dir / args.output
    key_metrics_path = output_dir / args.output.replace('.csv', '_key_metrics.csv')
    
    # Save full results
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(df)} experiments to {output_path}")
    
    # Create a condensed summary with key metrics
    key_metrics_cols = [
        'experiment_name', 'phase', 'evaluation_type', 'hidden_size', 'dropout', 'learning_rate',
        'best_val_loss', 'total_epochs', 'early_stopped',
        'checkpoint_selected', 'checkpoint_epoch', 'n_folds', 'fold_years',
        'test_mse', 'test_rmse', 'test_mae', 'test_r2',
        'dir_acc', 'sharpe_ratio', 'auc_roc', 'alpha',
        'precision', 'recall', 'f1_score', 'hit_rate', 'num_trades',
        # Rolling: std across folds for key metrics only
        'dir_acc_std', 'sharpe_std', 'healthy_pct_std',
        # 5-mode quality metrics (PRIMARY)
        'healthy_pct', 'degraded_pct', 'unidirectional_pct', 'weak_collapse_pct', 'strong_collapse_pct', 'problematic_pct',
        'has_any_collapse', 'has_strong_collapse', 'has_degradation', 'has_unidirectional',
        # Prediction characteristics
        'pred_std', 'num_unique', 'pct_positive', 'pct_negative', 'num_negative',
        'composite_score', 'evaluated', 'collapsed'
    ]
    # Only include columns that exist
    key_metrics_cols = [c for c in key_metrics_cols if c in df.columns]
    df_key_metrics = df[key_metrics_cols]
    
    df_key_metrics.to_csv(key_metrics_path, index=False)
    print(f"✓ Saved key metrics summary to {key_metrics_path}")
    
    # Split working vs collapsed if requested
    if args.split_working and 'collapsed' in df.columns:
        evaluated = df[df['evaluated'] == True].copy()
        working = evaluated[evaluated['collapsed'] == False].copy()
        collapsed = evaluated[evaluated['collapsed'] == True].copy()
        
        if len(working) > 0:
            working_path = output_dir / 'working_models.csv'
            working.to_csv(working_path, index=False)
            print(f"✓ Saved working_models.csv ({len(working)} models) to {working_path}")
        
        if len(collapsed) > 0:
            collapsed_path = output_dir / 'collapsed_models.csv'
            collapsed.to_csv(collapsed_path, index=False)
            print(f"✓ Saved collapsed_models.csv ({len(collapsed)} models) to {collapsed_path}")
    
    # Print quick stats
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total experiments: {len(df)}")
    if skipped_rolling > 0:
        print(f"  (Skipped {skipped_rolling} rolling experiments)")
    
    # Evaluation type breakdown
    if 'evaluation_type' in df.columns:
        print(f"\nBy evaluation type:")
        for eval_type, count in df['evaluation_type'].value_counts().items():
            print(f"  {eval_type}: {count}")
    
    if 'phase' in df.columns and df['phase'].notna().any():
        print(f"\nBy phase:")
        for phase, count in df['phase'].value_counts().sort_index().items():
            print(f"  {phase}: {count}")
    
    if 'evaluated' in df.columns:
        evaluated = df['evaluated'].sum()
        print(f"\nEvaluated: {evaluated}/{len(df)}")
    
    if 'collapsed' in df.columns and df['collapsed'].notna().any():
        collapsed_count = df[df['collapsed'] == True].shape[0]
        print(f"Collapsed: {collapsed_count}/{len(df)}")
    
    # Checkpoint selection stats
    if 'checkpoint_epoch' in df.columns and df['checkpoint_epoch'].notna().any():
        print(f"\nCheckpoint epochs selected:")
        print(f"  Min: {df['checkpoint_epoch'].min():.0f}")
        print(f"  Max: {df['checkpoint_epoch'].max():.0f}")
        print(f"  Mean: {df['checkpoint_epoch'].mean():.1f}")
    
    # Best models by different metrics
    evaluated_df = df[df['evaluated'] == True]
    if len(evaluated_df) > 0:
        print(f"\n{'='*80}")
        print("BEST MODELS BY METRIC")
        print(f"{'='*80}")
        
        if 'sharpe_ratio' in evaluated_df.columns:
            best = evaluated_df.nlargest(3, 'sharpe_ratio')
            print(f"\nTop 3 by Sharpe Ratio:")
            for _, row in best.iterrows():
                dir_acc = f"{row['dir_acc']:.2%}" if 'dir_acc' in row else 'N/A'
                phase = row.get('phase', 'N/A')
                print(f"  {phase}/{row['experiment_name']}: Sharpe={row['sharpe_ratio']:.4f}, Acc={dir_acc}")
        
        if 'auc_roc' in evaluated_df.columns:
            best = evaluated_df.nlargest(3, 'auc_roc')
            print(f"\nTop 3 by AUC-ROC:")
            for _, row in best.iterrows():
                dir_acc = f"{row['dir_acc']:.2%}" if 'dir_acc' in row else 'N/A'
                phase = row.get('phase', 'N/A')
                print(f"  {phase}/{row['experiment_name']}: AUC={row['auc_roc']:.4f}, Acc={dir_acc}")
        
        if 'dir_acc' in evaluated_df.columns:
            best = evaluated_df.nlargest(3, 'dir_acc')
            print(f"\nTop 3 by Directional Accuracy:")
            for _, row in best.iterrows():
                phase = row.get('phase', 'N/A')
                ckpt = row.get('checkpoint_epoch', 'N/A')
                print(f"  {phase}/{row['experiment_name']}: dir_acc={row['dir_acc']:.4f} (epoch {ckpt})")
        
        if 'composite_score' in evaluated_df.columns:
            best = evaluated_df.nlargest(3, 'composite_score')
            print(f"\nTop 3 by Composite Score:")
            for _, row in best.iterrows():
                phase = row.get('phase', 'N/A')
                print(f"  {phase}/{row['experiment_name']}: Score={row['composite_score']:.4f}")
    
    # Print working vs collapsed analysis
    print_working_vs_collapsed_summary(df)
    
    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()
