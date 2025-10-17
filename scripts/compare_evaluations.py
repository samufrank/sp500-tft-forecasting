"""
Compare evaluation results across experiments.
Focus on finding models that make VARIED predictions (not regression collapse).

Usage: python scripts/compare_evaluations.py
"""

import os
import json
import pandas as pd

def load_evaluation(exp_name):
    """Load evaluation metrics for an experiment."""
    eval_path = os.path.join('experiments', exp_name, 'evaluation', 'evaluation_metrics.json')
    
    if not os.path.exists(eval_path):
        return None
    
    with open(eval_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def analyze_evaluations():
    """Analyze all evaluated experiments."""
    results = []
    
    # Scan all experiments
    exp_dir = 'experiments'
    for exp_name in os.listdir(exp_dir):
        metrics = load_evaluation(exp_name)
        
        if metrics is None:
            continue
        
        # Extract key info
        conf_matrix = metrics['financial_metrics']['confusion_matrix']
        
        # Check if model makes negative predictions
        # conf_matrix = [[TN, FP], [FN, TP]]
        # TN = predicted down, actual down
        predicted_down = conf_matrix[0][0] + conf_matrix[1][0]  # TN + FN
        total_predictions = sum(sum(row) for row in conf_matrix)
        
        results.append({
            'experiment': exp_name,
            'directional_acc': metrics['financial_metrics']['directional_accuracy'],
            'auc_roc': metrics['financial_metrics']['auc_roc'],
            'sharpe': metrics['financial_metrics']['sharpe_ratio'],
            'precision': metrics['financial_metrics']['precision'],
            'recall': metrics['financial_metrics']['recall'],
            'alpha': metrics['financial_metrics']['alpha'],
            'num_trades': metrics['financial_metrics']['num_trades'],
            'predicted_down': predicted_down,
            'pct_down_predictions': predicted_down / total_predictions if total_predictions > 0 else 0,
            'r2': metrics['statistical_metrics']['r2'],
            'rmse': metrics['statistical_metrics']['rmse'],
        })
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*100)
    print("EVALUATION COMPARISON - FINDING MODELS THAT ACTUALLY WORK")
    print("="*100)
    
    print(f"\nTotal evaluated experiments: {len(df)}")
    
    # Filter to models that make SOME negative predictions (not collapsed)
    working = df[df['predicted_down'] > 0].copy()
    collapsed = df[df['predicted_down'] == 0].copy()
    
    print(f"\nModels making varied predictions: {len(working)}")
    print(f"Models collapsed to always-positive: {len(collapsed)}")
    
    if len(working) > 0:
        print("\n" + "="*100)
        print("WORKING MODELS (sorted by AUC-ROC)")
        print("="*100)
        
        working_sorted = working.sort_values('auc_roc', ascending=False)
        
        print("\nTop 10 by AUC (discriminative power):")
        display_cols = ['experiment', 'auc_roc', 'directional_acc', 'sharpe', 
                        'pct_down_predictions', 'num_trades']
        print(working_sorted[display_cols].head(10).to_string(index=False))
        
        print("\n\nTop 10 by Directional Accuracy:")
        working_acc = working.sort_values('directional_acc', ascending=False)
        print(working_acc[display_cols].head(10).to_string(index=False))
        
        print("\n\nTop 10 by Sharpe Ratio:")
        working_sharpe = working.sort_values('sharpe', ascending=False)
        print(working_sharpe[display_cols].head(10).to_string(index=False))
        
        # Best overall (balanced)
        # Composite score: AUC (most important) + directional acc
        working['composite_score'] = working['auc_roc'] + (working['directional_acc'] - 0.5)
        working_composite = working.sort_values('composite_score', ascending=False)
        
        print("\n\nBest Overall (composite score = AUC + (dir_acc - 0.5)):")
        print(working_composite[display_cols + ['composite_score']].head(10).to_string(index=False))
        
        # Save results
        working.to_csv('working_models.csv', index=False)
        collapsed.to_csv('collapsed_models.csv', index=False)
        
        print(f"\n\nResults saved:")
        print(f"  working_models.csv - {len(working)} models making varied predictions")
        print(f"  collapsed_models.csv - {len(collapsed)} models stuck at mean")
        
        # Best model summary
        best = working_composite.iloc[0]
        print("\n" + "="*100)
        print("RECOMMENDED BASELINE MODEL")
        print("="*100)
        print(f"Experiment: {best['experiment']}")
        print(f"  AUC-ROC: {best['auc_roc']:.4f} (discriminative power)")
        print(f"  Directional Accuracy: {best['directional_acc']:.2%}")
        print(f"  Sharpe Ratio: {best['sharpe']:.4f}")
        print(f"  Precision: {best['precision']:.4f}")
        print(f"  Recall: {best['recall']:.4f}")
        print(f"  Alpha: {best['alpha']:.2%}")
        print(f"  Negative Predictions: {best['predicted_down']}/{best['num_trades']} ({best['pct_down_predictions']:.1%})")
        print(f"  R²: {best['r2']:.4f}")
        print("\nUse this as your baseline for Method 1 comparison!")
    
    else:
        print("\n WARNING: No working models found!")
        print("All models collapsed to predicting only positive returns.")
    
    if len(collapsed) > 0:
        print("\n" + "="*100)
        print("COLLAPSED MODELS (for reference)")
        print("="*100)
        print(f"\nShowing {min(5, len(collapsed))} collapsed models:")
        print(collapsed[['experiment', 'directional_acc', 'sharpe']].head().to_string(index=False))
    
    return df

if __name__ == "__main__":
    analyze_evaluations()
