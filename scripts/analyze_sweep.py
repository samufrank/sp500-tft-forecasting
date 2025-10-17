"""
Analyze hyperparameter sweep results.
Usage: python scripts/analyze_sweep.py
"""

import os
import json
import pandas as pd

def analyze_sweep():
    results = []
    
    # Scan all sweep experiments
    exp_dir = 'experiments'
    for exp_name in os.listdir(exp_dir):
        if not exp_name.startswith('sweep_'):
            continue
        
        # Load config and metrics
        config_path = os.path.join(exp_dir, exp_name, 'config.json')
        metrics_path = os.path.join(exp_dir, exp_name, 'final_metrics.json')
        
        if not (os.path.exists(config_path) and os.path.exists(metrics_path)):
            print(f"Skipping {exp_name} - incomplete")
            continue
        
        with open(config_path) as f:
            config = json.load(f)
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        # Extract key info
        results.append({
            'experiment': exp_name,
            'learning_rate': config['training']['learning_rate'],
            'hidden_size': config['architecture']['hidden_size'],
            'max_encoder_length': config['architecture']['max_encoder_length'],
            'dropout': config['architecture']['dropout'],
            'max_epochs': config['training']['max_epochs'],
            'best_val_loss': metrics['best_val_loss'],
            'total_epochs': metrics['total_epochs'],
            'early_stopped': metrics.get('early_stopped', False),
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by val loss
    df = df.sort_values('best_val_loss')
    
    print("\n" + "="*80)
    print("HYPERPARAMETER SWEEP RESULTS")
    print("="*80)
    print(f"\nTotal experiments: {len(df)}")
    print(f"\nTop 5 by validation loss:")
    print(df.head(5).to_string(index=False))
    
    print(f"\n\nBy Learning Rate:")
    print(df.groupby('learning_rate')['best_val_loss'].agg(['mean', 'min', 'count']))
    
    print(f"\n\nBy Hidden Size:")
    print(df.groupby('hidden_size')['best_val_loss'].agg(['mean', 'min', 'count']))
    
    print(f"\n\nBy Encoder Length:")
    print(df.groupby('max_encoder_length')['best_val_loss'].agg(['mean', 'min', 'count']))
    
    print(f"\n\nBy Dropout:")
    print(df.groupby('dropout')['best_val_loss'].agg(['mean', 'min', 'count']))
    
    # Save full results
    df.to_csv('sweep_results.csv', index=False)
    print(f"\n\nFull results saved to: sweep_results.csv")
    
    # Best model
    best = df.iloc[0]
    print(f"\n" + "="*80)
    print("BEST MODEL")
    print("="*80)
    print(f"Experiment: {best['experiment']}")
    print(f"Val Loss: {best['best_val_loss']:.6f}")
    print(f"Learning Rate: {best['learning_rate']}")
    print(f"Hidden Size: {best['hidden_size']}")
    print(f"Encoder Length: {best['max_encoder_length']}")
    print(f"Dropout: {best['dropout']}")
    print(f"Epochs: {best['total_epochs']}/{best['max_epochs']}")

    print(f"\n\nEvaluating top 3 models on test set...")
    for i in range(min(3, len(df))):
        exp = df.iloc[i]['experiment']
        print(f"\nEvaluating {exp}...")
        os.system(f"python train/evaluate_tft.py --experiment-name {exp}")
    
    return df

if __name__ == "__main__":
    analyze_sweep()
