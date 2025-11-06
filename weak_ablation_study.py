"""
Automated ablation studies for weak supervision.

Experiments:
1. Vary number of clicks: (1+1), (3+3), (5+5), (10+10), (20+20)
2. Compare strategies: random, centroid, boundary
3. Compare with Part 1 fully supervised results
"""
import os
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def train_fully_supervised(model='unet', epochs=50):
    """Train a fully supervised model as baseline for comparison"""
    print(f"\n{'='*80}")
    print(f"Training FULLY SUPERVISED baseline")
    print(f"{'='*80}\n")
    
    cmd = [
        'python', 'train.py',
        '--dataset', 'PH2',
        '--model', model,
        '--loss', 'bce',  # Simple BCE for full supervision for similarity with point loss
        '--epochs', str(epochs),
        '--batch_size', '8',
        '--num_workers', '4'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running fully supervised baseline: {e}")
        return False


def get_fully_supervised_baseline():
    """Get the best fully supervised result for comparison"""
    results_dir = Path('results')
    
    if not results_dir.exists():
        print("⚠️  No fully supervised results found. Training baseline model...")
        if train_fully_supervised():
            return get_fully_supervised_baseline()  # Retry after training
        else:
            print("❌ Failed to train fully supervised baseline")
            return None
    
    # Find most recent fully supervised result
    best_result = None
    best_dice = 0.0
    
    for result_dir in results_dir.glob('PH2_unet_*'):
        if 'pos' in result_dir.name or 'neg' in result_dir.name:
            continue  # Skip weak supervision results
        
        results_file = result_dir / 'results.json'
        if not results_file.exists():
            results_file = result_dir / 'results.txt'
            if results_file.exists():
                # Parse old format
                with open(results_file, 'r') as f:
                    content = f.read()
                    # Try to extract Dice from text
                    import re
                    match = re.search(r'dice:\s*([\d.]+)', content)
                    if match:
                        dice = float(match.group(1))
                        if dice > best_dice:
                            best_dice = dice
                            best_result = {
                                'test_dice': dice,
                                'model': 'unet',
                                'supervision': 'full'
                            }
            continue
        
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        test_dice = data.get('test_metrics', {}).get('dice', 0.0)
        if test_dice > best_dice:
            best_dice = test_dice
            best_result = {
                'test_dice': test_dice,
                'test_iou': data['test_metrics'].get('iou', 0.0),
                'test_accuracy': data['test_metrics'].get('accuracy', 0.0),
                'test_sensitivity': data['test_metrics'].get('sensitivity', 0.0),
                'test_specificity': data['test_metrics'].get('specificity', 0.0),
                'model': data['args'].get('model', 'unet'),
                'supervision': 'full',
                'result_dir': str(result_dir)
            }
    
    if best_result:
        print(f"\n✓ Found fully supervised baseline:")
        print(f"  Test Dice: {best_result['test_dice']*100:.2f}%")
        return best_result
    else:
        print("\n⚠️  No fully supervised results found. Run:")
        print("  python train.py --dataset PH2 --model unet --epochs 50")
        return None


def run_experiment(n_pos, n_neg, strategy='random', model='unet', epochs=50):
    """Run a single weak supervision experiment"""
    print(f"\n{'='*80}")
    print(f"Running: {n_pos}+{n_neg} clicks, {strategy} strategy")
    print(f"{'='*80}\n")
    
    cmd = [
        'python', 'train_weak.py',
        '--dataset', 'PH2',
        '--model', model,
        '--n_pos_clicks', str(n_pos),
        '--n_neg_clicks', str(n_neg),
        '--click_strategy', strategy,
        '--epochs', str(epochs),
        '--batch_size', '8',
        '--num_workers', '4'  # Changed from 0 to 4
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        return False


def collect_results(results_base_dir='results'):
    """Collect all results from experiments"""
    results = []
    
    for result_dir in Path(results_base_dir).glob('*'):
        if not result_dir.is_dir():
            continue
        
        results_file = result_dir / 'results.json'
        if not results_file.exists():
            continue
        
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Extract key information
        args = data['args']
        test_metrics = data['test_metrics']
        
        results.append({
            'n_pos_clicks': args['n_pos_clicks'],
            'n_neg_clicks': args['n_neg_clicks'],
            'total_clicks': args['n_pos_clicks'] + args['n_neg_clicks'],
            'click_strategy': args['click_strategy'],
            'model': args['model'],
            'test_dice': test_metrics['dice'],
            'test_iou': test_metrics['iou'],
            'test_accuracy': test_metrics['accuracy'],
            'test_sensitivity': test_metrics['sensitivity'],
            'test_specificity': test_metrics['specificity'],
            'best_val_dice': data['best_val_dice'],
            'result_dir': str(result_dir)
        })
    
    return pd.DataFrame(results)


def plot_clicks_vs_performance(df, baseline_dice, save_path='ablation_clicks_vs_dice.png'):
    """Plot: Number of clicks vs Dice score"""
    plt.figure(figsize=(10, 6))
    
    # Group by total clicks and strategy
    for strategy in df['click_strategy'].unique():
        subset = df[df['click_strategy'] == strategy].sort_values('total_clicks')
        plt.plot(subset['total_clicks'], subset['test_dice'] * 100, 
                marker='o', label=strategy, linewidth=2, markersize=8)
    
    # Add horizontal line for fully supervised result (dynamic baseline)
    if baseline_dice:
        plt.axhline(y=baseline_dice * 100, color='red', linestyle='--', linewidth=2, 
                    label=f'Fully Supervised ({baseline_dice*100:.2f}%)')
    
    plt.xlabel('Total Number of Clicks (Positive + Negative)', fontsize=12)
    plt.ylabel('Test Dice Score (%)', fontsize=12)
    plt.title('Weak Supervision: Clicks vs Performance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_strategy_comparison(df, save_path='ablation_strategy_comparison.png'):
    """Plot: Compare different sampling strategies"""
    plt.figure(figsize=(10, 6))
    
    # Bar plot for each strategy at fixed number of clicks (e.g., 10 total)
    subset = df[df['total_clicks'] == 10]  # 5+5 clicks
    
    strategies = subset['click_strategy'].unique()
    metrics = ['test_dice', 'test_iou', 'test_accuracy']
    
    x = range(len(strategies))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [subset[subset['click_strategy'] == s][metric].values[0] * 100 
                 for s in strategies]
        plt.bar([pos + i * width for pos in x], values, width, 
               label=metric.replace('test_', '').upper())
    
    plt.xlabel('Sampling Strategy', fontsize=12)
    plt.ylabel('Score (%)', fontsize=12)
    plt.title('Strategy Comparison (5+5 clicks)', fontsize=14, fontweight='bold')
    plt.xticks([pos + width for pos in x], strategies)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_comparison_table(df, baseline_result, save_path='ablation_results_table.csv'):
    """Create a table comparing all experiments"""
    # Pivot table: clicks x strategy
    pivot = df.pivot_table(
        values='test_dice',
        index='total_clicks',
        columns='click_strategy',
        aggfunc='mean'
    )
    
    pivot = (pivot * 100).round(2)
    pivot.to_csv(save_path)
    print(f"✓ Saved: {save_path}")
    
    # Also create a nice formatted table
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS - Test Dice Score (%)")
    print("="*80)
    print(pivot.to_string())
    print("="*80)
    
    # Compare with fully supervised
    best_weak = df['test_dice'].max() * 100
    
    if baseline_result:
        fully_supervised = baseline_result['test_dice'] * 100
        gap = fully_supervised - best_weak
        
        print(f"\nBest Weak Supervision: {best_weak:.2f}%")
        print(f"Fully Supervised (Baseline): {fully_supervised:.2f}%")
        print(f"Performance Gap: {gap:.2f}%")
        print(f"Percentage of Fully Supervised: {(best_weak/fully_supervised)*100:.1f}%")
        
        # Find how many clicks needed to reach 90% of fully supervised
        threshold = 0.9 * fully_supervised
        sufficient = df[df['test_dice'] * 100 >= threshold]
        if len(sufficient) > 0:
            min_clicks = sufficient['total_clicks'].min()
            print(f"\nTo reach 90% of fully supervised ({threshold:.2f}%):")
            print(f"  Minimum clicks needed: {min_clicks}")
        else:
            print(f"\nNo configuration reached 90% of fully supervised ({threshold:.2f}%)")
    else:
        print(f"\nBest Weak Supervision: {best_weak:.2f}%")
        print("⚠️  No fully supervised baseline for comparison")
    
    print("="*80 + "\n")


def ablation_study_clicks():
    """Ablation Study 1: Vary number of clicks"""
    print("\n" + "="*80)
    print("ABLATION STUDY 1: Number of Clicks")
    print("="*80 + "\n")
    
    # Test different numbers of clicks
    click_configs = [
        (1, 1),    # Very sparse
        (2, 2),
        (3, 3),    # Sparse
        (4, 4),
        (5, 5),    # Medium
        (10, 10),  # Dense
        (20, 20),  # Very dense
    ]
    
    for n_pos, n_neg in click_configs:
        run_experiment(n_pos, n_neg, strategy='random', epochs=50)


def ablation_study_strategy():
    """Ablation Study 2: Compare sampling strategies"""
    print("\n" + "="*80)
    print("ABLATION STUDY 2: Sampling Strategies")
    print("="*80 + "\n")
    
    strategies = ['random', 'centroid', 'boundary']
    n_pos, n_neg = 5, 5  # Fixed number of clicks
    
    for strategy in strategies:
        run_experiment(n_pos, n_neg, strategy=strategy, epochs=50)


def quick_test():
    """Quick test with just one configuration"""
    print("\n" + "="*80)
    print("QUICK TEST")
    print("="*80 + "\n")
    
    run_experiment(n_pos=5, n_neg=5, strategy='random', epochs=30)


def main():
    import sys
    
    mode = sys.argv[1] if len(sys.argv) > 1 else 'help'
    
    if mode == 'baseline':
        # Train only the fully supervised baseline
        print("\n" + "="*80)
        print("TRAINING FULLY SUPERVISED BASELINE")
        print("="*80 + "\n")
        train_fully_supervised(epochs=50)
        return
    
    # Get or train fully supervised baseline
    print("\n" + "="*80)
    print("CHECKING FULLY SUPERVISED BASELINE")
    print("="*80)
    
    baseline_result = get_fully_supervised_baseline()
    
    if not baseline_result:
        print("\n⚠️  Training fully supervised baseline first...")
        if train_fully_supervised(epochs=50):
            baseline_result = get_fully_supervised_baseline()
        else:
            print("❌ Failed to create baseline. Continuing without comparison.")
            baseline_result = None
    
    # Run weak supervision experiments
    if mode == 'quick':
        quick_test()
    elif mode == 'clicks':
        ablation_study_clicks()
    elif mode == 'strategy':
        ablation_study_strategy()
    elif mode == 'full':
        ablation_study_clicks()
        ablation_study_strategy()
    elif mode == 'analyze':
        pass  # Only analyze existing results
    else:
        print("Usage:")
        print("  python run_ablation_study.py baseline   # Train fully supervised baseline only")
        print("  python run_ablation_study.py quick      # Quick test (5+5 clicks, 30 epochs)")
        print("  python run_ablation_study.py clicks     # Ablation on number of clicks")
        print("  python run_ablation_study.py strategy   # Ablation on sampling strategy")
        print("  python run_ablation_study.py full       # Full ablation study")
        print("  python run_ablation_study.py analyze    # Analyze existing results")
        return
    
    # Collect and visualize results
    print("\n" + "="*80)
    print("ANALYZING RESULTS")
    print("="*80 + "\n")
    
    df = collect_results()
    
    if len(df) == 0:
        print("No weak supervision results found!")
        return
    
    print(f"Found {len(df)} weak supervision experiments\n")
    
    # Create visualizations (pass baseline to plotting functions)
    baseline_dice = baseline_result['test_dice'] if baseline_result else None
    plot_clicks_vs_performance(df, baseline_dice)
    
    if len(df[df['total_clicks'] == 10]) >= 2:  # Need at least 2 strategies
        plot_strategy_comparison(df)
    
    create_comparison_table(df, baseline_result)
    
    # Save comprehensive comparison
    if baseline_result:
        comparison = {
            'fully_supervised': baseline_result,
            'weak_supervision_summary': {
                'num_experiments': len(df),
                'best_dice': df['test_dice'].max(),
                'best_config': df.loc[df['test_dice'].idxmax()].to_dict(),
                'all_results': df.to_dict('records')
            }
        }
        
        with open('weak_vs_full_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"✓ Saved: weak_vs_full_comparison.json\n")


if __name__ == '__main__':
    main()