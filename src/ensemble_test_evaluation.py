#!/usr/bin/env python3
"""
Ensemble Test Set Evaluation Pipeline
Evaluates all ensemble runs on the test set and generates comprehensive performance comparison.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import glob
from datetime import datetime
import argparse

# Add the current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datahandler import DataHandler
from PHX_base_model import ODENet
from csvreader import readcsv

def find_ensemble_runs(ensemble_dir):
    """Find all completed ensemble runs with model files."""
    runs = []
    
    # Look for run directories
    run_dirs = glob.glob(os.path.join(ensemble_dir, 'dash', 'run_*'))
    
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        run_num = run_name.split('_')[1]
        
        # Look for timestamp directories within each run
        timestamp_dirs = glob.glob(os.path.join(run_dir, '*'))
        timestamp_dirs = [d for d in timestamp_dirs if os.path.isdir(d)]
        
        if timestamp_dirs:
            # Use the first (and usually only) timestamp directory
            timestamp_dir = timestamp_dirs[0]
            
            # Check if model files exist
            model_files = [
                os.path.join(timestamp_dir, 'best_val_model_alpha_comb_prods.pt'),
                os.path.join(timestamp_dir, 'best_val_model_alpha_comb_sums.pt'),
                os.path.join(timestamp_dir, 'best_val_model_gene_multipliers.pt')
            ]
            
            if all(os.path.exists(f) for f in model_files):
                # Extract seed from config file
                config_files = glob.glob(os.path.join(ensemble_dir, 'configs', f'run_{run_num}_seed_*.cfg'))
                seed = None
                if config_files:
                    try:
                        with open(config_files[0], 'r') as f:
                            for line in f:
                                if line.strip().startswith('seed'):
                                    seed = int(line.split('=')[1].strip())
                                    break
                    except:
                        seed = None
                
                runs.append({
                    'run_name': run_name,
                    'run_num': run_num,
                    'timestamp_dir': timestamp_dir,
                    'seed': seed
                })
    
    return runs

def load_model(model_dir, device):
    """Load a trained model from the given directory."""
    print(f"Loading model from: {model_dir}")
    
    # Create model instance
    odenet = ODENet(device, 350, explicit_time=False, neurons=40, 
                    log_scale="linear", init_bias_y=0, init_sparsity=0.95)
    odenet.float()
    
    # Load model components
    try:
        # Try to load the complete model first (like the reference script)
        if os.path.exists(os.path.join(model_dir, 'best_val_model.pt')):
            odenet.inherit_params(os.path.join(model_dir, 'best_val_model.pt'))
        else:
            # Load individual components to reconstruct the model
            sums_module = torch.load(
                os.path.join(model_dir, 'best_val_model_sums.pt'), 
                map_location=device, weights_only=False
            )
            prods_module = torch.load(
                os.path.join(model_dir, 'best_val_model_prods.pt'), 
                map_location=device, weights_only=False
            )
            alpha_comb_sums = torch.load(
                os.path.join(model_dir, 'best_val_model_alpha_comb_sums.pt'), 
                map_location=device, weights_only=False
            )
            alpha_comb_prods = torch.load(
                os.path.join(model_dir, 'best_val_model_alpha_comb_prods.pt'), 
                map_location=device, weights_only=False
            )
            gene_mult = torch.load(
                os.path.join(model_dir, 'best_val_model_gene_multipliers.pt'), 
                map_location=device, weights_only=False
            )
            
            # Extract weights and assign to model components
            if hasattr(sums_module, 'linear_out'):
                odenet.net_sums.linear_out.weight = torch.nn.Parameter(sums_module.linear_out.weight)
                odenet.net_sums.linear_out.bias = torch.nn.Parameter(sums_module.linear_out.bias)
            else:
                # Alternative structure
                odenet.net_sums.linear_out.weight = torch.nn.Parameter(sums_module[1].odefunc.output_sums[1].sample_weights())
                odenet.net_sums.linear_out.bias = torch.nn.Parameter(sums_module[1].odefunc.output_sums[1].sample_bias())
            
            if hasattr(prods_module, 'linear_out'):
                odenet.net_prods.linear_out.weight = torch.nn.Parameter(prods_module.linear_out.weight)
                odenet.net_prods.linear_out.bias = torch.nn.Parameter(prods_module.linear_out.bias)
            else:
                # Alternative structure
                odenet.net_prods.linear_out.weight = torch.nn.Parameter(prods_module[1].odefunc.output_prods[1].sample_weights())
                odenet.net_prods.linear_out.bias = torch.nn.Parameter(prods_module[1].odefunc.output_prods[1].sample_bias())
            
            if hasattr(alpha_comb_sums, 'linear_out'):
                odenet.net_alpha_combine_sums.linear_out.weight = torch.nn.Parameter(alpha_comb_sums.linear_out.weight)
            else:
                odenet.net_alpha_combine_sums.linear_out.weight = torch.nn.Parameter(alpha_comb_sums[1].odefunc.output_sums[2].sample_weights())
            
            if hasattr(alpha_comb_prods, 'linear_out'):
                odenet.net_alpha_combine_prods.linear_out.weight = torch.nn.Parameter(alpha_comb_prods.linear_out.weight)
            else:
                odenet.net_alpha_combine_prods.linear_out.weight = torch.nn.Parameter(alpha_comb_prods[1].odefunc.output_prods[3].sample_weights())
            
            odenet.gene_multipliers = torch.nn.Parameter(gene_mult)
        
        print("Model loaded successfully!")
        return odenet
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_test_data(test_file, device):
    """Load and prepare test data."""
    print(f"Loading test data from: {test_file}")
    
    # Load data using the existing csvreader
    data_np, data_pt, t_np, t_pt, dim, ntraj, data_np_0noise, data_pt_0noise = readcsv(
        test_file, device, noise_to_add=0, scale_expression=1, log_scale=False
    )
    
    # Apply normalization manually (same as in DataHandler)
    max_val = 0
    for data in data_pt:
        if torch.max(torch.abs(data)) > max_val:
            max_val = torch.max(torch.abs(data))
    
    if max_val > 0:
        for i in range(ntraj):
            data_pt[i] = torch.div(data_pt[i], max_val)
            data_np[i] = data_np[i] / float(max_val)
    
    print(f"Test data loaded: {ntraj} trajectories, {dim} genes")
    print(f"Data normalized by factor: {max_val}")
    
    # Return the data directly instead of using DataHandler
    return {
        'data_pt': data_pt,
        'time_pt': t_pt,
        'ntraj': ntraj,
        'dim': dim,
        'device': device,
        'init_bias_y': 0
    }

def compute_trajectory_predictions(model, data_dict, method='dopri5'):
    """Compute model predictions for all test trajectories."""
    print("Computing trajectory predictions...")
    
    all_predictions = []
    all_targets = []
    all_times = []
    
    with torch.no_grad():
        for traj_idx in range(data_dict['ntraj']):
            # Get trajectory data
            trajectory_data = data_dict['data_pt'][traj_idx]
            trajectory_time = data_dict['time_pt'][traj_idx]
            
            # Predict trajectory
            predictions = []
            targets = []
            times = []
            
            for i in range(len(trajectory_data) - 1):
                # Input state
                current_state = trajectory_data[i:i+1]  # Add batch dimension
                current_time = trajectory_time[i:i+2]   # Current and next time
                
                # Target state
                target_state = trajectory_data[i+1:i+2]
                
                # Model prediction
                try:
                    from torchdiffeq import odeint_adjoint as odeint
                    odeint_result = odeint(model, current_state, current_time, method=method)
                    
                    if odeint_result is not None and len(odeint_result) > 1:
                        pred = odeint_result[1] + data_dict['init_bias_y']
                        predictions.append(pred.squeeze().cpu().numpy())
                        targets.append(target_state.squeeze().cpu().numpy())
                        times.append(current_time[1].cpu().numpy())
                    
                except Exception as e:
                    print(f"Error in trajectory {traj_idx}, step {i}: {e}")
                    continue
            
            if predictions:
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                all_times.extend(times)
    
    return np.array(all_predictions), np.array(all_targets), np.array(all_times)

def compute_metrics(predictions, targets):
    """Compute comprehensive performance metrics."""
    print("Computing performance metrics...")
    
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Basic MSE
    mse = np.mean((predictions - targets) ** 2)
    
    # R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))
    
    # Root Mean Square Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
    
    # Per-gene metrics
    gene_mse = np.mean((predictions - targets) ** 2, axis=0)
    gene_r2 = []
    for i in range(predictions.shape[1]):
        gene_pred = predictions[:, i]
        gene_target = targets[:, i]
        ss_res_gene = np.sum((gene_target - gene_pred) ** 2)
        ss_tot_gene = np.sum((gene_target - np.mean(gene_target)) ** 2)
        r2_gene = 1 - (ss_res_gene / ss_tot_gene) if ss_tot_gene > 0 else 0

        gene_r2.append(r2_gene)
    
    gene_r2 = np.array(gene_r2)
    
    metrics = {
        'mse': mse,
        'r_squared': r_squared,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'gene_mse': gene_mse,
        'gene_r2': gene_r2,
        'mean_gene_r2': np.mean(gene_r2),
        'std_gene_r2': np.std(gene_r2)
    }
    
    return metrics

def plot_ensemble_performance_comparison(all_metrics, output_dir):
    """Create comprehensive ensemble performance comparison plots."""
    print("Creating ensemble performance comparison plots...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    run_names = [f"Run {m['run_num']}" for m in all_metrics]
    seeds = [m['seed'] for m in all_metrics]
    mse_values = [m['metrics']['mse'] for m in all_metrics]
    r2_values = [m['metrics']['r_squared'] for m in all_metrics]
    mae_values = [m['metrics']['mae'] for m in all_metrics]
    rmse_values = [m['metrics']['rmse'] for m in all_metrics]
    mean_gene_r2_values = [m['metrics']['mean_gene_r2'] for m in all_metrics]
    
    # Sort by MSE (best performance first)
    sorted_indices = np.argsort(mse_values)
    run_names = [run_names[i] for i in sorted_indices]
    seeds = [seeds[i] for i in sorted_indices]
    mse_values = [mse_values[i] for i in sorted_indices]
    r2_values = [r2_values[i] for i in sorted_indices]
    mae_values = [mae_values[i] for i in sorted_indices]
    rmse_values = [rmse_values[i] for i in sorted_indices]
    mean_gene_r2_values = [mean_gene_r2_values[i] for i in sorted_indices]
    
    # Add ranking to run names
    run_names_with_rank = [f"{run_name}\n(#{i+1})" for i, run_name in enumerate(run_names)]
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Ensemble Test Set Performance Comparison', fontsize=12, fontweight='bold')
    
    # Colors for different runs - use a more distinct color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # 1. MSE Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(run_names_with_rank, mse_values, color=colors, alpha=0.8)
    ax1.set_ylabel('MSE (squared units)')
    ax1.set_title('Mean Squared Error by Run')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    # Add value labels on bars
    for bar, value in zip(bars1, mse_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. R² Comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(run_names_with_rank, r2_values, color=colors, alpha=0.8)
    ax2.set_ylabel('R² Value')
    ax2.set_title('Overall R² by Run')
    ax2.set_ylim(0, 1)  # R² range from 0 to 1 (no negative values)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    # Add value labels on bars
    for bar, value in zip(bars2, r2_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. MAE Comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(run_names_with_rank, mae_values, color=colors, alpha=0.8)
    ax3.set_ylabel('MAE (linear units)')
    ax3.set_title('Mean Absolute Error by Run')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    # Add value labels on bars
    for bar, value in zip(bars3, mae_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. Per-gene R² Distribution (box plot)
    ax4 = axes[1, 1]
    # Reorder gene_r2_data to match the sorted order
    gene_r2_data = [all_metrics[i]['metrics']['gene_r2'] for i in sorted_indices]
    bp = ax4.boxplot(gene_r2_data, tick_labels=run_names_with_rank, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_ylabel('R² Value')
    ax4.set_title('Per-Gene R² Distribution')
    ax4.set_ylim(0, 1)  # R² range from 0 to 1 (no negative values)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ensemble_test_performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Ensemble performance comparison plot saved to: {output_dir}/ensemble_test_performance_comparison.png")



def save_ensemble_metrics_to_csv(all_metrics, output_dir):
    """Save detailed ensemble metrics to CSV files."""
    print("Saving ensemble metrics to CSV files...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Overall metrics for all runs
    overall_metrics = []
    for metrics in all_metrics:
        overall_metrics.append({
            'Run': f"Run_{metrics['run_num']}",
            'Seed': metrics['seed'] if metrics['seed'] else 'N/A',
            'MSE': metrics['metrics']['mse'],
            'R2': metrics['metrics']['r_squared'],
            'MAE': metrics['metrics']['mae'],
            'RMSE': metrics['metrics']['rmse'],
            'MAPE': metrics['metrics']['mape'],
            'Mean_Gene_R2': metrics['metrics']['mean_gene_r2'],
            'Std_Gene_R2': metrics['metrics']['std_gene_r2']
        })
    
    overall_df = pd.DataFrame(overall_metrics)
    overall_df.to_csv(f'{output_dir}/ensemble_overall_metrics.csv', index=False)
    
    # Ensemble statistics
    ensemble_stats = {
        'Metric': ['MSE', 'R²', 'MAE', 'RMSE', 'MAPE', 'Mean_Gene_R²', 'Std_Gene_R²'],
        'Mean': [
            np.mean([m['metrics']['mse'] for m in all_metrics]),
            np.mean([m['metrics']['r_squared'] for m in all_metrics]),
            np.mean([m['metrics']['mae'] for m in all_metrics]),
            np.mean([m['metrics']['rmse'] for m in all_metrics]),
            np.mean([m['metrics']['mape'] for m in all_metrics]),
            np.mean([m['metrics']['mean_gene_r2'] for m in all_metrics]),
            np.mean([m['metrics']['std_gene_r2'] for m in all_metrics])
        ],
        'Std': [
            np.std([m['metrics']['mse'] for m in all_metrics]),
            np.std([m['metrics']['r_squared'] for m in all_metrics]),
            np.std([m['metrics']['mae'] for m in all_metrics]),
            np.std([m['metrics']['rmse'] for m in all_metrics]),
            np.std([m['metrics']['mape'] for m in all_metrics]),
            np.std([m['metrics']['mean_gene_r2'] for m in all_metrics]),
            np.std([m['metrics']['std_gene_r2'] for m in all_metrics])
        ],
        'Min': [
            np.min([m['metrics']['mse'] for m in all_metrics]),
            np.min([m['metrics']['r_squared'] for m in all_metrics]),
            np.min([m['metrics']['mae'] for m in all_metrics]),
            np.min([m['metrics']['rmse'] for m in all_metrics]),
            np.min([m['metrics']['mape'] for m in all_metrics]),
            np.min([m['metrics']['mean_gene_r2'] for m in all_metrics]),
            np.min([m['metrics']['std_gene_r2'] for m in all_metrics])
        ],
        'Max': [
            np.max([m['metrics']['mse'] for m in all_metrics]),
            np.max([m['metrics']['r_squared'] for m in all_metrics]),
            np.max([m['metrics']['mae'] for m in all_metrics]),
            np.max([m['metrics']['rmse'] for m in all_metrics]),
            np.max([m['metrics']['mape'] for m in all_metrics]),
            np.max([m['metrics']['mean_gene_r2'] for m in all_metrics]),
            np.max([m['metrics']['std_gene_r2'] for m in all_metrics])
        ]
    }
    
    stats_df = pd.DataFrame(ensemble_stats)
    stats_df.to_csv(f'{output_dir}/ensemble_statistics.csv', index=False)
    
    print(f"Ensemble metrics saved to: {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Evaluate all ensemble runs on test set')
    parser.add_argument('--ensemble_dir', type=str, required=True,
                       help='Directory containing the ensemble runs')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data CSV file')
    parser.add_argument('--output_dir', type=str, default='ensemble_test_evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Find all ensemble runs
    print(f"Finding ensemble runs in: {args.ensemble_dir}")
    runs = find_ensemble_runs(args.ensemble_dir)
    
    if not runs:
        print("No completed ensemble runs found!")
        return
    
    print(f"Found {len(runs)} ensemble runs: {[r['run_name'] for r in runs]}")
    
    # Load test data
    test_data = load_test_data(args.test_data, device)
    
    # Evaluate all runs
    all_metrics = []
    
    for run in runs:
        print(f"\n{'='*50}")
        print(f"EVALUATING {run['run_name'].upper()}")
        print(f"{'='*50}")
        
        # Load model
        model = load_model(run['timestamp_dir'], device)
        
        if model is None:
            print(f"Failed to load model for {run['run_name']}, skipping...")
            continue
        
        # Compute predictions and metrics
        predictions, targets, times = compute_trajectory_predictions(model, test_data)
        metrics = compute_metrics(predictions, targets)
        
        # Store results
        all_metrics.append({
            'run_name': run['run_name'],
            'run_num': run['run_num'],
            'seed': run['seed'],
            'timestamp_dir': run['timestamp_dir'],
            'metrics': metrics
        })
        
        # Print run results
        print(f"\n{run['run_name']} Results:")
        print(f"  MSE: {metrics['mse']:.2e}")
        print(f"  R²: {metrics['r_squared']:.3f}")
        print(f"  MAE: {metrics['mae']:.2e}")
        print(f"  RMSE: {metrics['rmse']:.3f}")
        print(f"  Mean Gene R²: {metrics['mean_gene_r2']:.3f}")
    
    if not all_metrics:
        print("No successful evaluations completed!")
        return
    
    # Print ensemble summary
    print(f"\n{'='*50}")
    print("ENSEMBLE TEST SET PERFORMANCE SUMMARY")
    print(f"{'='*50}")
    print(f"{'Run':<10} {'Seed':<8} {'MSE':<12} {'R²':<8} {'MAE':<12} {'Mean Gene R²':<15}")
    print("-" * 70)
    
    for metrics in all_metrics:
        print(f"{metrics['run_name']:<10} {str(metrics['seed']):<8} "
              f"{metrics['metrics']['mse']:<12.2e} {metrics['metrics']['r_squared']:<8.3f} "
              f"{metrics['metrics']['mae']:<12.2e} {metrics['metrics']['mean_gene_r2']:<15.3f}")
    
    # Calculate ensemble statistics
    mse_values = [m['metrics']['mse'] for m in all_metrics]
    r2_values = [m['metrics']['r_squared'] for m in all_metrics]
    mae_values = [m['metrics']['mae'] for m in all_metrics]
    mean_gene_r2_values = [m['metrics']['mean_gene_r2'] for m in all_metrics]
    
    print(f"\n{'='*50}")
    print("ENSEMBLE STATISTICS")
    print(f"{'='*50}")
    print(f"{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 65)
    print(f"{'MSE':<15} {np.mean(mse_values):<12.2e} {np.std(mse_values):<12.2e} "
          f"{np.min(mse_values):<12.2e} {np.max(mse_values):<12.2e}")
    print(f"{'R²':<15} {np.mean(r2_values):<12.3f} {np.std(r2_values):<12.3f} "
          f"{np.min(r2_values):<12.3f} {np.max(r2_values):<12.3f}")
    print(f"{'MAE':<15} {np.mean(mae_values):<12.2e} {np.std(mae_values):<12.2e} "
          f"{np.min(mae_values):<12.2e} {np.max(mae_values):<12.2e}")
    print(f"{'Mean Gene R²':<15} {np.mean(mean_gene_r2_values):<12.3f} {np.std(mean_gene_r2_values):<12.3f} "
          f"{np.min(mean_gene_r2_values):<12.3f} {np.max(mean_gene_r2_values):<12.3f}")
    
    # Create plots and save results
    plot_ensemble_performance_comparison(all_metrics, args.output_dir)
    save_ensemble_metrics_to_csv(all_metrics, args.output_dir)
    
    print(f"\nAll results saved to: {args.output_dir}/")
    print("Ensemble evaluation completed successfully!")

if __name__ == "__main__":
    main()
