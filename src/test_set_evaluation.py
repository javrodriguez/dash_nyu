#!/usr/bin/env python3
"""
Comprehensive Test Set Evaluation Script
Evaluates both Base and DASH models on the test set and generates performance metrics and plots.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import argparse

# Add the current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datahandler import DataHandler
from PHX_base_model import ODENet
from csvreader import readcsv

def load_model(model_dir, device):
    """Load a trained model from the given directory."""
    print(f"Loading model from: {model_dir}")
    
    # Create model instance
    odenet = ODENet(device, 350, explicit_time=False, neurons=40, 
                    log_scale="linear", init_bias_y=0, init_sparsity=0.95)
    odenet.float()
    
    # Load model components
    try:
        odenet.inherit_params(f"{model_dir}/best_val_model.pt")
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

def plot_performance_comparison(base_metrics, dash_metrics, output_dir):
    """Create comprehensive performance comparison plots."""
    print("Creating performance comparison plots...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Test Set Performance Comparison: Base vs DASH Models', fontsize=16, fontweight='bold')
    
    # Colors
    base_color = '#ff7f0e'  # Orange for base
    dash_color = '#1f77b4'  # Blue for DASH
    
    # 1. Error Metrics Comparison (separate subplots for different scales)
    # MSE subplot
    ax1 = axes[0, 0]
    mse_values = [base_metrics['mse'], dash_metrics['mse']]
    model_names = ['Base', 'DASH']
    bars1 = ax1.bar(model_names, mse_values, color=[base_color, dash_color], alpha=0.8)
    ax1.set_ylabel('MSE (squared units)')
    ax1.set_title('Mean Squared Error')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(mse_values) * 1.2)
    # Add value labels on bars
    for bar, value in zip(bars1, mse_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2e}', ha='center', va='bottom', fontsize=9)
    
    # MAE subplot
    ax1b = axes[0, 1]
    mae_values = [base_metrics['mae'], dash_metrics['mae']]
    bars1b = ax1b.bar(model_names, mae_values, color=[base_color, dash_color], alpha=0.8)
    ax1b.set_ylabel('MAE (linear units)')
    ax1b.set_title('Mean Absolute Error')
    ax1b.grid(True, alpha=0.3)
    ax1b.set_ylim(0, max(mae_values) * 1.2)
    # Add value labels on bars
    for bar, value in zip(bars1b, mae_values):
        height = bar.get_height()
        ax1b.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2e}', ha='center', va='bottom', fontsize=9)
    
    # RMSE subplot
    ax1c = axes[0, 2]
    rmse_values = [base_metrics['rmse'], dash_metrics['rmse']]
    bars1c = ax1c.bar(model_names, rmse_values, color=[base_color, dash_color], alpha=0.8)
    ax1c.set_ylabel('RMSE (linear units)')
    ax1c.set_title('Root Mean Square Error')
    ax1c.grid(True, alpha=0.3)
    ax1c.set_ylim(0, max(rmse_values) * 1.2)
    # Add value labels on bars
    for bar, value in zip(bars1c, rmse_values):
        height = bar.get_height()
        ax1c.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. R-squared Comparison
    ax2 = axes[1, 0]
    r2_values = [base_metrics['r_squared'], dash_metrics['r_squared']]
    
    bars = ax2.bar(model_names, r2_values, color=[base_color, dash_color], alpha=0.8)
    ax2.set_ylabel('R² Value')
    ax2.set_title('Overall R² Comparison')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, r2_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 3. Per-gene R² Distribution
    ax3 = axes[1, 1]
    ax3.hist(base_metrics['gene_r2'], bins=30, alpha=0.7, label='Base Model', 
             color=base_color, density=True)
    ax3.hist(dash_metrics['gene_r2'], bins=30, alpha=0.7, label='DASH Model', 
             color=dash_color, density=True)
    ax3.set_xlabel('R² Value')
    ax3.set_ylabel('Density')
    ax3.set_title('Per-Gene R² Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance Summary Table
    ax4 = axes[1, 2]
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary table
    summary_data = [
        ['Metric', 'Base Model', 'DASH Model', 'Improvement'],
        ['Overall MSE', f"{base_metrics['mse']:.2e}", f"{dash_metrics['mse']:.2e}", 
         f"{((base_metrics['mse'] - dash_metrics['mse']) / base_metrics['mse'] * 100):.1f}%"],
        ['Overall R²', f"{base_metrics['r_squared']:.3f}", f"{dash_metrics['r_squared']:.3f}", 
         f"{((dash_metrics['r_squared'] - base_metrics['r_squared']) / base_metrics['r_squared'] * 100):.1f}%"],
        ['MAE', f"{base_metrics['mae']:.2e}", f"{dash_metrics['mae']:.2e}", 
         f"{((base_metrics['mae'] - dash_metrics['mae']) / base_metrics['mae'] * 100):.1f}%"],
        ['RMSE', f"{base_metrics['rmse']:.3f}", f"{dash_metrics['rmse']:.3f}", 
         f"{((base_metrics['rmse'] - dash_metrics['rmse']) / base_metrics['rmse'] * 100):.1f}%"],
        ['Mean Gene R²', f"{base_metrics['mean_gene_r2']:.3f}", f"{dash_metrics['mean_gene_r2']:.3f}", 
         f"{((dash_metrics['mean_gene_r2'] - base_metrics['mean_gene_r2']) / base_metrics['mean_gene_r2'] * 100):.1f}%"]
    ]
    
    table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0], 
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code the improvement column
    for i in range(1, len(summary_data)):
        if i < 5 and summary_data[i][3] != "":
            try:
                improvement = float(summary_data[i][3].replace('%', ''))
                if improvement > 0:
                    table[(i, 3)].set_facecolor('#90EE90')  # Light green
                else:
                    table[(i, 3)].set_facecolor('#FFB6C1')  # Light red
            except:
                pass
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/test_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance comparison plot saved to: {output_dir}/test_performance_comparison.png")

def save_metrics_to_csv(base_metrics, dash_metrics, output_dir):
    """Save detailed metrics to CSV files."""
    print("Saving metrics to CSV files...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Overall metrics
    overall_metrics = pd.DataFrame({
        'Metric': ['MSE', 'R²', 'MAE', 'RMSE', 'MAPE', 'Mean_Gene_R²', 'Std_Gene_R²'],
        'Base_Model': [
            base_metrics['mse'],
            base_metrics['r_squared'],
            base_metrics['mae'],
            base_metrics['rmse'],
            base_metrics['mape'],
            base_metrics['mean_gene_r2'],
            base_metrics['std_gene_r2']
        ],
        'DASH_Model': [
            dash_metrics['mse'],
            dash_metrics['r_squared'],
            dash_metrics['mae'],
            dash_metrics['rmse'],
            dash_metrics['mape'],
            dash_metrics['mean_gene_r2'],
            dash_metrics['std_gene_r2']
        ]
    })
    
    overall_metrics.to_csv(f'{output_dir}/overall_metrics.csv', index=False)
    
    # Per-gene metrics
    gene_metrics = pd.DataFrame({
        'Gene_Index': range(len(base_metrics['gene_mse'])),
        'Base_MSE': base_metrics['gene_mse'],
        'DASH_MSE': dash_metrics['gene_mse'],
        'Base_R2': base_metrics['gene_r2'],
        'DASH_R2': dash_metrics['gene_r2']
    })
    
    gene_metrics.to_csv(f'{output_dir}/per_gene_metrics.csv', index=False)
    
    print(f"Metrics saved to: {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Base and DASH models on test set')
    parser.add_argument('--base_model_dir', type=str, required=True,
                       help='Directory containing the base model files')
    parser.add_argument('--dash_model_dir', type=str, required=True,
                       help='Directory containing the DASH model files')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data CSV file')
    parser.add_argument('--output_dir', type=str, default='test_evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load test data
    test_data = load_test_data(args.test_data, device)
    
    # Load models
    base_model = load_model(args.base_model_dir, device)
    dash_model = load_model(args.dash_model_dir, device)
    
    if base_model is None or dash_model is None:
        print("Error: Failed to load one or both models")
        return
    
    # Evaluate models
    print("\n" + "="*50)
    print("EVALUATING BASE MODEL")
    print("="*50)
    base_predictions, base_targets, base_times = compute_trajectory_predictions(base_model, test_data)
    base_metrics = compute_metrics(base_predictions, base_targets)
    
    print("\n" + "="*50)
    print("EVALUATING DASH MODEL")
    print("="*50)
    dash_predictions, dash_targets, dash_times = compute_trajectory_predictions(dash_model, test_data)
    dash_metrics = compute_metrics(dash_predictions, dash_targets)
    
    # Print results
    print("\n" + "="*50)
    print("TEST SET PERFORMANCE SUMMARY")
    print("="*50)
    print(f"{'Metric':<15} {'Base Model':<15} {'DASH Model':<15} {'Improvement':<15}")
    print("-" * 60)
    print(f"{'MSE':<15} {base_metrics['mse']:<15.2e} {dash_metrics['mse']:<15.2e} "
          f"{((base_metrics['mse'] - dash_metrics['mse']) / base_metrics['mse'] * 100):<15.1f}%")
    print(f"{'R²':<15} {base_metrics['r_squared']:<15.3f} {dash_metrics['r_squared']:<15.3f} "
          f"{((dash_metrics['r_squared'] - base_metrics['r_squared']) / base_metrics['r_squared'] * 100):<15.1f}%")
    print(f"{'MAE':<15} {base_metrics['mae']:<15.2e} {dash_metrics['mae']:<15.2e} "
          f"{((base_metrics['mae'] - dash_metrics['mae']) / base_metrics['mae'] * 100):<15.1f}%")
    print(f"{'Mean Gene R²':<15} {base_metrics['mean_gene_r2']:<15.3f} {dash_metrics['mean_gene_r2']:<15.3f} "
          f"{((dash_metrics['mean_gene_r2'] - base_metrics['mean_gene_r2']) / base_metrics['mean_gene_r2'] * 100):<15.1f}%")
    
    # Create plots and save results
    plot_performance_comparison(base_metrics, dash_metrics, args.output_dir)
    save_metrics_to_csv(base_metrics, dash_metrics, args.output_dir)
    
    print(f"\nAll results saved to: {args.output_dir}/")
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main() 