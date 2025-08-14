#!/usr/bin/env python3
"""
Ensemble Binary Adjacency Matrices

This script creates ensembles of binary adjacency matrices from the best models
and compares them against the ground truth prior matrix.

Key Features:
1. Loads binary adjacency matrices from all ensemble runs
2. Implements multiple ensemble methods for signed networks:
   - Simple averaging
   - Majority voting
   - Threshold-based consensus
   - Confidence-weighted averaging
3. Compares ensemble matrices against ground truth
4. Generates comprehensive evaluation plots and metrics
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import argparse
from datetime import datetime
import json

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
                os.path.join(timestamp_dir, 'best_val_model_sums.pt'),
                os.path.join(timestamp_dir, 'best_val_model_prods.pt'),
                os.path.join(timestamp_dir, 'best_val_model_alpha_comb_sums.pt'),
                os.path.join(timestamp_dir, 'best_val_model_alpha_comb_prods.pt'),
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

def extract_grn_from_model(model_dir, device='cpu'):
    """Extract GRN adjacency matrix from a trained model."""
    try:
        import torch
        
        # Load model components
        sums_model = torch.load(os.path.join(model_dir, 'best_val_model_sums.pt'), 
                               map_location=device, weights_only=False)
        prods_model = torch.load(os.path.join(model_dir, 'best_val_model_prods.pt'), 
                                map_location=device, weights_only=False)
        alpha_comb_sums = torch.load(os.path.join(model_dir, 'best_val_model_alpha_comb_sums.pt'), 
                                    map_location=device, weights_only=False)
        alpha_comb_prods = torch.load(os.path.join(model_dir, 'best_val_model_alpha_comb_prods.pt'), 
                                     map_location=device, weights_only=False)
        gene_mult = torch.load(os.path.join(model_dir, 'best_val_model_gene_multipliers.pt'), 
                              map_location=device, weights_only=False)
        
        # Extract weights
        if hasattr(sums_model, 'linear_out'):
            sums_weight = sums_model.linear_out.weight.detach().numpy()
            sums_bias = sums_model.linear_out.bias.detach().numpy()
        else:
            sums_weight = sums_model[1].odefunc.output_sums[1].sample_weights().detach().numpy()
            sums_bias = sums_model[1].odefunc.output_sums[1].sample_bias().detach().numpy()
        
        if hasattr(prods_model, 'linear_out'):
            prods_weight = prods_model.linear_out.weight.detach().numpy()
            prods_bias = prods_model.linear_out.bias.detach().numpy()
        else:
            prods_weight = prods_model[1].odefunc.output_prods[1].sample_weights().detach().numpy()
            prods_bias = prods_model[1].odefunc.output_prods[1].sample_bias().detach().numpy()
        
        if hasattr(alpha_comb_sums, 'linear_out'):
            alpha_sums_weight = alpha_comb_sums.linear_out.weight.detach().numpy()
        else:
            alpha_sums_weight = alpha_comb_sums[1].odefunc.output_sums[2].sample_weights().detach().numpy()
        
        if hasattr(alpha_comb_prods, 'linear_out'):
            alpha_prods_weight = alpha_comb_prods.linear_out.weight.detach().numpy()
        else:
            alpha_prods_weight = alpha_comb_prods[1].odefunc.output_prods[3].sample_weights().detach().numpy()
        
        gene_multipliers = gene_mult.detach().numpy()
        
        # Compute GRN matrix
        # GRN = (alpha_sums_weight @ sums_weight + alpha_prods_weight @ prods_weight) * gene_multipliers.T
        grn_matrix = (alpha_sums_weight @ sums_weight + alpha_prods_weight @ prods_weight) * gene_multipliers.T
        
        return grn_matrix
        
    except Exception as e:
        print(f"Error extracting GRN from model: {e}")
        return None

def create_binary_matrix(grn_matrix, method='percentile', threshold=0.995):
    """Create binary matrix from GRN weights."""
    if method == 'percentile':
        # Use percentile threshold
        abs_weights = np.abs(grn_matrix)
        threshold_value = np.percentile(abs_weights, threshold * 100)
        binary_matrix = np.where(abs_weights >= threshold_value, np.sign(grn_matrix), 0)
    elif method == 'absolute':
        # Use absolute threshold
        binary_matrix = np.where(np.abs(grn_matrix) >= threshold, np.sign(grn_matrix), 0)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return binary_matrix

def ensemble_binary_matrices(binary_matrices, method='majority_vote', weights=None, threshold=0.5):
    """
    Ensemble binary matrices using different methods.
    
    Args:
        binary_matrices: List of binary matrices (-1, 0, 1)
        method: 'simple_average', 'majority_vote', 'threshold_consensus', 'confidence_weighted'
        weights: Model weights for confidence-weighted method
        threshold: Threshold for consensus methods
    
    Returns:
        ensemble_matrix: Ensembled binary matrix
        confidence_matrix: Confidence scores (for uncertainty analysis)
    """
    n_models = len(binary_matrices)
    matrix_shape = binary_matrices[0].shape
    
    if method == 'simple_average':
        # Simple averaging (problematic for signed networks)
        ensemble_matrix = np.mean(binary_matrices, axis=0)
        confidence_matrix = np.std(binary_matrices, axis=0)  # High std = low confidence
        
    elif method == 'majority_vote':
        # Majority voting for each interaction
        ensemble_matrix = np.zeros(matrix_shape)
        confidence_matrix = np.zeros(matrix_shape)
        
        for i in range(matrix_shape[0]):
            for j in range(matrix_shape[1]):
                values = [mat[i, j] for mat in binary_matrices]
                unique_values, counts = np.unique(values, return_counts=True)
                
                # Find most common value
                majority_value = unique_values[np.argmax(counts)]
                majority_count = np.max(counts)
                
                ensemble_matrix[i, j] = majority_value
                confidence_matrix[i, j] = majority_count / n_models  # Confidence = proportion of agreement
                
    elif method == 'threshold_consensus':
        # Threshold-based consensus
        ensemble_matrix = np.zeros(matrix_shape)
        confidence_matrix = np.zeros(matrix_shape)
        
        for i in range(matrix_shape[0]):
            for j in range(matrix_shape[1]):
                values = [mat[i, j] for mat in binary_matrices]
                
                # Count positive, negative, and zero values
                pos_count = sum(1 for v in values if v > 0)
                neg_count = sum(1 for v in values if v < 0)
                zero_count = sum(1 for v in values if v == 0)
                
                total_count = len(values)
                
                # Determine consensus based on threshold
                if pos_count / total_count >= threshold:
                    ensemble_matrix[i, j] = 1
                    confidence_matrix[i, j] = pos_count / total_count
                elif neg_count / total_count >= threshold:
                    ensemble_matrix[i, j] = -1
                    confidence_matrix[i, j] = neg_count / total_count
                else:
                    ensemble_matrix[i, j] = 0
                    confidence_matrix[i, j] = max(pos_count, neg_count, zero_count) / total_count
                    
    elif method == 'confidence_weighted':
        # Confidence-weighted averaging
        if weights is None:
            weights = np.ones(n_models) / n_models
        
        weights = np.array(weights) / np.sum(weights)  # Normalize
        
        ensemble_matrix = np.zeros(matrix_shape)
        confidence_matrix = np.zeros(matrix_shape)
        
        for i in range(matrix_shape[0]):
            for j in range(matrix_shape[1]):
                values = np.array([mat[i, j] for mat in binary_matrices])
                
                # Weighted average
                weighted_avg = np.sum(values * weights)
                ensemble_matrix[i, j] = np.sign(weighted_avg) if abs(weighted_avg) >= threshold else 0
                
                # Confidence based on weighted variance
                confidence_matrix[i, j] = 1 - np.sum(weights * (values - weighted_avg)**2)
                
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    return ensemble_matrix, confidence_matrix

def calculate_performance_weights(ensemble_dir, runs):
    """Calculate performance-based weights for ensemble combination."""
    print("Calculating performance-based weights...")
    
    weights = []
    valid_runs = []
    
    for run in runs:
        # Look for comprehensive metrics file
        metrics_file = os.path.join(run['timestamp_dir'], 'comprehensive_metrics.csv')
        if os.path.exists(metrics_file):
            try:
                metrics_df = pd.read_csv(metrics_file)
                # Get the best validation loss (minimum)
                best_val_loss = metrics_df['val_loss'].min()
                weights.append(best_val_loss)
                valid_runs.append(run)
                print(f"  {run['run_name']}: Best val_loss = {best_val_loss:.6f}")
            except Exception as e:
                print(f"  Error reading metrics for {run['run_name']}: {e}")
                continue
        else:
            print(f"  No metrics file found for {run['run_name']}")
            continue
    
    if not weights:
        print("No valid performance metrics found. Using uniform weights.")
        return [1.0/len(runs)] * len(runs), runs
    
    # Convert to numpy array and calculate softmax weights
    weights = np.array(weights)
    # Lower MSE = higher weight
    softmax_weights = np.exp(-weights)
    softmax_weights = softmax_weights / np.sum(softmax_weights)
    
    print("\nPerformance-based weights:")
    for i, (run, weight) in enumerate(zip(valid_runs, softmax_weights)):
        print(f"  {run['run_name']}: weight = {weight:.4f} (MSE = {weights[i]:.6f})")
    
    return softmax_weights.tolist(), valid_runs

def load_ground_truth_matrix(ensemble_dir):
    """Load the ground truth prior matrix."""
    prior_matrix_file = os.path.join(ensemble_dir, 'prior_matrix.csv')
    if os.path.exists(prior_matrix_file):
        prior_matrix = pd.read_csv(prior_matrix_file, index_col=0).values
        return prior_matrix
    else:
        print(f"Warning: Prior matrix not found at {prior_matrix_file}")
        return None

def evaluate_matrix_similarity(predicted_matrix, ground_truth_matrix):
    """Evaluate similarity between predicted and ground truth matrices."""
    # Ensure matrices have the same shape
    if predicted_matrix.shape != ground_truth_matrix.shape:
        print(f"Warning: Matrix shapes don't match. Predicted: {predicted_matrix.shape}, Ground truth: {ground_truth_matrix.shape}")
        return None
    
    # Flatten matrices for comparison
    pred_flat = predicted_matrix.flatten()
    gt_flat = ground_truth_matrix.flatten()
    
    # Calculate metrics
    metrics = {}
    
    # MSE
    metrics['mse'] = np.mean((pred_flat - gt_flat) ** 2)
    
    # MAE
    metrics['mae'] = np.mean(np.abs(pred_flat - gt_flat))
    
    # RMSE
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Correlation coefficient
    correlation = np.corrcoef(pred_flat, gt_flat)[0, 1]
    metrics['correlation'] = correlation if not np.isnan(correlation) else 0
    
    # Binary accuracy (for edge existence)
    pred_binary = (pred_flat != 0).astype(int)
    gt_binary = (gt_flat != 0).astype(int)
    metrics['binary_accuracy'] = np.mean(pred_binary == gt_binary)
    
    # Signed accuracy (for edge direction)
    pred_signed = np.sign(pred_flat)
    gt_signed = np.sign(gt_flat)
    metrics['signed_accuracy'] = np.mean(pred_signed == gt_signed)
    
    # Precision, Recall, F1 for edge existence
    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    fp = np.sum((pred_binary == 1) & (gt_binary == 0))
    fn = np.sum((pred_binary == 0) & (gt_binary == 1))
    
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
    
    return metrics

def plot_ensemble_comparison(ensemble_matrices, confidence_matrices, ground_truth, output_dir):
    """Create comprehensive comparison plots."""
    print("Creating ensemble comparison plots...")
    
    # Create figure with subplots
    n_methods = len(ensemble_matrices)
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(5 * (n_methods + 1), 10))
    
    # Plot ground truth
    im0 = axes[0, 0].imshow(ground_truth, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    axes[0, 0].set_title('Ground Truth', fontweight='bold')
    axes[0, 0].set_xlabel('Target Gene')
    axes[0, 0].set_ylabel('Regulator Gene')
    
    # Plot ensemble matrices
    for i, (method, matrix) in enumerate(ensemble_matrices.items()):
        im = axes[0, i + 1].imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        axes[0, i + 1].set_title(f'{method.replace("_", " ").title()}', fontweight='bold')
        axes[0, i + 1].set_xlabel('Target Gene')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[0, i + 1], shrink=0.8)
    
    # Plot confidence matrices
    for i, (method, conf_matrix) in enumerate(confidence_matrices.items()):
        im = axes[1, i + 1].imshow(conf_matrix, cmap='viridis', vmin=0, vmax=1, aspect='equal')
        axes[1, i + 1].set_title(f'{method.replace("_", " ").title()} Confidence', fontweight='bold')
        axes[1, i + 1].set_xlabel('Target Gene')
        axes[1, i + 1].set_ylabel('Regulator Gene')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, i + 1], shrink=0.8)
    
    # Hide the bottom-left subplot
    axes[1, 0].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ensemble_matrices_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Ensemble comparison plot saved to: {output_dir}/ensemble_matrices_comparison.png")

def plot_metrics_comparison(metrics_dict, output_dir):
    """Create metrics comparison plots."""
    print("Creating metrics comparison plots...")
    
    # Prepare data for plotting
    methods = list(metrics_dict.keys())
    metric_names = ['mse', 'mae', 'rmse', 'correlation', 'binary_accuracy', 'signed_accuracy', 'precision', 'recall', 'f1_score']
    
    # Create subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Ensemble Methods Performance Comparison', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metric_names):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        values = [metrics_dict[method][metric] for method in methods]
        bars = ax.bar(methods, values, color=['red', 'blue', 'green', 'orange'])
        
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ensemble_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics comparison plot saved to: {output_dir}/ensemble_metrics_comparison.png")

def save_results(ensemble_matrices, confidence_matrices, metrics_dict, output_dir):
    """Save all results to files."""
    print("Saving results...")
    
    # Save ensemble matrices
    for method, matrix in ensemble_matrices.items():
        np.save(f'{output_dir}/ensemble_matrix_{method}.npy', matrix)
        pd.DataFrame(matrix).to_csv(f'{output_dir}/ensemble_matrix_{method}.csv')
    
    # Save confidence matrices
    for method, conf_matrix in confidence_matrices.items():
        np.save(f'{output_dir}/confidence_matrix_{method}.npy', conf_matrix)
        pd.DataFrame(conf_matrix).to_csv(f'{output_dir}/confidence_matrix_{method}.csv')
    
    # Save metrics
    metrics_df = pd.DataFrame(metrics_dict).T
    metrics_df.to_csv(f'{output_dir}/ensemble_metrics.csv')
    
    # Save summary
    summary = {
        'creation_time': datetime.now().isoformat(),
        'n_models': len(ensemble_matrices),
        'methods_used': list(ensemble_matrices.keys()),
        'best_method_mse': min(metrics_dict.items(), key=lambda x: x[1]['mse'])[0],
        'best_method_f1': max(metrics_dict.items(), key=lambda x: x[1]['f1_score'])[0],
        'best_method_correlation': max(metrics_dict.items(), key=lambda x: x[1]['correlation'])[0]
    }
    
    with open(f'{output_dir}/ensemble_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to: {output_dir}/")
    
    # Print summary
    print("\n" + "="*60)
    print("ENSEMBLE ADJACENCY MATRICES SUMMARY")
    print("="*60)
    print(f"Best method by MSE: {summary['best_method_mse']}")
    print(f"Best method by F1: {summary['best_method_f1']}")
    print(f"Best method by Correlation: {summary['best_method_correlation']}")
    print("\nDetailed metrics:")
    for method, metrics in metrics_dict.items():
        print(f"\n{method}:")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  Correlation: {metrics['correlation']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Binary Accuracy: {metrics['binary_accuracy']:.4f}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Ensemble binary adjacency matrices from ensemble runs')
    parser.add_argument('--ensemble_dir', type=str, required=True,
                       help='Directory containing the ensemble runs')
    parser.add_argument('--output_dir', type=str, default='ensemble_adjacency_results',
                       help='Output directory for results')
    parser.add_argument('--binary_method', type=str, default='percentile',
                       choices=['percentile', 'absolute'],
                       help='Method for creating binary matrices')
    parser.add_argument('--binary_threshold', type=float, default=0.995,
                       help='Threshold for binary matrix creation')
    parser.add_argument('--ensemble_threshold', type=float, default=0.5,
                       help='Threshold for consensus methods')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find ensemble runs
    print(f"Finding ensemble runs in: {args.ensemble_dir}")
    runs = find_ensemble_runs(args.ensemble_dir)
    
    if not runs:
        print("No completed ensemble runs found!")
        return 1
    
    print(f"Found {len(runs)} ensemble runs")
    
    # Calculate performance weights
    weights, valid_runs = calculate_performance_weights(args.ensemble_dir, runs)
    
    # Extract binary matrices from all runs
    print("\nExtracting binary adjacency matrices...")
    binary_matrices = []
    successful_runs = []
    
    for i, run in enumerate(valid_runs):
        print(f"  Processing {run['run_name']}...")
        
        # Extract GRN matrix
        grn_matrix = extract_grn_from_model(run['timestamp_dir'])
        if grn_matrix is None:
            print(f"    Failed to extract GRN from {run['run_name']}")
            continue
        
        # Create binary matrix
        binary_matrix = create_binary_matrix(grn_matrix, args.binary_method, args.binary_threshold)
        binary_matrices.append(binary_matrix)
        successful_runs.append(run)
        
        print(f"    Successfully created binary matrix with {np.sum(binary_matrix != 0)} edges")
    
    if not binary_matrices:
        print("No successful binary matrix extraction!")
        return 1
    
    print(f"\nSuccessfully extracted {len(binary_matrices)} binary matrices")
    
    # Load ground truth matrix
    ground_truth = load_ground_truth_matrix(args.ensemble_dir)
    if ground_truth is None:
        print("Warning: No ground truth matrix found. Skipping evaluation.")
        ground_truth = np.zeros_like(binary_matrices[0])
    else:
        # Ensure ground truth has the same shape as predicted matrices
        if ground_truth.shape != binary_matrices[0].shape:
            print(f"Warning: Ground truth shape {ground_truth.shape} doesn't match predicted shape {binary_matrices[0].shape}")
            print("Resizing ground truth to match predicted matrix...")
            # Create a new matrix of the correct size
            new_ground_truth = np.zeros_like(binary_matrices[0])
            # Copy the smaller matrix into the larger one
            min_size = min(ground_truth.shape[0], new_ground_truth.shape[0])
            new_ground_truth[:min_size, :min_size] = ground_truth[:min_size, :min_size]
            ground_truth = new_ground_truth
    
    # Create ensembles using different methods
    print("\nCreating ensemble matrices...")
    ensemble_methods = ['simple_average', 'majority_vote', 'threshold_consensus', 'confidence_weighted']
    
    ensemble_matrices = {}
    confidence_matrices = {}
    metrics_dict = {}
    
    for method in ensemble_methods:
        print(f"  Creating {method} ensemble...")
        
        # Create ensemble
        if method == 'confidence_weighted':
            ensemble_matrix, confidence_matrix = ensemble_binary_matrices(
                binary_matrices, method, weights[:len(binary_matrices)], args.ensemble_threshold
            )
        else:
            ensemble_matrix, confidence_matrix = ensemble_binary_matrices(
                binary_matrices, method, threshold=args.ensemble_threshold
            )
        
        ensemble_matrices[method] = ensemble_matrix
        confidence_matrices[method] = confidence_matrix
        
        # Evaluate against ground truth
        metrics = evaluate_matrix_similarity(ensemble_matrix, ground_truth)
        metrics_dict[method] = metrics
        
        print(f"    {method}: MSE = {metrics['mse']:.6f}, F1 = {metrics['f1_score']:.4f}")
    
    # Create plots and save results
    plot_ensemble_comparison(ensemble_matrices, confidence_matrices, ground_truth, args.output_dir)
    plot_metrics_comparison(metrics_dict, args.output_dir)
    save_results(ensemble_matrices, confidence_matrices, metrics_dict, args.output_dir)
    
    print(f"\nAll results saved to: {args.output_dir}/")
    
    return 0

if __name__ == "__main__":
    exit(main())
