#!/usr/bin/env python3
"""
Script to plot and compare sparsity metrics between DASH and base training runs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path

def load_comprehensive_metrics(file_path):
    """Load comprehensive metrics from CSV file."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} epochs from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_sparsity_comparison(dash_metrics, base_metrics, output_dir="sparsity_plots"):
    """Create comprehensive sparsity comparison plots."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DASH vs Base Training: Sparsity Metrics Comparison', fontsize=16, fontweight='bold')
    
    # Colors for the plots
    dash_color = '#1f77b4'  # Blue for DASH
    base_color = '#ff7f0e'  # Orange for base
    
    # 1. Proportion of Nonzero Weights
    ax1 = axes[0, 0]
    if dash_metrics is not None:
        ax1.plot(dash_metrics['epoch'], dash_metrics['proportion_nonzero'], 
                color=dash_color, linewidth=2, label='DASH (with pruning)', marker='o', markersize=4)
    if base_metrics is not None:
        ax1.plot(base_metrics['epoch'], base_metrics['proportion_nonzero'], 
                color=base_color, linewidth=2, label='Base (no pruning)', marker='s', markersize=4)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Proportion of Nonzero Weights')
    ax1.set_title('Sparsity Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # 2. Total Number of Nonzero Weights
    ax2 = axes[0, 1]
    if dash_metrics is not None:
        ax2.plot(dash_metrics['epoch'], dash_metrics['total_nonzero'], 
                color=dash_color, linewidth=2, label='DASH (with pruning)', marker='o', markersize=4)
    if base_metrics is not None:
        ax2.plot(base_metrics['epoch'], base_metrics['total_nonzero'], 
                color=base_color, linewidth=2, label='Base (no pruning)', marker='s', markersize=4)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Total Nonzero Weights')
    ax2.set_title('Number of Active Parameters')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. L1 Norm of Weights
    ax3 = axes[1, 0]
    if dash_metrics is not None:
        ax3.plot(dash_metrics['epoch'], dash_metrics['l1_norm'], 
                color=dash_color, linewidth=2, label='DASH (with pruning)', marker='o', markersize=4)
    if base_metrics is not None:
        ax3.plot(base_metrics['epoch'], base_metrics['l1_norm'], 
                color=base_color, linewidth=2, label='Base (no pruning)', marker='s', markersize=4)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('L1 Norm')
    ax3.set_title('Weight Magnitude (L1 Norm)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Training Loss Comparison
    ax4 = axes[1, 1]
    if dash_metrics is not None:
        ax4.plot(dash_metrics['epoch'], dash_metrics['train_loss'], 
                color=dash_color, linewidth=2, label='DASH (with pruning)', marker='o', markersize=4)
    if base_metrics is not None:
        ax4.plot(base_metrics['epoch'], base_metrics['train_loss'], 
                color=base_color, linewidth=2, label='Base (no pruning)', marker='s', markersize=4)
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Training Loss')
    ax4.set_title('Training Loss Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sparsity_comparison_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Overview plot saved to {output_dir}/sparsity_comparison_overview.png")
    
    # Create detailed sparsity analysis plot
    create_detailed_sparsity_plot(dash_metrics, base_metrics, output_dir)

def create_detailed_sparsity_plot(dash_metrics, base_metrics, output_dir):
    """Create a detailed sparsity analysis plot."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Detailed Sparsity Analysis', fontsize=16, fontweight='bold')
    
    dash_color = '#1f77b4'
    base_color = '#ff7f0e'
    
    # 1. Sparsity percentage over time
    ax1 = axes[0, 0]
    if dash_metrics is not None:
        sparsity_percent = (1 - dash_metrics['proportion_nonzero']) * 100
        ax1.plot(dash_metrics['epoch'], sparsity_percent, 
                color=dash_color, linewidth=2, label='DASH (with pruning)', marker='o', markersize=4)
    if base_metrics is not None:
        sparsity_percent = (1 - base_metrics['proportion_nonzero']) * 100
        ax1.plot(base_metrics['epoch'], sparsity_percent, 
                color=base_color, linewidth=2, label='Base (no pruning)', marker='s', markersize=4)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Sparsity (%)')
    ax1.set_title('Model Sparsity Percentage')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # 2. Weight statistics
    ax2 = axes[0, 1]
    if dash_metrics is not None:
        ax2.plot(dash_metrics['epoch'], dash_metrics['median_nonzero'], 
                color=dash_color, linewidth=2, label='DASH Median', marker='o', markersize=4)
    if base_metrics is not None:
        ax2.plot(base_metrics['epoch'], base_metrics['median_nonzero'], 
                color=base_color, linewidth=2, label='Base Median', marker='s', markersize=4)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Median Weight Value')
    ax2.set_title('Median Nonzero Weight Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Weight range
    ax3 = axes[1, 0]
    if dash_metrics is not None:
        ax3.fill_between(dash_metrics['epoch'], 
                        dash_metrics['min_nonzero'], 
                        dash_metrics['max_nonzero'], 
                        alpha=0.3, color=dash_color, label='DASH Range')
        ax3.plot(dash_metrics['epoch'], dash_metrics['min_nonzero'], 
                color=dash_color, linewidth=1, alpha=0.7)
        ax3.plot(dash_metrics['epoch'], dash_metrics['max_nonzero'], 
                color=dash_color, linewidth=1, alpha=0.7)
    if base_metrics is not None:
        ax3.fill_between(base_metrics['epoch'], 
                        base_metrics['min_nonzero'], 
                        base_metrics['max_nonzero'], 
                        alpha=0.3, color=base_color, label='Base Range')
        ax3.plot(base_metrics['epoch'], base_metrics['min_nonzero'], 
                color=base_color, linewidth=1, alpha=0.7)
        ax3.plot(base_metrics['epoch'], base_metrics['max_nonzero'], 
                color=base_color, linewidth=1, alpha=0.7)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Weight Value')
    ax3.set_title('Weight Value Range')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Validation loss comparison
    ax4 = axes[1, 1]
    if dash_metrics is not None and 'val_loss' in dash_metrics.columns:
        ax4.plot(dash_metrics['epoch'], dash_metrics['val_loss'], 
                color=dash_color, linewidth=2, label='DASH (with pruning)', marker='o', markersize=4)
    if base_metrics is not None and 'val_loss' in base_metrics.columns:
        ax4.plot(base_metrics['epoch'], base_metrics['val_loss'], 
                color=base_color, linewidth=2, label='Base (no pruning)', marker='s', markersize=4)
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Loss')
    ax4.set_title('Validation Loss Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/detailed_sparsity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed analysis plot saved to {output_dir}/detailed_sparsity_analysis.png")

def print_summary_statistics(dash_metrics, base_metrics):
    """Print summary statistics comparing the two training runs."""
    
    print("\n" + "="*60)
    print("SPARSITY COMPARISON SUMMARY")
    print("="*60)
    
    if dash_metrics is not None:
        print(f"\nDASH Training (with pruning):")
        print(f"  Total epochs: {len(dash_metrics)}")
        print(f"  Initial sparsity: {100*(1-dash_metrics['proportion_nonzero'].iloc[0]):.1f}%")
        print(f"  Final sparsity: {100*(1-dash_metrics['proportion_nonzero'].iloc[-1]):.1f}%")
        print(f"  Max sparsity achieved: {100*(1-dash_metrics['proportion_nonzero'].min()):.1f}%")
        print(f"  Final nonzero weights: {dash_metrics['total_nonzero'].iloc[-1]:,}")
        
        # Find epoch with maximum pruning
        max_pruning_epoch = dash_metrics.loc[dash_metrics['proportion_nonzero'].idxmin()]
        print(f"  Maximum pruning at epoch {max_pruning_epoch['epoch']:.0f}: {100*(1-max_pruning_epoch['proportion_nonzero']):.1f}% sparsity")
    
    if base_metrics is not None:
        print(f"\nBase Training (no pruning):")
        print(f"  Total epochs: {len(base_metrics)}")
        print(f"  Initial sparsity: {100*(1-base_metrics['proportion_nonzero'].iloc[0]):.1f}%")
        print(f"  Final sparsity: {100*(1-base_metrics['proportion_nonzero'].iloc[-1]):.1f}%")
        print(f"  Final nonzero weights: {base_metrics['total_nonzero'].iloc[-1]:,}")
    
    if dash_metrics is not None and base_metrics is not None:
        print(f"\nComparison:")
        final_dash_sparsity = 100*(1-dash_metrics['proportion_nonzero'].iloc[-1])
        final_base_sparsity = 100*(1-base_metrics['proportion_nonzero'].iloc[-1])
        print(f"  Final sparsity difference: {final_dash_sparsity - final_base_sparsity:.1f}%")
        
        # Compare training losses
        if 'train_loss' in dash_metrics.columns and 'train_loss' in base_metrics.columns:
            final_dash_loss = dash_metrics['train_loss'].iloc[-1]
            final_base_loss = base_metrics['train_loss'].iloc[-1]
            print(f"  Final training loss - DASH: {final_dash_loss:.6f}, Base: {final_base_loss:.6f}")
            print(f"  Loss ratio (DASH/Base): {final_dash_loss/final_base_loss:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Plot sparsity comparison between DASH and base training')
    parser.add_argument('--dash_metrics', type=str, required=True, 
                       help='Path to DASH comprehensive_metrics.csv file')
    parser.add_argument('--base_metrics', type=str, required=True, 
                       help='Path to base comprehensive_metrics.csv file')
    parser.add_argument('--output_dir', type=str, default='sparsity_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load metrics
    print("Loading DASH metrics...")
    dash_metrics = load_comprehensive_metrics(args.dash_metrics)
    
    print("Loading base metrics...")
    base_metrics = load_comprehensive_metrics(args.base_metrics)
    
    if dash_metrics is None and base_metrics is None:
        print("Error: No metrics files could be loaded!")
        return
    
    # Create plots
    print("Creating comparison plots...")
    plot_sparsity_comparison(dash_metrics, base_metrics, args.output_dir)
    
    # Print summary
    print_summary_statistics(dash_metrics, base_metrics)
    
    print(f"\nAll plots saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 