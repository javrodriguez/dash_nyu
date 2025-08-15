import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import glob
from pathlib import Path

from matplotlib.colors import to_rgba
from matplotlib.ticker import ScalarFormatter
from adjustText import adjust_text

def darken(color, factor=0.6):
    rgba = np.array(to_rgba(color))
    return tuple(rgba[:3] * factor) + (rgba[3],)

def read_loss_csv(path):
    """Read and parse the comprehensive metrics CSV file."""
    data = pd.read_csv(path, header=0)
    if data.shape[1] == 13:
        data.columns = [
            'epoch', 'train_loss', 'val_loss', 'true_mse', 'true_r2', 'prior_loss',
            'total_nonzero', 'median_nonzero', 'min_nonzero', 'max_nonzero',
            'proportion_nonzero', 'l1_norm', 'time_hrs'
        ]
        # For plotting, drop 'epoch' and 'time_hrs'
        data = data.drop(columns=['epoch', 'time_hrs'])
        has_prior_loss = True
    elif data.shape[1] == 11:
        data.columns = [
            'train_loss', 'val_loss', 'true_mse', 'true_r2', 'prior_loss',
            'total_nonzero', 'median_nonzero', 'min_nonzero', 'max_nonzero',
            'proportion_nonzero', 'l1_norm'
        ]
        has_prior_loss = True
    elif data.shape[1] == 5:
        data.columns = ['train_loss', 'val_loss', 'true_mse', 'true_r2', 'prior_loss']
        has_prior_loss = True
    elif data.shape[1] == 4:
        data.columns = ['train_loss', 'val_loss', 'true_mse', 'true_r2']
        has_prior_loss = False
    else:
        raise ValueError(f"Unexpected number of columns: {data.shape[1]}")
    return data, has_prior_loss

def extract_seed_from_config(ensemble_dir, run_num):
    """Extract seed number from config file for a given run."""
    # First, try to find the config file in the run directory (new SLURM format)
    run_config_file = os.path.join(ensemble_dir, 'dash', f'run_{run_num}', f'config_run_{run_num}.cfg')
    
    if os.path.exists(run_config_file):
        try:
            with open(run_config_file, 'r') as f:
                for line in f:
                    if line.strip().startswith('seed ='):
                        return int(line.split('=')[1].strip())
        except:
            pass
    
    # Fallback: try the old format in configs directory
    config_dir = os.path.join(ensemble_dir, 'configs')
    config_pattern = f"run_{run_num}_seed_*.cfg"
    config_files = glob.glob(os.path.join(config_dir, config_pattern))
    
    if config_files:
        # Extract seed from filename (e.g., "run_1_seed_42.cfg" -> 42)
        config_filename = os.path.basename(config_files[0])
        seed_str = config_filename.split('_seed_')[1].split('.')[0]
        return int(seed_str)
    else:
        # Try to read from config file content in configs directory
        config_files = glob.glob(os.path.join(config_dir, f"run_{run_num}_*.cfg"))
        if config_files:
            try:
                with open(config_files[0], 'r') as f:
                    for line in f:
                        if line.strip().startswith('seed ='):
                            return int(line.split('=')[1].strip())
            except:
                pass
    return None

def find_ensemble_runs(ensemble_dir):
    """Find all completed ensemble runs and their metrics files."""
    runs = []
    dash_dir = os.path.join(ensemble_dir, 'dash')
    
    if not os.path.exists(dash_dir):
        print(f"Error: {dash_dir} does not exist")
        return runs
    
    # Find all run directories
    run_dirs = glob.glob(os.path.join(dash_dir, 'run_*'))
    run_dirs.sort()  # Sort to ensure consistent ordering
    
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        run_num = run_name.split('_')[1]
        
        # Find the timestamp directory inside the run
        timestamp_dirs = glob.glob(os.path.join(run_dir, '*'))
        # Filter out non-directories
        timestamp_dirs = [d for d in timestamp_dirs if os.path.isdir(d)]
        
        if not timestamp_dirs:
            continue
            
        # Use the first timestamp directory (should be only one)
        timestamp_dir = timestamp_dirs[0]
        metrics_file = os.path.join(timestamp_dir, 'comprehensive_metrics.csv')
        
        if os.path.exists(metrics_file):
            # Extract seed number
            seed = extract_seed_from_config(ensemble_dir, run_num)
            
            runs.append({
                'run_num': int(run_num),
                'run_name': run_name,
                'metrics_file': metrics_file,
                'timestamp_dir': timestamp_dir,
                'seed': seed
            })
    
    return runs

def clamp(val, vmin, vmax):
    """Clamp a value between min and max."""
    return max(vmin, min(val, vmax))

def save_summary_table(best_epochs, ensemble_dir):
    """Save a summary table with the performance information of the best model of each run."""
    if not best_epochs:
        print("No best epochs data to save")
        return
    
    # Sort by validation loss for ranking
    best_epochs_sorted = sorted(best_epochs, key=lambda x: x['val_loss'])
    
    # Create DataFrame for summary table - include ALL runs with comprehensive metrics
    summary_data = []
    for i, best in enumerate(best_epochs_sorted):
        row_data = {
            'Rank': i + 1,
            'Run_Number': best['run_num'],
            'Seed': best['seed'],
            'Best_Epoch': best['epoch'],
            'Training_MSE': f"{best['train_loss']:.2e}",
            'Validation_MSE': f"{best['val_loss']:.2e}",
            'True_MSE': f"{best['true_mse']:.2e}",
            'R2_Score': f"{best['true_r2']:.3f}"
        }
        
        # Add optional metrics if available
        if 'prior_loss' in best:
            row_data['Prior_Loss'] = f"{best['prior_loss']:.2e}"
        
        if 'proportion_nonzero' in best:
            row_data['Sparsity_Proportion'] = f"{best['proportion_nonzero']:.3f}"
            row_data['Total_Nonzero_Params'] = int(best['total_nonzero'])
            row_data['L1_Norm'] = f"{best['l1_norm']:.1f}"
            row_data['Sparsity_Percentage'] = f"{best['proportion_nonzero']*100:.1f}%"
        
        summary_data.append(row_data)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save as CSV
    csv_path = os.path.join(ensemble_dir, 'ensemble_best_models_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"Summary table saved to: {csv_path}")
    
    # Also save a more detailed version with additional statistics
    if len(best_epochs) > 1:
        val_losses = [b['val_loss'] for b in best_epochs]
        true_mses = [b['true_mse'] for b in best_epochs]
        true_r2s = [b['true_r2'] for b in best_epochs]
        
        stats_data = {
            'Metric': ['Validation_MSE', 'True_MSE', 'R2_Score'],
            'Mean': [np.mean(val_losses), np.mean(true_mses), np.mean(true_r2s)],
            'Std': [np.std(val_losses), np.std(true_mses), np.std(true_r2s)],
            'Min': [np.min(val_losses), np.min(true_mses), np.min(true_r2s)],
            'Max': [np.max(val_losses), np.max(true_mses), np.max(true_r2s)],
            'Median': [np.median(val_losses), np.median(true_mses), np.median(true_r2s)]
        }
        
        stats_df = pd.DataFrame(stats_data)
        stats_path = os.path.join(ensemble_dir, 'ensemble_statistics.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"Ensemble statistics saved to: {stats_path}")
    
    return summary_df

def main():
    # Parse arguments
    no_annotate = False
    args = sys.argv[1:]
    if '--no-annotate' in args:
        no_annotate = True
        args.remove('--no-annotate')
    
    if len(args) < 1:
        print("Usage: python compare_training_performance_ensemble.py <ensemble_dir> [--no-annotate]")
        print("Example: python compare_training_performance_ensemble.py ensemble_350genes_150samples_50runs")
        sys.exit(1)
    
    ensemble_dir = args[0]
    
    # Find all ensemble runs
    runs = find_ensemble_runs(ensemble_dir)
    
    if not runs:
        print(f"No completed runs found in {ensemble_dir}")
        sys.exit(1)
    
    print(f"Found {len(runs)} completed runs: {[r['run_name'] for r in runs]}")
    
    # Read all metrics files
    all_data = {}
    run_seeds = {}  # Store seed information for each run
    has_prior_loss = False
    
    for run in runs:
        try:
            data, run_has_prior = read_loss_csv(run['metrics_file'])
            all_data[run['run_num']] = data
            run_seeds[run['run_num']] = run['seed']
            has_prior_loss = has_prior_loss or run_has_prior
            print(f"Loaded {run['run_name']}: {len(data)} epochs (seed: {run['seed']})")
        except Exception as e:
            print(f"Error loading {run['metrics_file']}: {e}")
            continue
    
    if not all_data:
        print("No valid metrics files found")
        sys.exit(1)
    
    # Set up colors for multiple runs
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(18, 20))
    
    # Plot training and validation loss
    for i, (run_num, data) in enumerate(all_data.items()):
        epochs = range(1, len(data) + 1)
        color = colors[i]
        
        # Training loss
        ax1.plot(epochs, data['train_loss'], color=color, 
                label=f'Run {run_num} (train)', linewidth=1.5, alpha=0.8)
        
        # Validation loss
        if data['val_loss'].sum() > 0:
            ax1.plot(epochs, data['val_loss'], color=color, linestyle='--',
                    label=f'Run {run_num} (val)', linewidth=1.5, alpha=0.8)
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Error (MSE)', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Training and Validation Loss Over Epochs', fontsize=14, pad=10)
    
    # Plot true mean losses
    for i, (run_num, data) in enumerate(all_data.items()):
        epochs = range(1, len(data) + 1)
        color = colors[i]
        ax2.plot(epochs, data['true_mse'], color=color, 
                label=f'Run {run_num}', linewidth=1.5, alpha=0.8)
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('True Mean Losses (MSE)', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('True Mean Losses Over Epochs', fontsize=14, pad=10)
    
    # Plot R² values
    for i, (run_num, data) in enumerate(all_data.items()):
        epochs = range(1, len(data) + 1)
        color = colors[i]
        ax3.plot(epochs, data['true_r2'], color=color, 
                label=f'Run {run_num}', linewidth=1.5, alpha=0.8)
    
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('R² Values', fontsize=12)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('R² Values Over Epochs', fontsize=14, pad=10)
    
    # Plot prior loss (if available)
    if has_prior_loss:
        for i, (run_num, data) in enumerate(all_data.items()):
            if 'prior_loss' in data.columns:
                epochs = range(1, len(data) + 1)
                color = colors[i]
                ax4.plot(epochs, data['prior_loss'], color=color, 
                        label=f'Run {run_num}', linewidth=1.5, alpha=0.8)
        
        ax4.set_yscale('log')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Prior Loss', fontsize=12)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.set_title('Prior Loss Over Epochs', fontsize=14, pad=10)
    else:
        ax4.text(0.5, 0.5, 'Prior Loss Data\nNot Available', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax4.set_title('Prior Loss Over Epochs', fontsize=14, pad=10)
    
    # Plot sparsity (proportion_nonzero)
    has_sparsity = any('proportion_nonzero' in data.columns for data in all_data.values())
    if has_sparsity:
        for i, (run_num, data) in enumerate(all_data.items()):
            if 'proportion_nonzero' in data.columns:
                epochs = range(1, len(data) + 1)
                color = colors[i]
                ax5.plot(epochs, data['proportion_nonzero'], color=color, 
                        label=f'Run {run_num}', linewidth=1.5, alpha=0.8)
        
        ax5.set_xlabel('Epoch', fontsize=12)
        ax5.set_ylabel('Proportion Nonzero', fontsize=12)
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax5.grid(True, alpha=0.3)
        ax5.set_title('Sparsity (Proportion Nonzero) Over Epochs', fontsize=14, pad=10)
    else:
        ax5.text(0.5, 0.5, 'Sparsity Data\nNot Available', ha='center', va='center', 
                transform=ax5.transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax5.set_title('Sparsity (Proportion Nonzero) Over Epochs', fontsize=14, pad=10)
    
    # Plot best validation epochs summary
    best_epochs = []
    for run_num, data in all_data.items():
        if data['val_loss'].sum() > 0:
            idx = data['val_loss'].idxmin()
            best_epoch = int(np.asarray(idx).item()) + 1
            best_val_loss = data['val_loss'].min()
            
            # Get all available metrics at the best epoch
            best_metrics = {
                'run_num': run_num,
                'seed': run_seeds.get(run_num, None),
                'epoch': best_epoch,
                'val_loss': best_val_loss,
                'true_mse': data.loc[best_epoch-1, 'true_mse'],
                'true_r2': data.loc[best_epoch-1, 'true_r2'],
                'train_loss': data.loc[best_epoch-1, 'train_loss']
            }
            
            # Add optional metrics if available
            if 'prior_loss' in data.columns:
                best_metrics['prior_loss'] = data.loc[best_epoch-1, 'prior_loss']
            
            if 'proportion_nonzero' in data.columns:
                best_metrics['proportion_nonzero'] = data.loc[best_epoch-1, 'proportion_nonzero']
                best_metrics['total_nonzero'] = data.loc[best_epoch-1, 'total_nonzero']
                best_metrics['l1_norm'] = data.loc[best_epoch-1, 'l1_norm']
            
            best_epochs.append(best_metrics)
    
    if best_epochs:
        # Sort by validation loss
        best_epochs.sort(key=lambda x: x['val_loss'])
        
        # Create summary table
        run_nums = [b['run_num'] for b in best_epochs]
        val_losses = [b['val_loss'] for b in best_epochs]
        true_mses = [b['true_mse'] for b in best_epochs]
        true_r2s = [b['true_r2'] for b in best_epochs]
        
        # Plot validation losses
        bars = ax6.bar(range(len(run_nums)), val_losses, color=colors[:len(run_nums)], alpha=0.7)
        ax6.set_xlabel('Run Number', fontsize=12)
        ax6.set_ylabel('Best Validation Loss (MSE)', fontsize=12)
        ax6.set_title('Best Validation Loss by Run', fontsize=14, pad=10)
        ax6.set_xticks(range(len(run_nums)))
        ax6.set_xticklabels(run_nums)
        ax6.set_yscale('log')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, val_losses)):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2e}', ha='center', va='bottom', fontsize=8)
    else:
        ax6.text(0.5, 0.5, 'No validation data\navailable', ha='center', va='center', 
                transform=ax6.transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax6.set_title('Best Validation Loss by Run', fontsize=14, pad=10)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(ensemble_dir, 'ensemble_training_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    # Save summary table with best model performance information
    summary_df = save_summary_table(best_epochs, ensemble_dir)
    
    # Print summary statistics
    print("\n=== ENSEMBLE TRAINING SUMMARY ===")
    print(f"Total runs analyzed: {len(all_data)}")
    
    if best_epochs:
        print("\nBest validation performance by run:")
        for i, best in enumerate(best_epochs):
            print(f"  {i+1}. Run {best['run_num']}: Epoch {best['epoch']}, "
                  f"Val MSE: {best['val_loss']:.2e}, "
                  f"True MSE: {best['true_mse']:.2e}, "
                  f"R²: {best['true_r2']:.3f}")
        
        # Calculate ensemble statistics
        val_losses = [b['val_loss'] for b in best_epochs]
        true_mses = [b['true_mse'] for b in best_epochs]
        true_r2s = [b['true_r2'] for b in best_epochs]
        
        print(f"\nEnsemble Statistics:")
        print(f"  Validation Loss - Mean: {np.mean(val_losses):.2e}, Std: {np.std(val_losses):.2e}")
        print(f"  True MSE - Mean: {np.mean(true_mses):.2e}, Std: {np.std(true_mses):.2e}")
        print(f"  R² - Mean: {np.mean(true_r2s):.3f}, Std: {np.std(true_r2s):.3f}")

if __name__ == "__main__":
    main()
