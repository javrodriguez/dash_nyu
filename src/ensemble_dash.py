#!/usr/bin/env python3
"""
Ensemble DASH Training Script

This script runs multiple DASH training runs with data preprocessing and subsetting.
Based on the DASH + Phoenix implementation for Gene Regulatory Network prediction.

Usage:
    python3 ensemble_dash.py [options]

Parameters:
    --prior_matrix: Path to prior matrix CSV (default: external_data/DASH_original/ground_truth_simulator/clean_data/edge_prior_matrix_simu_350_noise_0.05.csv)
    --rna_data: Path to RNA data CSV (default: external_data/DASH_original/ground_truth_simulator/clean_data/simu_350genes_150samples_train_val.csv)
    --config_file: Path to DASH config file (default: src_base/config_simu_dash.cfg)
    --n_samples: Number of samples to use (default: 100)
    --n_genes: Number of genes to subset (default: 200)
    --n_runs: Number of DASH training runs (default: 10)
    --output_dir: Output directory name (default: ensemble_dash_run)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import subprocess
import time
import datetime
import json
import logging
import sys
from pathlib import Path

def load_prior_matrix(file_path):
    """Load prior matrix from CSV file."""
    print(f"Loading prior matrix from: {file_path}")
    try:
        prior_matrix = pd.read_csv(file_path, header=None)
        print(f"Prior matrix loaded: {prior_matrix.shape[0]}x{prior_matrix.shape[1]}")
        return prior_matrix
    except Exception as e:
        print(f"Error loading prior matrix: {e}")
        sys.exit(1)

def load_rna_data(file_path):
    """Load RNA data from CSV file."""
    print(f"Loading RNA data from: {file_path}")
    try:
        # RNA data has a header row with dimensions
        rna_data = pd.read_csv(file_path, header=None)
        print(f"RNA data loaded: {rna_data.shape[0]}x{rna_data.shape[1]}")
        return rna_data
    except Exception as e:
        print(f"Error loading RNA data: {e}")
        sys.exit(1)

def subset_prior_matrix(prior_matrix, n_genes, output_dir):
    """
    Subset prior matrix to n_genes while maintaining similar proportion of 0s and 1s.
    
    Args:
        prior_matrix: Original prior matrix
        n_genes: Number of genes to subset to
        output_dir: Output directory to save the subset matrix
    
    Returns:
        subset_matrix: Subset of the prior matrix
        selected_indices: Indices of selected genes
    """
    original_genes = prior_matrix.shape[0]
    
    if n_genes > original_genes:
        print(f"Error: n_genes ({n_genes}) is higher than the number of rows in prior matrix ({original_genes})")
        sys.exit(1)
    
    if n_genes == original_genes:
        print("n_genes equals prior matrix size, using full matrix")
        subset_matrix = prior_matrix
        selected_indices = list(range(original_genes))
    else:
        print(f"Subsetting prior matrix from {original_genes} to {n_genes} genes")
        
        # Calculate proportion of non-zero elements in original matrix
        non_zero_proportion = (prior_matrix != 0).sum().sum() / (prior_matrix.shape[0] * prior_matrix.shape[1])
        print(f"Original matrix non-zero proportion: {non_zero_proportion:.3f}")
        
        # Try to find a subset that maintains similar sparsity
        best_subset = None
        best_sparsity_diff = float('inf')
        selected_indices = None
        
        # Try multiple random subsets to find one with similar sparsity
        for attempt in range(100):
            # Randomly select n_genes rows and columns
            indices = np.random.choice(original_genes, n_genes, replace=False)
            indices.sort()  # Keep original order
            
            subset = prior_matrix.iloc[indices, indices]
            subset_sparsity = (subset != 0).sum().sum() / (subset.shape[0] * subset.shape[1])
            sparsity_diff = abs(subset_sparsity - non_zero_proportion)
            
            if sparsity_diff < best_sparsity_diff:
                best_sparsity_diff = sparsity_diff
                best_subset = subset
                selected_indices = indices
        
        subset_matrix = best_subset
        print(f"Selected subset with sparsity: {subset_sparsity:.3f} (diff: {best_sparsity_diff:.3f})")
    
    # Save subset matrix
    output_path = os.path.join(output_dir, 'prior_matrix.csv')
    subset_matrix.to_csv(output_path, index=False, header=False)
    print(f"Subset prior matrix saved to: {output_path}")
    
    return subset_matrix, selected_indices

def filter_rna_data(rna_data, selected_indices, n_samples, output_dir):
    """
    Filter RNA data to include only the selected genes and randomly sample trajectories.
    
    Args:
        rna_data: Original RNA data
        selected_indices: Indices of selected genes
        n_samples: Number of trajectories to sample
        output_dir: Output directory to save filtered data
    
    Returns:
        filtered_data: Filtered RNA data
    """
    print(f"Filtering RNA data to include {len(selected_indices)} selected genes and {n_samples} trajectories")
    
    # RNA data format: first row contains [dim, ntraj], then for each trajectory:
    # - dim rows of gene expression data (one row per gene)
    # - 1 row of time data
    # Total rows: 1 + ntraj * (dim + 1)
    
    # Read the header to get dimensions
    header = rna_data.iloc[0].values
    original_dim, ntraj = int(header[0]), int(header[1])
    
    print(f"Original data: {original_dim} genes, {ntraj} trajectories")
    print(f"Selected genes: {len(selected_indices)} genes")
    print(f"Requested samples: {n_samples} trajectories")
    
    # Check if n_samples is valid
    if n_samples > ntraj:
        print(f"Error: n_samples ({n_samples}) is higher than available trajectories ({ntraj})")
        sys.exit(1)
    
    # Randomly sample trajectories
    if n_samples < ntraj:
        selected_trajectories = np.random.choice(ntraj, n_samples, replace=False)
        selected_trajectories.sort()  # Keep original order
        print(f"Randomly selected trajectories: {selected_trajectories}")
    else:
        selected_trajectories = list(range(ntraj))
        print("Using all available trajectories")
    
    # Create new data structure
    filtered_data = []
    
    # Add new header with updated dimensions
    filtered_data.append([len(selected_indices), n_samples])
    
    # For each selected trajectory
    for traj_idx in selected_trajectories:
        # Calculate start and end indices for this trajectory
        traj_start = 1 + traj_idx * (original_dim + 1)  # Skip header
        gene_data_start = traj_start
        gene_data_end = traj_start + original_dim
        time_data_row = gene_data_end
        
        # Get gene data for this trajectory and filter to selected genes
        traj_gene_data = rna_data.iloc[gene_data_start:gene_data_end]
        filtered_traj_gene_data = traj_gene_data.iloc[selected_indices]
        
        # Get time data for this trajectory
        traj_time_data = rna_data.iloc[time_data_row:time_data_row+1]
        
        # Add filtered gene data and time data to result
        for _, row in filtered_traj_gene_data.iterrows():
            filtered_data.append(row.values.tolist())
        filtered_data.append(traj_time_data.iloc[0].values.tolist())
    
    # Convert to DataFrame and save
    filtered_df = pd.DataFrame(filtered_data)
    output_path = os.path.join(output_dir, 'rna_data.csv')
    filtered_df.to_csv(output_path, index=False, header=False)
    
    print(f"Saved filtered RNA data to {output_path}")
    print(f"Expected rows: 1 + {n_samples} * ({len(selected_indices)} + 1) = {1 + n_samples * (len(selected_indices) + 1)}")
    print(f"Actual rows: {len(filtered_data)}")
    
    return filtered_df

def setup_logging(output_dir):
    """Setup logging to both file and console."""
    log_file = os.path.join(output_dir, 'ensemble_log.txt')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def save_status(status_data, output_dir):
    """Save current status to a JSON file for monitoring."""
    status_file = os.path.join(output_dir, 'ensemble_status.json')
    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2, default=str)

def get_run_status(run_dir):
    """Check if a run has completed successfully."""
    if not os.path.exists(run_dir):
        return "not_started"
    
    # Look for the most recent output directory
    output_dirs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]
    if not output_dirs:
        return "running"
    
    # Get the most recent directory
    latest_dir = max(output_dirs, key=lambda x: os.path.getctime(os.path.join(run_dir, x)))
    latest_path = os.path.join(run_dir, latest_dir)
    
    # Check for comprehensive metrics file
    metrics_file = os.path.join(latest_path, 'comprehensive_metrics.csv')
    if os.path.exists(metrics_file):
        return "completed"
    else:
        return "running"

def print_status_summary(status_data, logger=None):
    """Print a formatted status summary."""
    separator = "="*80
    summary_lines = [
        "",
        separator,
        "ENSEMBLE STATUS SUMMARY",
        separator,
        f"Start time: {status_data['start_time']}",
        f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total runs: {status_data['total_runs']}",
        f"Completed runs: {status_data['completed_runs']}",
        f"Failed runs: {status_data['failed_runs']}",
        f"Current run: {status_data['current_run']}",
        f"Progress: {status_data['completed_runs']}/{status_data['total_runs']} ({status_data['completed_runs']/status_data['total_runs']*100:.1f}%)"
    ]
    
    if status_data['completed_runs'] > 0:
        elapsed = datetime.datetime.now() - status_data['start_time']
        avg_time_per_run = elapsed / status_data['completed_runs']
        remaining_runs = status_data['total_runs'] - status_data['completed_runs']
        estimated_remaining = avg_time_per_run * remaining_runs
        summary_lines.extend([
            f"Average time per run: {avg_time_per_run}",
            f"Estimated time remaining: {estimated_remaining}"
        ])
    
    summary_lines.extend([
        f"Status file: {status_data['output_dir']}/ensemble_status.json",
        separator
    ])
    
    summary_text = "\n".join(summary_lines)
    
    if logger:
        logger.info(summary_text)
    else:
        print(summary_text)

def create_temp_config_with_seed(original_config_path, seed, output_dir, run_number):
    """
    Create a configuration file with a specific random seed and save it permanently.
    
    Args:
        original_config_path: Path to original config file
        seed: Random seed to use
        output_dir: Output directory for config files
        run_number: Current run number for organization
    
    Returns:
        Path to saved config file
    """
    import shutil
    
    # Create configs directory if it doesn't exist
    configs_dir = os.path.join(output_dir, 'configs')
    os.makedirs(configs_dir, exist_ok=True)
    
    # Read original config
    with open(original_config_path, 'r') as f:
        config_content = f.read()
    
    # Replace or add seed line
    if 'seed = ' in config_content:
        # Replace existing seed
        import re
        config_content = re.sub(r'seed = \d+', f'seed = {seed}', config_content)
    else:
        # Add seed at the end
        config_content += f'\nseed = {seed}\n'
    
    # Create config file with descriptive name
    config_filename = f'run_{run_number}_seed_{seed}.cfg'
    config_path = os.path.join(configs_dir, config_filename)
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path

def run_dash_training(config_file, data_file, prior_file, output_dir, run_number, status_data, logger=None):
    """
    Run a single DASH training session with status tracking.
    
    Args:
        config_file: Path to DASH config file
        data_file: Path to RNA data file
        prior_file: Path to prior matrix file
        output_dir: Base output directory
        run_number: Current run number
        status_data: Status tracking dictionary
        logger: Logger instance for output
    """
    run_output_dir = os.path.join(output_dir, 'dash', f'run_{run_number}')
    
    # Generate seed for this run
    import random
    if run_number == 1:
        # First run always uses seed 42 for reproducibility
        unique_seed = 42
        if logger:
            logger.info(f"Using fixed seed 42 for run {run_number} (first run)")
        else:
            print(f"Using fixed seed 42 for run {run_number} (first run)")
    else:
        # Subsequent runs use random seeds for diversity
        unique_seed = random.randint(1, 1000000)
        if logger:
            logger.info(f"Generated unique seed {unique_seed} for run {run_number}")
        else:
            print(f"Generated unique seed {unique_seed} for run {run_number}")
    
    # Create config with unique seed
    config_path = create_temp_config_with_seed(config_file, unique_seed, output_dir, run_number)
    if logger:
        logger.info(f"Created config with seed {unique_seed}: {config_path}")
    else:
        print(f"Created config with seed {unique_seed}: {config_path}")
    
    separator = "="*60
    start_msg = f"\n{separator}\nStarting DASH training run {run_number}/{status_data['total_runs']}\nStart time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{separator}"
    
    if logger:
        logger.info(start_msg)
    else:
        print(start_msg)
    
    # Update status
    status_data['current_run'] = run_number
    status_data['run_start_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    save_status(status_data, output_dir)
    
    # Construct the command with config
    cmd = [
        'python3', 'src/train_DASH_simu.py',
        '--settings', config_path,
        '--data', data_file,
        '--prior', prior_file,
        '--output_dir', run_output_dir
    ]
    
    cmd_str = ' '.join(cmd)
    if logger:
        logger.info(f"Command: {cmd_str}")
    else:
        print(f"Command: {cmd_str}")
    
    try:
        # Run the DASH training and capture all output
        if logger:
            logger.info(f"Starting DASH training process for run {run_number}...")
        
        # Use subprocess.Popen to capture real-time output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1,
            universal_newlines=True
        )
        
        # Capture and log output in real-time
        training_output = []
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.strip()
                training_output.append(line)
                if logger:
                    logger.info(f"[RUN {run_number}] {line}")
                else:
                    print(f"[RUN {run_number}] {line}")
        
        # Wait for process to complete
        return_code = process.wait()
        
        # Check if run completed successfully
        run_status = get_run_status(run_output_dir)
        if run_status == "completed" and return_code == 0:
            success_msg = f"✅ DASH training run {run_number} completed successfully (return code: {return_code})"
            status_data['completed_runs'] += 1
            status_data['successful_runs'].append(run_number)
        else:
            success_msg = f"❌ DASH training run {run_number} failed or incomplete (return code: {return_code})"
            status_data['failed_runs'] += 1
            status_data['failed_runs_list'].append(run_number)
        
        if logger:
            logger.info(success_msg)
        else:
            print(success_msg)
        
        # Save training output to a separate file for this run
        run_log_file = os.path.join(output_dir, 'dash', f'run_{run_number}', 'training_output.log')
        os.makedirs(os.path.dirname(run_log_file), exist_ok=True)
        with open(run_log_file, 'w') as f:
            f.write('\n'.join(training_output))
        
        if logger:
            logger.info(f"Training output saved to: {run_log_file}")
        
        # Update status
        status_data['current_run'] = None
        status_data['run_start_time'] = None
        save_status(status_data, output_dir)
        
        # Print progress
        print_status_summary(status_data, logger)
        
        return run_status == "completed" and return_code == 0
        
    except Exception as e:
        error_msg = f"❌ Error in DASH training run {run_number}: {e}"
        
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        
        # Update status
        status_data['failed_runs'] += 1
        status_data['failed_runs_list'].append(run_number)
        status_data['current_run'] = None
        status_data['run_start_time'] = None
        save_status(status_data, output_dir)
        
        return False

def main():
    parser = argparse.ArgumentParser(description='Ensemble DASH Training')
    parser.add_argument('--prior_matrix', 
                       default='external_data/DASH_original/ground_truth_simulator/clean_data/edge_prior_matrix_simu_350_noise_0.05.csv',
                       help='Path to prior matrix CSV')
    parser.add_argument('--rna_data',
                       default='external_data/DASH_original/ground_truth_simulator/clean_data/simu_350genes_150samples_train_val.csv',
                       help='Path to RNA data CSV')
    parser.add_argument('--config_file',
                       default='src_base/config_simu_dash.cfg',
                       help='Path to DASH config file')

    parser.add_argument('--n_runs', type=int, default=10,
                       help='Number of DASH training runs')
    parser.add_argument('--output_dir', default='ensemble_dash_run',
                       help='Output directory name')
    
    args = parser.parse_args()
    
    # 1. Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Setup logging
    logger = setup_logging(args.output_dir)
    
    # 3. Load prior matrix
    logger.info("2. Loading prior matrix")
    prior_matrix = load_prior_matrix(args.prior_matrix)
    
    # 4. Load RNA data
    logger.info("3. Loading RNA data")
    rna_data = load_rna_data(args.rna_data)
    
    # 5. Log initial parameters
    header = "="*80
    initial_info = [
        header,
        "ENSEMBLE DASH TRAINING",
        header,
        f"Prior matrix: {args.prior_matrix}",
        f"RNA data: {args.rna_data}",
        f"Config file: {args.config_file}",
        f"Number of genes: {prior_matrix.shape[0]} (full dataset)",
        f"Number of samples: {rna_data.iloc[0,1]} (full dataset)",
        f"Number of runs: {args.n_runs}",
        f"Output directory: {args.output_dir}",
        header
    ]
    
    logger.info("\n".join(initial_info))
    
    # 6. Create output directory (already done above)
    logger.info(f"1. Creating output directory: {args.output_dir}")
    
    # 7. Use full prior matrix (no subsetting)
    logger.info("4. Using full prior matrix (no subsetting)")
    subset_prior = prior_matrix
    selected_indices = list(range(prior_matrix.shape[0]))
    
    # 8. Use full RNA data (no subsetting)
    logger.info("5. Using full RNA data (no subsetting)")
    filtered_rna = rna_data
    
    # Save the data files to the output directory
    logger.info("6. Saving data files to output directory")
    data_file = os.path.join(args.output_dir, 'rna_data.csv')
    prior_file = os.path.join(args.output_dir, 'prior_matrix.csv')
    
    # Save RNA data
    filtered_rna.to_csv(data_file, index=False, header=False)
    logger.info(f"Saved RNA data to: {data_file}")
    
    # Save prior matrix
    subset_prior.to_csv(prior_file, index=False, header=False)
    logger.info(f"Saved prior matrix to: {prior_file}")
    
    # 9. Create dash directory
    logger.info("7. Creating dash directory")
    dash_dir = os.path.join(args.output_dir, 'dash')
    os.makedirs(dash_dir, exist_ok=True)
    
    # 10. Initialize status tracking
    logger.info("8. Initializing status tracking")
    status_data = {
        'start_time': datetime.datetime.now(),
        'total_runs': args.n_runs,
        'completed_runs': 0,
        'failed_runs': 0,
        'current_run': None,
        'run_start_time': None,
        'successful_runs': [],
        'failed_runs_list': [],
        'output_dir': args.output_dir,
        'parameters': {
            'n_genes': prior_matrix.shape[0],
            'n_samples': int(rna_data.iloc[0,1]),
            'config_file': args.config_file
        }
    }
    save_status(status_data, args.output_dir)
    
    # 11. Run DASH training for each run
    logger.info("9. Starting DASH training runs")
    data_file = os.path.join(args.output_dir, 'rna_data.csv')
    prior_file = os.path.join(args.output_dir, 'prior_matrix.csv')
    
    for run_number in range(1, args.n_runs + 1):
        success = run_dash_training(args.config_file, data_file, prior_file, args.output_dir, run_number, status_data, logger)
        
        # Wait a bit between runs to avoid resource conflicts
        if run_number < args.n_runs:
            logger.info("Waiting 5 seconds before next run...")
            time.sleep(5)
    
    # 12. Final summary
    final_summary = [
        "",
        "="*80,
        "ENSEMBLE DASH TRAINING COMPLETED",
        "="*80,
        f"Total runs: {args.n_runs}",
        f"Successful runs: {status_data['completed_runs']}",
        f"Failed runs: {status_data['failed_runs']}",
        f"Success rate: {status_data['completed_runs']/args.n_runs*100:.1f}%",
        f"Output directory: {args.output_dir}",
        f"Individual run results: {args.output_dir}/dash/",
        f"Status file: {args.output_dir}/ensemble_status.json",
        f"Log file: {args.output_dir}/ensemble_log.txt",
        "="*80
    ]
    
    logger.info("\n".join(final_summary))

def check_status(output_dir):
    """Check the status of an ongoing ensemble run."""
    status_file = os.path.join(output_dir, 'ensemble_status.json')
    if not os.path.exists(status_file):
        print(f"❌ No status file found in {output_dir}")
        return
    
    with open(status_file, 'r') as f:
        status_data = json.load(f)
    
    print_status_summary(status_data)
    
    # Also show log file location
    log_file = os.path.join(output_dir, 'ensemble_log.txt')
    if os.path.exists(log_file):
        print(f"\n📋 Full log available at: {log_file}")
        print(f"💡 Use 'tail -f {log_file}' to monitor progress in real-time")

if __name__ == "__main__":
    import sys
    
    # Check if user wants to check status
    if len(sys.argv) > 1 and sys.argv[1] == "--check-status":
        if len(sys.argv) > 2:
            output_dir = sys.argv[2]
            check_status(output_dir)
        else:
            print("Usage: python3 ensemble_dash.py --check-status <output_dir>")
        sys.exit(0)
    
    main()
