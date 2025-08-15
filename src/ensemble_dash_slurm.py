#!/usr/bin/env python3
"""
SLURM-compatible Ensemble DASH Training Script

This script adapts the ensemble_dash.py pipeline for HPC systems with SLURM job scheduler.
It can run multiple DASH training runs in parallel using SLURM job arrays.

Usage:
    # Submit the ensemble job
    sbatch ensemble_dash_slurm.py --n_runs 50 --output_dir ensemble_results
    
    # Check status
    python ensemble_dash_slurm.py --check-status ensemble_results
    
    # Monitor progress
    tail -f ensemble_results/ensemble_log.txt
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
import glob
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
        rna_data = pd.read_csv(file_path, header=None)
        print(f"RNA data loaded: {rna_data.shape[0]}x{rna_data.shape[1]}")
        return rna_data
    except Exception as e:
        print(f"Error loading RNA data: {e}")
        sys.exit(1)

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

def create_config_with_seed(original_config_path, new_config_path, seed):
    """
    Create a new configuration file with a specific random seed.
    
    Args:
        original_config_path: Path to original config file
        new_config_path: Path to new config file
        seed: Random seed to use
    """
    # Read original config
    with open(original_config_path, 'r') as f:
        config_content = f.read()
    
    # Add or update seed in config
    if 'seed' in config_content:
        # Replace existing seed
        import re
        config_content = re.sub(r'seed\s*=\s*\d+', f'seed = {seed}', config_content)
    else:
        # Add seed to the end
        config_content += f'\nseed = {seed}\n'
    
    # Write new config file
    with open(new_config_path, 'w') as f:
        f.write(config_content)

def create_slurm_script(run_number, config_file, data_file, prior_file, output_dir, slurm_config):
    """
    Create a SLURM script for a single DASH training run.
    
    Args:
        run_number: Current run number
        config_file: Path to DASH config file
        data_file: Path to RNA data file
        prior_file: Path to prior matrix file
        output_dir: Output directory
        slurm_config: SLURM configuration dictionary
    
    Returns:
        Path to the created SLURM script
    """
    # Generate random seed for this run
    seed = np.random.randint(1, 1000000)
    
    # Create run-specific output directory
    run_dir = os.path.join(output_dir, 'dash', f'run_{run_number}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Create a run-specific config file with the seed
    run_config_file = os.path.join(run_dir, f'config_run_{run_number}.cfg')
    create_config_with_seed(config_file, run_config_file, seed)
    
    # Create SLURM script content
    slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name=dash_run_{run_number}
#SBATCH --output={run_dir}/slurm_%j.out
#SBATCH --error={run_dir}/slurm_%j.err
#SBATCH --time={slurm_config['time']}
#SBATCH --nodes={slurm_config['nodes']}
#SBATCH --ntasks-per-node={slurm_config['ntasks_per_node']}
#SBATCH --cpus-per-task={slurm_config['cpus_per_task']}
#SBATCH --mem={slurm_config['mem']}
#SBATCH --partition={slurm_config['partition']}

# Activate conda environment
{slurm_config['conda_activate']}

# Set random seed for reproducibility
export PYTHONHASHSEED={seed}

# Run DASH training
cd {os.getcwd()}
python src/train_DASH_simu.py \\
  --settings {run_config_file} \\
  --data {data_file} \\
  --prior {prior_file} \\
  --output_dir {run_dir}

# Check if training completed successfully
if [ -f "{run_dir}/comprehensive_metrics.csv" ]; then
    echo "Run {run_number} completed successfully"
    exit 0
else
    echo "Run {run_number} failed - no comprehensive_metrics.csv found"
    exit 1
fi
"""
    
    # Write SLURM script to file
    slurm_script_path = os.path.join(run_dir, f'slurm_run_{run_number}.sh')
    with open(slurm_script_path, 'w') as f:
        f.write(slurm_script_content)
    
    # Make script executable
    os.chmod(slurm_script_path, 0o755)
    
    return slurm_script_path

def submit_slurm_jobs(slurm_scripts, slurm_config, logger):
    """
    Submit SLURM jobs for all runs.
    
    Args:
        slurm_scripts: List of paths to SLURM scripts
        slurm_config: SLURM configuration dictionary
        logger: Logger instance
    
    Returns:
        List of job IDs
    """
    job_ids = []
    
    for i, script_path in enumerate(slurm_scripts, 1):
        try:
            # Submit job
            result = subprocess.run(['sbatch', script_path], 
                                  capture_output=True, text=True, check=True)
            
            # Extract job ID from output
            job_id = result.stdout.strip().split()[-1]
            job_ids.append(job_id)
            
            logger.info(f"Submitted job {i}/{len(slurm_scripts)}: {job_id}")
            
            # Small delay between submissions
            time.sleep(1)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit job {i}: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            return None
    
    return job_ids

def monitor_jobs(job_ids, output_dir, logger):
    """
    Monitor SLURM jobs and update status.
    
    Args:
        job_ids: List of SLURM job IDs
        output_dir: Output directory
        logger: Logger instance
    
    Returns:
        Dictionary with job status information
    """
    status_data = {
        'job_ids': job_ids,
        'total_jobs': len(job_ids),
        'completed_jobs': 0,
        'failed_jobs': 0,
        'running_jobs': 0,
        'pending_jobs': 0,
        'job_status': {}
    }
    
    for job_id in job_ids:
        try:
            # Check job status
            result = subprocess.run(['squeue', '-j', job_id, '--noheader'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                # Job is still in queue
                status_line = result.stdout.strip().split()
                if len(status_line) >= 5:
                    status = status_line[4]
                    status_data['job_status'][job_id] = status
                    
                    if status in ['R', 'CG']:
                        status_data['running_jobs'] += 1
                    elif status in ['PD', 'CF']:
                        status_data['pending_jobs'] += 1
                    elif status in ['F', 'CA', 'CD', 'TO']:
                        status_data['failed_jobs'] += 1
            else:
                # Job completed (not in queue anymore)
                status_data['job_status'][job_id] = 'COMPLETED'
                status_data['completed_jobs'] += 1
                
        except Exception as e:
            logger.error(f"Error checking status for job {job_id}: {e}")
            status_data['job_status'][job_id] = 'UNKNOWN'
    
    return status_data

def check_run_completion(output_dir, n_runs, logger):
    """
    Check which runs have completed successfully.
    
    Args:
        output_dir: Output directory
        n_runs: Total number of runs
        logger: Logger instance
    
    Returns:
        List of completed run numbers
    """
    completed_runs = []
    
    for run_number in range(1, n_runs + 1):
        run_dir = os.path.join(output_dir, 'dash', f'run_{run_number}')
        metrics_file = os.path.join(run_dir, 'comprehensive_metrics.csv')
        
        if os.path.exists(metrics_file):
            completed_runs.append(run_number)
            logger.info(f"Run {run_number} completed successfully")
        else:
            logger.info(f"Run {run_number} not yet completed")
    
    return completed_runs

def create_slurm_config():
    """Create default SLURM configuration."""
    return {
        'time': '4:00:00',           # Job time limit
        'nodes': 1,                   # Number of nodes
        'ntasks_per_node': 1,         # Tasks per node
        'cpus_per_task': 4,           # CPUs per task
        'mem': '40G',                  # Memory per node
        'partition': 'gpu8_short,gpu4_short',       # SLURM partition
        'conda_activate': 'source /gpfs/home/rodrij92/home_abl/miniconda3/etc/profile.d/conda.sh && conda activate DASH_NYU'  # Conda activation
    }

def main():
    parser = argparse.ArgumentParser(description='SLURM Ensemble DASH Training')
    parser.add_argument('--prior_matrix', 
                       default='data/sample_data/edge_prior_matrix_simu_350_noise_0.05.csv',
                       help='Path to prior matrix CSV')
    parser.add_argument('--rna_data',
                       default='data/sample_data/simu_350genes_150samples_train_val.csv',
                       help='Path to RNA data CSV')
    parser.add_argument('--config_file',
                       default='src/config_simu350_dash.cfg',
                       help='Path to DASH config file')
    parser.add_argument('--n_runs', type=int, default=10,
                       help='Number of DASH training runs')
    parser.add_argument('--output_dir', default='ensemble_dash_slurm',
                       help='Output directory name')
    parser.add_argument('--slurm_config', default=None,
                       help='Path to SLURM configuration JSON file')
    parser.add_argument('--submit_only', action='store_true',
                       help='Only submit jobs, do not monitor')
    parser.add_argument('--monitor_only', action='store_true',
                       help='Only monitor existing jobs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    # Load SLURM configuration
    if args.slurm_config and os.path.exists(args.slurm_config):
        with open(args.slurm_config, 'r') as f:
            slurm_config = json.load(f)
    else:
        slurm_config = create_slurm_config()
        logger.info("Using default SLURM configuration")
    
    if args.monitor_only:
        # Only monitor existing jobs
        status_file = os.path.join(args.output_dir, 'ensemble_status.json')
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status_data = json.load(f)
            job_ids = status_data.get('job_ids', [])
            if job_ids:
                logger.info("Monitoring existing jobs...")
                monitor_jobs(job_ids, args.output_dir, logger)
            else:
                logger.error("No job IDs found in status file")
        else:
            logger.error("No status file found")
        return
    
    # Load data
    logger.info("Loading data files...")
    prior_matrix = load_prior_matrix(args.prior_matrix)
    rna_data = load_rna_data(args.rna_data)
    
    # Save data files to output directory
    data_file = os.path.join(args.output_dir, 'rna_data.csv')
    prior_file = os.path.join(args.output_dir, 'prior_matrix.csv')
    
    rna_data.to_csv(data_file, index=False, header=False)
    prior_matrix.to_csv(prior_file, index=False, header=False)
    
    logger.info(f"Saved data files to {args.output_dir}")
    
    # Create dash directory
    dash_dir = os.path.join(args.output_dir, 'dash')
    os.makedirs(dash_dir, exist_ok=True)
    
    # Initialize status tracking
    status_data = {
        'start_time': datetime.datetime.now(),
        'total_runs': args.n_runs,
        'completed_runs': 0,
        'failed_runs': 0,
        'output_dir': args.output_dir,
        'parameters': {
            'n_genes': prior_matrix.shape[0],
            'n_samples': int(rna_data.iloc[0,1]),
            'config_file': args.config_file
        }
    }
    save_status(status_data, args.output_dir)
    
    # Create SLURM scripts for all runs
    logger.info(f"Creating SLURM scripts for {args.n_runs} runs...")
    slurm_scripts = []
    
    for run_number in range(1, args.n_runs + 1):
        script_path = create_slurm_script(
            run_number, args.config_file, data_file, prior_file, 
            args.output_dir, slurm_config
        )
        slurm_scripts.append(script_path)
        logger.info(f"Created SLURM script for run {run_number}")
    
    # Submit jobs
    logger.info("Submitting SLURM jobs...")
    job_ids = submit_slurm_jobs(slurm_scripts, slurm_config, logger)
    
    if job_ids is None:
        logger.error("Failed to submit jobs")
        return
    
    # Save job IDs to status file
    status_data['job_ids'] = job_ids
    save_status(status_data, args.output_dir)
    
    logger.info(f"Submitted {len(job_ids)} jobs: {job_ids}")
    
    if args.submit_only:
        logger.info("Jobs submitted. Use --monitor_only to check status later.")
        return
    
    # Monitor jobs
    logger.info("Monitoring job progress...")
    while True:
        job_status = monitor_jobs(job_ids, args.output_dir, logger)
        
        # Update status
        status_data.update(job_status)
        save_status(status_data, args.output_dir)
        
        # Check for completed runs
        completed_runs = check_run_completion(args.output_dir, args.n_runs, logger)
        status_data['completed_runs'] = len(completed_runs)
        
        # Print summary
        logger.info(f"Job Status: {job_status['running_jobs']} running, "
                   f"{job_status['pending_jobs']} pending, "
                   f"{job_status['completed_jobs']} completed, "
                   f"{job_status['failed_jobs']} failed")
        
        # Check if all jobs are done
        if job_status['completed_jobs'] + job_status['failed_jobs'] == len(job_ids):
            logger.info("All jobs completed!")
            break
        
        # Wait before next check
        time.sleep(30)
    
    # Final summary
    final_summary = [
        "",
        "="*80,
        "SLURM ENSEMBLE DASH TRAINING COMPLETED",
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
    """Check the status of an ongoing SLURM ensemble run."""
    status_file = os.path.join(output_dir, 'ensemble_status.json')
    if not os.path.exists(status_file):
        print(f"❌ No status file found in {output_dir}")
        return
    
    with open(status_file, 'r') as f:
        status_data = json.load(f)
    
    print(f"📊 Ensemble Status: {output_dir}")
    print(f"🕒 Start time: {status_data['start_time']}")
    print(f"📈 Total runs: {status_data['total_runs']}")
    print(f"✅ Completed runs: {status_data['completed_runs']}")
    print(f"❌ Failed runs: {status_data['failed_runs']}")
    
    if 'job_ids' in status_data:
        print(f"🔧 Job IDs: {status_data['job_ids']}")
        
        # Check current job status
        for job_id in status_data['job_ids']:
            try:
                result = subprocess.run(['squeue', '-j', job_id, '--noheader'], 
                                      capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    status_line = result.stdout.strip().split()
                    if len(status_line) >= 5:
                        status = status_line[4]
                        print(f"   Job {job_id}: {status}")
                else:
                    print(f"   Job {job_id}: COMPLETED")
            except:
                print(f"   Job {job_id}: UNKNOWN")
    
    # Show log file location
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
            print("Usage: python3 ensemble_dash_slurm.py --check-status <output_dir>")
        sys.exit(0)
    
    main()
