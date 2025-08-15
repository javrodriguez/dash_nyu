# SLURM Ensemble DASH Training

This directory contains the SLURM-compatible version of the ensemble DASH training pipeline, designed for High-Performance Computing (HPC) systems with SLURM job scheduler.

## Overview

The `ensemble_dash_slurm.py` script enables parallel execution of multiple DASH training runs across multiple CPUs/cores on HPC systems. Instead of running training sequentially, this script:

- **Creates individual SLURM scripts** for each training run
- **Submits jobs in parallel** to the SLURM queue
- **Monitors job progress** in real-time
- **Tracks completion status** and provides comprehensive logging

## Key Features

### 🚀 **Parallel Execution**
- Run multiple DASH training instances simultaneously
- Utilize multiple CPU cores across the HPC cluster
- Significantly reduce total training time

### 📊 **Job Management**
- Automatic SLURM script generation
- Real-time job status monitoring
- Comprehensive logging and status tracking

### 🔧 **Flexible Configuration**
- Customizable SLURM parameters (time, memory, CPUs, etc.)
- Support for different HPC partitions
- Conda environment activation
- Configurable via JSON configuration files

### 📈 **Progress Tracking**
- Real-time monitoring of job status
- Completion tracking and success rate calculation
- Detailed logging for debugging and analysis

## Files

### Core Scripts
- **`ensemble_dash_slurm.py`**: Main SLURM ensemble script
- **`slurm_config_example.json`**: Example SLURM configuration file

### Configuration
- **`config_simu350_dash.cfg`**: DASH training configuration
- **`config_simu350_base.cfg`**: BASE training configuration

## Quick Start

### 0. Prerequisites

**Important**: Before running the SLURM ensemble script, make sure you have the conda environment activated:

```bash
# Activate the DASH_NYU conda environment
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate DASH_NYU

# Verify the environment is active
python -c "import pandas, torch, torchdiffeq; print('✅ All dependencies available')"
```

### 1. Basic Usage

```bash
# Submit ensemble with default settings (10 runs)
python src/ensemble_dash_slurm.py --n_runs 10 --output_dir ensemble_results

# Submit with custom parameters
python src/ensemble_dash_slurm.py \
  --n_runs 50 \
  --output_dir large_ensemble \
  --config_file src/config_simu350_dash.cfg
```

### 2. Custom SLURM Configuration

```bash
# Create custom SLURM config
cp src/slurm_config_example.json my_slurm_config.json
# Edit my_slurm_config.json with your HPC settings

# Use custom config
python src/ensemble_dash_slurm.py \
  --n_runs 20 \
  --slurm_config my_slurm_config.json \
  --output_dir custom_ensemble
```

### 3. Submit and Monitor Separately

```bash
# Submit jobs only
python src/ensemble_dash_slurm.py \
  --n_runs 30 \
  --submit_only \
  --output_dir my_ensemble

# Monitor existing jobs later
python src/ensemble_dash_slurm.py \
  --monitor_only \
  --output_dir my_ensemble
```

### 4. Check Status

```bash
# Check ensemble status
python src/ensemble_dash_slurm.py --check-status ensemble_results

# Monitor log in real-time
tail -f ensemble_results/ensemble_log.txt
```

## SLURM Configuration

### Default Configuration
```json
{
  "time": "24:00:00",
  "nodes": 1,
  "ntasks_per_node": 1,
  "cpus_per_task": 4,
  "mem": "8G",
  "partition": "default",
  "conda_activate": "source ~/miniconda3/etc/profile.d/conda.sh && conda activate DASH_NYU"
}
```

### Configuration Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `time` | Job time limit | `"24:00:00"` |
| `nodes` | Number of nodes per job | `1` |
| `ntasks_per_node` | Tasks per node | `1` |
| `cpus_per_task` | CPUs per task | `4` |
| `mem` | Memory per node | `"8G"` |
| `partition` | SLURM partition | `"default"` |


| `conda_activate` | Conda environment activation | `"source ~/conda.sh && conda activate DASH_NYU"` |

### Common HPC Configurations

#### **NYU HPC (Greene)**
```json
{
  "time": "24:00:00",
  "nodes": 1,
  "ntasks_per_node": 1,
  "cpus_per_task": 4,
  "mem": "8G",
  "partition": "gpu",
  "conda_activate": "source ~/miniconda3/etc/profile.d/conda.sh && conda activate DASH_NYU"
}
```

#### **Standard CPU Cluster**
```json
{
  "time": "12:00:00",
  "nodes": 1,
  "ntasks_per_node": 1,
  "cpus_per_task": 8,
  "mem": "16G",
  "partition": "cpu",
  "conda_activate": "source ~/miniconda3/etc/profile.d/conda.sh && conda activate DASH_NYU"
}
```

## Command Line Options

### Required Arguments
- `--n_runs`: Number of DASH training runs to execute in parallel

### Optional Arguments
- `--prior_matrix`: Path to prior matrix CSV (default: `data/sample_data/edge_prior_matrix_simu_350_noise_0.05.csv`)
- `--rna_data`: Path to RNA data CSV (default: `data/sample_data/simu_350genes_150samples_train_val.csv`)
- `--config_file`: Path to DASH config file (default: `src/config_simu350_dash.cfg`)
- `--output_dir`: Output directory name (default: `ensemble_dash_slurm`)
- `--slurm_config`: Path to SLURM configuration JSON file
- `--submit_only`: Only submit jobs, do not monitor
- `--monitor_only`: Only monitor existing jobs

### Status Checking
- `--check-status <output_dir>`: Check status of existing ensemble run

## Output Structure

```
ensemble_output/
├── ensemble_status.json          # Job status and metadata
├── ensemble_log.txt              # Detailed execution log
├── rna_data.csv                  # RNA data used for training
├── prior_matrix.csv              # Prior matrix used for training
└── dash/
    ├── run_1/
    │   ├── slurm_run_1.sh        # SLURM script for run 1
    │   ├── slurm_12345.out       # SLURM output
    │   ├── slurm_12345.err       # SLURM error log
    │   └── [training outputs]    # DASH training results
    ├── run_2/
    │   └── [training outputs]
    └── ...
```

## Monitoring and Debugging

### Real-time Monitoring
```bash
# Monitor log file
tail -f ensemble_results/ensemble_log.txt

# Check job status
squeue -u $USER

# Check specific job
squeue -j <job_id>
```

### Job Status Codes
- `PD`: Pending (waiting for resources)
- `R`: Running
- `CG`: Completing
- `F`: Failed
- `CA`: Cancelled
- `CD`: Completed

### Common Issues and Solutions

#### **Jobs Stuck in Pending**
- Check partition availability: `sinfo -p <partition>`
- Reduce resource requirements in SLURM config
- Use different partition: `--partition cpu`

#### **Jobs Failing**
- Check error logs: `cat run_X/slurm_*.err`
- Verify conda environment: `conda list | grep torch`
- Check data file paths and permissions

#### **Memory Issues**
- Increase memory in SLURM config: `"mem": "16G"`
- Reduce batch size in DASH config
- Use fewer CPUs per task

## Performance Optimization

### **Resource Allocation**
- **CPU-intensive**: Use more CPUs per task (`cpus_per_task: 8`)
- **Memory-intensive**: Increase memory allocation (`mem: "16G"`)
- **I/O-intensive**: Use local scratch directories

### **Scaling Considerations**
- **Small ensembles** (10-20 runs): Use single node
- **Large ensembles** (50+ runs): Consider job arrays or multiple submissions
- **Very large ensembles** (100+ runs): Split into multiple batches

### **Time Estimation**
- **Single run**: ~30-60 minutes (depending on data size)
- **10 runs**: ~1-2 hours (parallel execution)
- **50 runs**: ~2-4 hours (parallel execution)

## Integration with Existing Pipeline

The SLURM ensemble script is fully compatible with the existing DASH pipeline:

```bash
# Run SLURM ensemble
python src/ensemble_dash_slurm.py --n_runs 50 --output_dir slurm_ensemble

# Use results with existing analysis scripts
python src/ensemble_adjacency_matrices.py slurm_ensemble
python src/ensemble_test_evaluation.py slurm_ensemble
```

## Best Practices

### **1. Resource Planning**
- Start with small ensembles (5-10 runs) to test configuration
- Monitor resource usage and adjust SLURM parameters
- Use appropriate partition for your workload

### **2. Data Management**
- Ensure data files are accessible from compute nodes
- Use absolute paths in configuration files
- Consider using local scratch directories for I/O performance

### **3. Monitoring**
- Set up email notifications for job completion
- Monitor log files regularly during execution
- Check SLURM queue status periodically

### **4. Error Handling**
- Always check error logs for failed jobs
- Implement retry mechanisms for transient failures
- Keep backup of successful runs

## Troubleshooting

### **SLURM Commands Not Found**
```bash
# Add SLURM to PATH
export PATH=/usr/local/slurm/bin:$PATH

# Or load SLURM module
module load slurm
```

### **Conda Environment Issues**
```bash
# Verify conda installation
which conda

# Check environment
conda env list

# Recreate environment if needed
conda env create -f environment.yml
```

### **Permission Issues**
```bash
# Make scripts executable
chmod +x src/ensemble_dash_slurm.py

# Check file permissions
ls -la src/
```

## Support

For issues specific to SLURM ensemble training:

1. Check the log files in your output directory
2. Verify SLURM configuration matches your HPC system
3. Consult your HPC system documentation
4. Check SLURM queue status and resource availability

## Examples

### **Complete Workflow Example**
```bash
# 1. Setup environment
conda activate DASH_NYU

# 2. Submit ensemble
python src/ensemble_dash_slurm.py \
  --n_runs 25 \
  --output_dir my_ensemble \
  --slurm_config my_slurm_config.json

# 3. Monitor progress
tail -f my_ensemble/ensemble_log.txt

# 4. Check final status
python src/ensemble_dash_slurm.py --check-status my_ensemble

# 5. Analyze results
python src/ensemble_adjacency_matrices.py my_ensemble
```
