#!/usr/bin/env python3
"""
Sample usage script for DASH/BASE training and ensemble pipeline.

This script demonstrates how to use the DASH_NYU package for:
1. DASH training with prior knowledge
2. BASE training without prior knowledge
3. Ensemble pipeline execution
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Error!")
        print("Error:", e.stderr)
        return False

def main():
    """Main function demonstrating DASH/BASE usage."""
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"
    data_dir = project_root / "data" / "sample_data"
    
    print("🚀 DASH_NYU Sample Usage")
    print("="*60)
    
    # Check if required files exist
    required_files = [
        src_dir / "train_DASH_simu.py",
        src_dir / "ensemble_dash.py",
        data_dir / "simu_350genes_150samples_train_val.csv",
        data_dir / "edge_prior_matrix_simu_350_noise_0.05.csv"
    ]
    
    for file_path in required_files:
        if not file_path.exists():
            print(f"❌ Missing required file: {file_path}")
            return False
        else:
            print(f"✅ Found: {file_path}")
    
    print("\n📁 Project structure verified!")
    
    # Example 1: DASH Training
    print("\n🎯 Example 1: DASH Training with Prior Knowledge")
    dash_cmd = [
        sys.executable, str(src_dir / "train_DASH_simu.py"),
        "--settings", str(src_dir / "config_simu350_dash.cfg"),
        "--data", str(data_dir / "simu_350genes_150samples_train_val.csv"),
        "--prior", str(data_dir / "edge_prior_matrix_simu_350_noise_0.05.csv")
    ]
    
    success = run_command(dash_cmd, "DASH Training")
    if not success:
        print("⚠️  DASH training failed, but this is expected if dependencies aren't installed")
    
    # Example 2: BASE Training
    print("\n🎯 Example 2: BASE Training without Prior Knowledge")
    base_cmd = [
        sys.executable, str(src_dir / "train_DASH_simu.py"),
        "--settings", str(src_dir / "config_simu350_base.cfg"),
        "--data", str(data_dir / "simu_350genes_150samples_train_val.csv"),
        "--prior", str(data_dir / "edge_prior_matrix_simu_350_noise_0.05.csv")
    ]
    
    success = run_command(base_cmd, "BASE Training")
    if not success:
        print("⚠️  BASE training failed, but this is expected if dependencies aren't installed")
    
    # Example 3: Ensemble Pipeline
    print("\n🎯 Example 3: Ensemble Pipeline")
    ensemble_cmd = [
        sys.executable, str(src_dir / "ensemble_dash.py"),
        "--prior_matrix", str(data_dir / "edge_prior_matrix_simu_350_noise_0.05.csv"),
        "--rna_data", str(data_dir / "simu_350genes_150samples_train_val.csv"),
        "--config_file", str(src_dir / "config_simu350_dash.cfg"),
        "--n_runs", "2",  # Small number for demo
        "--output_dir", str(project_root / "results" / "demo_ensemble")
    ]
    
    success = run_command(ensemble_cmd, "Ensemble Pipeline")
    if not success:
        print("⚠️  Ensemble pipeline failed, but this is expected if dependencies aren't installed")
    
    # Print usage instructions
    print("\n📖 Usage Instructions")
    print("="*60)
    print("""
To run the examples successfully:

1. Install dependencies:
   pip install -r src/requirements.txt

2. Run DASH training:
   python src/train_DASH_simu.py \
     --settings src/config_simu350_dash.cfg \
     --data data/sample_data/simu_350genes_150samples_train_val.csv \
     --prior data/sample_data/edge_prior_matrix_simu_350_noise_0.05.csv

3. Run BASE training:
   python src/train_DASH_simu.py \
     --settings src/config_simu350_base.cfg \
     --data data/sample_data/simu_350genes_150samples_train_val.csv \
     --prior data/sample_data/edge_prior_matrix_simu_350_noise_0.05.csv

4. Run ensemble pipeline:
   python src/ensemble_dash.py --n_runs 10

For more examples, see the examples/ directory and README.md
""")
    
    print("✅ Sample usage demonstration completed!")
    return True

if __name__ == "__main__":
    main()
