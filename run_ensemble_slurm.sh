#!/bin/bash
# Wrapper script to run SLURM ensemble with proper conda environment

# Activate conda environment
source /gpfs/home/rodrij92/home_abl/miniconda3/etc/profile.d/conda.sh
conda activate DASH_NYU

# Check if conda environment is activated
if [ "$CONDA_DEFAULT_ENV" != "DASH_NYU" ]; then
    echo "Error: Failed to activate DASH_NYU conda environment"
    echo "Current environment: $CONDA_DEFAULT_ENV"
    exit 1
fi

echo "✅ Conda environment activated: $CONDA_DEFAULT_ENV"
echo "📦 Python path: $(which python)"
echo "📦 Pandas available: $(python -c 'import pandas; print("Yes")' 2>/dev/null || echo "No")"

# Run the ensemble script with all arguments passed through
python src/ensemble_dash_slurm.py "$@"
