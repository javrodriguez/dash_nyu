# DASH/BASE Training with Phoenix and Ensemble Pipeline

This repository contains implementations of Domain-Aware Sparse Hypernetworks (DASH) and BASE training approaches using the Phoenix neural ODE architecture, along with an ensemble pipeline for robust gene regulatory network inference.

## Overview

This project implements two main training approaches:

1. **DASH Training**: Domain-Aware Sparse Hypernetworks with prior knowledge integration and iterative pruning
2. **BASE Training**: Standard neural ODE training without domain knowledge or pruning
3. **Ensemble Pipeline**: Robust ensemble approach combining multiple DASH model runs

## Architecture

The core architecture is based on **Phoenix**, a neural ODE framework designed for gene regulatory network inference:

- **Neural ODE**: Continuous-time dynamics modeling
- **Domain Knowledge Integration**: Prior matrix incorporation for biological constraints
- **Iterative Pruning**: Sparse network learning with PPI and motif-based pruning
- **Ensemble Methods**: Robust inference through multiple model runs

## Repository Structure

```
├── src/                          # Enhanced DASH/BASE implementation
│   ├── train_DASH_simu.py       # Conditional DASH/BASE training
│   ├── ensemble_dash.py         # Ensemble pipeline
│   ├── ensemble_adjacency_matrices.py  # GRN matrix extraction
│   ├── ensemble_test_evaluation.py     # Ensemble evaluation
│   ├── config_simu350_dash.cfg  # DASH configuration
│   ├── config_simu350_base.cfg  # BASE configuration
│   └── requirements.txt         # Dependencies
├── data/                        # Sample data directory
│   └── sample_data/             # Example datasets
├── examples/                    # Example configurations and usage
└── tests/                       # Test suite
```

## Quick Start

### 1. Installation

#### Option A: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/javrodriguez/dash_nyu.git
cd dash_nyu

# Create and activate conda environment
conda env create -f environment.yml
conda activate DASH_NYU
```

#### Option B: Using pip

```bash
# Clone the repository
git clone https://github.com/javrodriguez/dash_nyu.git
cd dash_nyu

# Install dependencies
pip install -r src/requirements.txt
```

### 2. DASH Training

```bash
# Train with domain knowledge and pruning
python src/train_DASH_simu.py \
  --settings src/config_simu350_dash.cfg \
  --data data/sample_data/simu_350genes_150samples_train_val.csv \
  --prior data/sample_data/edge_prior_matrix_simu_350_noise_0.05.csv
```

### 3. BASE Training

```bash
# Train without domain knowledge or pruning
python src/train_DASH_simu.py \
  --settings src/config_simu350_base.cfg \
  --data data/sample_data/simu_350genes_150samples_train_val.csv \
  --prior data/sample_data/edge_prior_matrix_simu_350_noise_0.05.csv
```

### 4. Ensemble Pipeline

```bash
# Run ensemble with default parameters
python src/ensemble_dash.py

# Run with custom parameters
python src/ensemble_dash.py \
  --prior_matrix data/sample_data/edge_prior_matrix_simu_350_noise_0.05.csv \
  --rna_data data/sample_data/simu_350genes_150samples_train_val.csv \
  --config_file src/config_simu350_dash.cfg \
  --n_genes 350 \
  --n_runs 10 \
  --output_dir results/ensemble_run
```

## Key Features

### DASH Training
- **Prior Knowledge Integration**: Incorporates biological domain knowledge through prior matrices
- **Iterative Pruning**: Sparse network learning with PPI and motif-based pruning scores
- **Adaptive Learning**: Different learning rates for different parameter groups
- **Validation-based Model Selection**: Saves best model based on validation performance

### BASE Training
- **Standard Neural ODE**: Clean implementation without domain knowledge
- **Data-driven Learning**: Pure data-based training approach
- **Baseline Comparison**: Provides baseline for DASH performance evaluation
- **Note**: BASE training still requires a prior file argument, but ignores the prior knowledge during training

### Ensemble Pipeline
- **Multiple Runs**: Combines results from multiple training runs
- **Robust Statistics**: Mean, median, and standard deviation of GRN matrices
- **Comprehensive Evaluation**: Performance metrics across ensemble members
- **Visualization**: Heatmaps and statistical summaries

## Configuration

### DASH Configuration (`config_simu350_dash.cfg`)
```ini
[Training]
epochs = 200
neurons_per_layer = 40
batch_type = trajectory
method = dopri5
pruning = True

[Data]
noise = 0.05
prior_weight = 0.1

[Pruning]
initial_hit_percentage = 70
pruning_percentage = 10
pruning_frequency = 10
```

### BASE Configuration (`config_simu350_base.cfg`)
```ini
[Training]
epochs = 200
neurons_per_layer = 40
batch_type = trajectory
method = dopri5
pruning = False

[Data]
noise = 0.05
```

## Output Structure

### Individual Training
```
output/
├── run_timestamp/
│   ├── best_val_model_*.pt      # Best validation model
│   ├── final_model_*.pt         # Final trained model
│   ├── settings.csv             # Training configuration
│   ├── network.txt              # Model architecture
│   ├── rep_epoch_losses.csv     # Training losses
│   └── img/
│       ├── MSE_loss.png         # Loss curves
│       └── viz_genes_epoch*.png # Gene visualizations
```

### Ensemble Pipeline
```
ensemble_output/
├── ensemble_model/              # Combined ensemble model
├── extracted_grn_matrices/      # Individual GRN matrices
├── ensemble_statistics/         # Statistical summaries
├── heatmaps/                    # Visualization plots
└── ensemble_statistics.csv      # Performance metrics
```

## Model Architecture

The Phoenix neural ODE architecture consists of:

1. **Product Network** (`net_prods`): Linear layer with LogShiftedSoftSignMod activation
2. **Sum Network** (`net_sums`): Linear layer with SoftsignMod activation  
3. **Alpha Combination Networks**: Linear layers for parameter combination
4. **Gene Multipliers**: Learnable gene-specific scaling factors

## Environment Setup

### Conda Environment

The project includes a `environment.yml` file for easy environment setup:

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate DASH_NYU

# Deactivate environment
conda deactivate

# Verify installation
python -c "import torch; import torchdiffeq; import numpy as np; import matplotlib.pyplot as plt; import tqdm; import pandas as pd; print('✅ All dependencies imported successfully!')"
```

### Manual Environment Setup

If you prefer to create the environment manually:

```bash
# Create new conda environment
conda create -n DASH_NYU python=3.10

# Activate environment
conda activate DASH_NYU

# Install dependencies
pip install torch torchdiffeq numpy matplotlib tqdm pandas configparser adjustText

# Verify installation
python -c "import torch; import torchdiffeq; import numpy as np; import matplotlib.pyplot as plt; import tqdm; import pandas as pd; print('✅ All dependencies imported successfully!')"
```

## Dependencies

- **PyTorch** >= 1.9.0: Deep learning framework
- **torchdiffeq** >= 0.2.0: Neural ODE solvers
- **NumPy** >= 1.19.0: Numerical computing
- **Matplotlib** >= 3.3.0: Plotting and visualization
- **tqdm** >= 4.60.0: Progress bars
- **pandas** >= 1.3.0: Data manipulation and analysis
- **configparser**: Configuration file parsing
- **adjustText**: Text adjustment for plots

## Citation

If you use this code in your research, please cite:

```bibtex
@article{dash_phoenix_2024,
  title={Domain-Aware Sparse Hypernetworks for Gene Regulatory Network Inference},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Troubleshooting

### Common Issues

1. **Missing Arguments**: Both DASH and BASE training require `--data` and `--prior` arguments
2. **Environment Issues**: Make sure the DASH_NYU conda environment is activated
3. **CUDA Out of Memory**: Reduce `neurons_per_layer` or batch size
4. **Training Divergence**: Reduce learning rate or increase weight decay
5. **Poor Convergence**: Check data preprocessing and prior matrix format

### Performance Tips

1. **GPU Usage**: Ensure CUDA is available for faster training
2. **Memory Management**: Monitor GPU memory usage during training
3. **Data Preprocessing**: Normalize data appropriately for better convergence
4. **Hyperparameter Tuning**: Experiment with learning rates and pruning parameters

## Contact

For questions and support, please open an issue on GitHub or contact [your-email@domain.com].
