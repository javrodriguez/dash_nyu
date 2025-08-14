# Examples

This directory contains example configurations and usage patterns for the DASH/BASE training and ensemble pipeline.

## Quick Examples

### 1. Basic DASH Training

```bash
# Train DASH model with default settings
python ../src/train_DASH_simu.py \
  --settings config_simu350_dash.cfg \
  --data ../data/sample_data/simu_350genes_150samples_train_val.csv \
  --prior ../data/sample_data/edge_prior_matrix_simu_350_noise_0.05.csv
```

### 2. Basic BASE Training

```bash
# Train BASE model without pruning
python ../src/train_DASH_simu.py \
  --settings config_simu350_base.cfg \
  --data ../data/sample_data/simu_350genes_150samples_train_val.csv \
  --prior ../data/sample_data/edge_prior_matrix_simu_350_noise_0.05.csv
```

### 3. Ensemble Training

```bash
# Run ensemble with 10 runs
python ../src/ensemble_dash.py \
  --n_runs 10 \
  --config_file ../src/config_simu350_dash.cfg
```

## Configuration Examples

### DASH Configuration (config_simu350_dash.cfg)
```ini
[Training]
epochs = 200
neurons_per_layer = 40
batch_type = trajectory
method = dopri5
pruning = True
learning_rate = 0.001
weight_decay = 0.0001

[Data]
noise = 0.05
prior_weight = 0.1
train_split = 0.8

[Pruning]
initial_hit_percentage = 70
pruning_percentage = 10
pruning_frequency = 10
```

### BASE Configuration (config_simu350_base.cfg)
```ini
[Training]
epochs = 200
neurons_per_layer = 40
batch_type = trajectory
method = dopri5
pruning = False
learning_rate = 0.001
weight_decay = 0.0001

[Data]
noise = 0.05
train_split = 0.8
```

### Ensemble Configuration
```bash
# Custom ensemble parameters
python ../src/ensemble_dash.py \
  --prior_matrix ../data/sample_data/edge_prior_matrix_simu_350_noise_0.05.csv \
  --rna_data ../data/sample_data/simu_350genes_150samples_train_val.csv \
  --n_genes 350 \
  --n_runs 10 \
  --output_dir ../results/my_ensemble \
  --epochs 200 \
  --neurons_per_layer 40
```

## Data Format Examples

### RNA Data Format (CSV)
```csv
gene_1,gene_2,gene_3,...,gene_n
1.234,0.567,2.345,...,1.123
0.987,1.456,0.789,...,2.456
...
```

### Prior Matrix Format (CSV)
```csv
gene_1,gene_2,gene_3,...,gene_n
0.0,0.1,0.0,...,0.2
0.1,0.0,0.3,...,0.0
...
```

## Output Examples

### Training Output Structure
```
output/
├── 2025-01-15(14;30)_simu_350genes_150samples_train_val_200epochs/
│   ├── best_val_model_alpha_comb_prods.pt
│   ├── best_val_model_alpha_comb_sums.pt
│   ├── best_val_model_gene_multipliers.pt
│   ├── settings.csv
│   ├── network.txt
│   ├── rep_epoch_losses.csv
│   └── img/
│       ├── MSE_loss.png
│       └── viz_genes_epoch0.png
```

### Ensemble Output Structure
```
ensemble_output/
├── ensemble_model/
│   ├── ensemble_info.json
│   └── ensemble_model_*.pt
├── extracted_grn_matrices/
│   ├── grn_binary_run_1_seed_42.csv
│   ├── grn_binary_run_2_seed_520128.csv
│   └── ...
├── ensemble_statistics/
│   ├── ensemble_mean_grn.csv
│   ├── ensemble_median_grn.csv
│   └── ensemble_std_grn.csv
└── ensemble_statistics.csv
```

## Common Use Cases

### 1. Compare DASH vs BASE Performance
```bash
# Train DASH model
python ../src/train_DASH_simu.py \
  --settings config_simu350_dash.cfg \
  --data ../data/sample_data/simu_350genes_150samples_train_val.csv \
  --prior ../data/sample_data/edge_prior_matrix_simu_350_noise_0.05.csv

# Train BASE model
python ../src/train_DASH_simu.py \
  --settings config_simu350_base.cfg \
  --data ../data/sample_data/simu_350genes_150samples_train_val.csv \
  --prior ../data/sample_data/edge_prior_matrix_simu_350_noise_0.05.csv

# Compare results
python ../src/compare_training_performance_ensemble.py
```

### 2. Robust Ensemble Inference
```bash
# Run large ensemble for robust results
python ../src/ensemble_dash.py \
  --n_runs 50 \
  --output_dir ../results/robust_ensemble
```

### 3. Custom Dataset Training
```bash
# Train on custom dataset
python ../src/train_DASH_simu.py \
  --settings config_custom.cfg \
  --data_path ../data/custom_dataset.csv \
  --prior_path ../data/custom_prior.csv
```

## Environment Setup

Before running the examples, make sure you have the DASH_NYU environment activated:

```bash
# Activate the conda environment
conda activate DASH_NYU

# Verify installation
python -c "import torch; import torchdiffeq; import numpy as np; import matplotlib.pyplot as plt; import tqdm; import pandas as pd; print('✅ Environment ready!')"
```

## Troubleshooting

### Common Issues

1. **Environment Issues**: Make sure the DASH_NYU conda environment is activated
2. **CUDA Out of Memory**: Reduce `neurons_per_layer` or batch size
3. **Training Divergence**: Reduce learning rate or increase weight decay
4. **Poor Convergence**: Check data preprocessing and prior matrix format
5. **Ensemble Failures**: Ensure consistent random seeds and data splits

### Performance Tips

1. **GPU Usage**: Ensure CUDA is available for faster training
2. **Memory Management**: Monitor GPU memory usage during training
3. **Data Preprocessing**: Normalize data appropriately for better convergence
4. **Hyperparameter Tuning**: Experiment with learning rates and pruning parameters
