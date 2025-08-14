# Conditional Neural ODE Training (Pruning Optional)

This directory contains a modified version of the DASH training code that allows you to choose whether to use pruning and prior knowledge or not through a configuration parameter.

## Key Features

### Conditional Pruning:
- **`pruning = False`**: Regular neural ODE training without pruning or prior knowledge
- **`pruning = True`**: Full DASH training with pruning and prior knowledge

### Configuration Control:
- All pruning functionality is controlled by the `pruning` parameter in `config_simu.cfg`
- When `pruning = False`, the model trains as a regular neural ODE with only data loss
- When `pruning = True`, the model uses full DASH functionality with prior knowledge and iterative pruning

## Usage

### For Regular Neural ODE Training (No Pruning):
```bash
# Edit config_simu.cfg and set:
pruning = False

# Then run:
python train_DASH_simu.py --settings config_simu.cfg
```

### For DASH Training (With Pruning):
```bash
# Edit config_simu.cfg and set:
pruning = True

# Then run:
python train_DASH_simu.py --settings config_simu.cfg
```

## Configuration Parameters

### Core Parameters (Always Used):
- `noise = 0.05`: Noise level in the data
- `epochs = 200`: Number of training epochs
- `neurons_per_layer = 40`: Hidden layer size
- `batch_type = trajectory`: Training on full trajectories
- `method = dopri5`: ODE solver method

### Pruning Parameters (Only Used When `pruning = True`):
- `pruning = True/False`: Enable/disable pruning and prior knowledge
- `pretrained_model = False`: Load pretrained model for post-hoc pruning

## What Changes Based on Pruning Setting

### When `pruning = False`:
- **Loss Function**: Only data loss (MSE between predictions and targets)
- **Optimizer**: Standard Adam optimizer
- **Training**: Clean training loop without pruning logic
- **Output**: Regular neural ODE model

### When `pruning = True`:
- **Loss Function**: Combined data loss + prior loss
- **Optimizer**: Custom Adam with different learning rates for different parameter groups
- **Training**: Full DASH training with iterative pruning
- **Prior Knowledge**: Loads prior matrix and computes prior loss
- **Pruning**: Iterative pruning with PPI and motif-based scores
- **Output**: Sparse neural ODE model with domain knowledge

## Output

The training will generate:
- `final_model_*.pt`: Final trained model weights
- `best_val_model_*.pt`: Best validation model weights  
- `settings.csv`: Training configuration
- `network.txt`: Model architecture details
- `rep_epoch_losses.csv`: Representative epoch losses
- `epoch_times.csv`: Training time per epoch
- `img/MSE_loss.png`: Training and validation loss plots (prior loss included if pruning enabled)

## Model Architecture

The model uses the same PHOENIX neural ODE architecture in both modes:
- `net_prods`: Linear layer with LogShiftedSoftSignMod activation
- `net_sums`: Linear layer with SoftsignMod activation
- `net_alpha_combine_sums`: Linear layer (no bias)
- `net_alpha_combine_prods`: Linear layer (no bias)
- `gene_multipliers`: Learnable gene-specific multipliers

## Purpose

This conditional version allows you to:
1. **Compare approaches**: Train the same model with and without pruning
2. **Control experiments**: Use the same codebase for both regular and DASH training
3. **Flexible experimentation**: Easily switch between modes by changing one config parameter 