# Sample Data

This directory contains sample datasets for testing and demonstrating the DASH/BASE training pipeline.

## Files

### `sample_data/simu_350genes_150samples_train_val.csv`
- **Description**: Full simulation dataset with 350 genes and 150 samples
- **Format**: CSV with genes as columns and samples as rows
- **Size**: ~4.5MB
- **Usage**: Main training dataset for DASH and BASE models

### `sample_data/simu_350genes_10samples_for_testing.csv`
- **Description**: Small test dataset with 350 genes and 10 samples
- **Format**: CSV with genes as columns and samples as rows
- **Size**: ~300KB
- **Usage**: Quick testing and validation

### `sample_data/edge_prior_matrix_simu_350_noise_0.05.csv`
- **Description**: Prior knowledge matrix for 350 genes with 5% noise
- **Format**: CSV matrix with gene-gene interaction scores
- **Size**: ~240KB
- **Usage**: Domain knowledge integration for DASH training

### `sample_data/gene_names.csv`
- **Description**: Gene names and identifiers
- **Format**: CSV with gene information
- **Size**: ~6KB
- **Usage**: Gene annotation and visualization

## Data Format

### RNA Expression Data
```csv
gene_1,gene_2,gene_3,...,gene_n
1.234,0.567,2.345,...,1.123
0.987,1.456,0.789,...,2.456
...
```

### Prior Matrix
```csv
gene_1,gene_2,gene_3,...,gene_n
0.0,0.1,0.0,...,0.2
0.1,0.0,0.3,...,0.0
...
```

## Usage Examples

### Training with Sample Data
```bash
# DASH training with prior knowledge
python src/train_DASH_simu.py \
  --settings src/config_simu350_dash.cfg \
  --data_path data/sample_data/simu_350genes_150samples_train_val.csv \
  --prior_path data/sample_data/edge_prior_matrix_simu_350_noise_0.05.csv

# BASE training without prior knowledge
python src/train_DASH_simu.py \
  --settings src/config_simu350_base.cfg \
  --data_path data/sample_data/simu_350genes_150samples_train_val.csv
```

### Ensemble Training
```bash
python src/ensemble_dash.py \
  --prior_matrix data/sample_data/edge_prior_matrix_simu_350_noise_0.05.csv \
  --rna_data data/sample_data/simu_350genes_150samples_train_val.csv \
  --n_genes 350 \
  --n_runs 10
```

## Data Sources

These datasets are derived from the DASH original paper simulations and are used for:
- **Reproducibility**: Ensuring consistent results across different environments
- **Testing**: Validating the implementation against known benchmarks
- **Demonstration**: Providing working examples for users
- **Development**: Testing new features and improvements

## Notes

- The data is preprocessed and ready for immediate use
- All datasets use the same gene set for consistency
- The prior matrix contains domain knowledge scores between 0 and 1
- RNA expression data is normalized and log-transformed
