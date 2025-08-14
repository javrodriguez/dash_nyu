import torch
import numpy as np
import pandas as pd
import os
import glob
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def extract_seed_from_config(ensemble_dir, run_num):
    """Extract seed number from config file for a given run."""
    config_dir = os.path.join(ensemble_dir, 'configs')
    config_pattern = f"run_{run_num}_seed_*.cfg"
    config_files = glob.glob(os.path.join(config_dir, config_pattern))
    
    if config_files:
        # Extract seed from filename (e.g., "run_1_seed_42.cfg" -> 42)
        config_filename = os.path.basename(config_files[0])
        seed_str = config_filename.split('_seed_')[1].split('.')[0]
        return int(seed_str)
    else:
        # Try to read from config file content
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
    """Find all completed ensemble runs and their model files."""
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
        
        # Check for required model files
        required_files = [
            'best_val_model_sums.pt',
            'best_val_model_prods.pt', 
            'best_val_model_alpha_comb_sums.pt',
            'best_val_model_alpha_comb_prods.pt',
            'best_val_model_gene_multipliers.pt'
        ]
        
        all_files_exist = all(
            os.path.exists(os.path.join(timestamp_dir, f)) 
            for f in required_files
        )
        
        if all_files_exist:
            # Extract seed number
            seed = extract_seed_from_config(ensemble_dir, run_num)
            
            runs.append({
                'run_num': int(run_num),
                'run_name': run_name,
                'timestamp_dir': timestamp_dir,
                'seed': seed
            })
    
    return runs

def extract_grn_from_model(model_dir, run_num, seed):
    """
    Extract GRN adjacency matrix from a trained model.
    
    Args:
        model_dir: Directory containing model files
        run_num: Run number
        seed: Seed number used for this run
    
    Returns:
        adjacency_matrix: n×n numpy array representing the GRN
        sparsity: Sparsity percentage of the matrix
    """
    print(f"  Extracting GRN from Run {run_num} (seed: {seed})...")
    
    # Load model components
    try:
        # Load neural network components with weights_only=False for compatibility
        sums_model = torch.load(os.path.join(model_dir, 'best_val_model_sums.pt'), weights_only=False)
        prods_model = torch.load(os.path.join(model_dir, 'best_val_model_prods.pt'), weights_only=False)
        
        # Load combination weight matrices
        alpha_comb_sums = torch.load(os.path.join(model_dir, 'best_val_model_alpha_comb_sums.pt'), weights_only=False)
        alpha_comb_prods = torch.load(os.path.join(model_dir, 'best_val_model_alpha_comb_prods.pt'), weights_only=False)
        
        # Load gene multipliers
        gene_mult = torch.load(os.path.join(model_dir, 'best_val_model_gene_multipliers.pt'), weights_only=False)
        
    except Exception as e:
        print(f"    Error loading model files: {e}")
        return None, None
    
    with torch.no_grad():
        # Extract weight matrices
        # For sums network
        if hasattr(sums_model, 'linear_out'):
            Wo_sums = np.transpose(sums_model.linear_out.weight.detach().numpy())
        else:
            # Alternative structure
            Wo_sums = np.transpose(sums_model[1].odefunc.output_sums[1].sample_weights().detach().numpy())
        
        # For prods network  
        if hasattr(prods_model, 'linear_out'):
            Wo_prods = np.transpose(prods_model.linear_out.weight.detach().numpy())
        else:
            # Alternative structure
            Wo_prods = np.transpose(prods_model[1].odefunc.output_prods[1].sample_weights().detach().numpy())
        
        # Extract alpha combination matrices
        if hasattr(alpha_comb_sums, 'linear_out'):
            alpha_comb_sums_weights = np.transpose(alpha_comb_sums.linear_out.weight.detach().numpy())
        else:
            alpha_comb_sums_weights = np.transpose(alpha_comb_sums[1].odefunc.output_sums[2].sample_weights().detach().numpy())
        
        if hasattr(alpha_comb_prods, 'linear_out'):
            alpha_comb_prods_weights = np.transpose(alpha_comb_prods.linear_out.weight.detach().numpy())
        else:
            alpha_comb_prods_weights = np.transpose(alpha_comb_prods[1].odefunc.output_prods[3].sample_weights().detach().numpy())
        
        # Extract gene multipliers
        if hasattr(gene_mult, 'detach'):
            gene_mult_np = np.transpose(torch.relu(gene_mult.detach()).numpy())
        else:
            gene_mult_np = np.transpose(torch.relu(gene_mult).numpy())
        
        # Compute effects matrix
        effects_mat = np.matmul(Wo_sums, alpha_comb_sums_weights) + np.matmul(Wo_prods, alpha_comb_prods_weights)
        
        # Apply gene multipliers to create final adjacency matrix
        adjacency_matrix = effects_mat * np.transpose(gene_mult_np)
        
        # Calculate sparsity
        sparsity = np.sum(adjacency_matrix == 0) / adjacency_matrix.size
        
        print(f"    Matrix shape: {adjacency_matrix.shape}")
        print(f"    Sparsity: {sparsity:.2%}")
        print(f"    Non-zero elements: {np.count_nonzero(adjacency_matrix)}")
        
        return adjacency_matrix, sparsity

def create_binary_matrix(adjacency_matrix, threshold_method='quantile', threshold_value=0.995):
    """
    Create binary version of adjacency matrix using specified threshold.
    Preserves sign information: 0 (no connection), 1 (activation), -1 (repression)
    
    Args:
        adjacency_matrix: Raw adjacency matrix
        threshold_method: 'quantile', 'absolute', or 'std'
        threshold_value: Threshold value (quantile, absolute value, or std multiplier)
    
    Returns:
        binary_matrix: 0/±1 matrix indicating presence/absence and type of regulatory relationships
    """
    if threshold_method == 'quantile':
        # Use quantile-based threshold (e.g., 99.5th percentile)
        threshold = np.quantile(np.abs(adjacency_matrix), threshold_value)
    elif threshold_method == 'absolute':
        # Use absolute threshold (e.g., 1e-5 as in bone marrow paper)
        threshold = threshold_value
    elif threshold_method == 'std':
        # Use standard deviation-based threshold
        threshold = threshold_value * np.std(adjacency_matrix)
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")
    
    # Create binary matrix preserving signs: ±1 if |value| > threshold, 0 otherwise
    binary_matrix = np.where(np.abs(adjacency_matrix) > threshold, 
                           np.sign(adjacency_matrix), 0)
    
    return binary_matrix, threshold

def save_grn_matrix(adjacency_matrix, run_num, seed, output_dir):
    """Save GRN adjacency matrix to file."""
    if adjacency_matrix is None:
        return
    
    # Save raw matrix as CSV
    csv_filename = f"grn_run_{run_num}_seed_{seed}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    pd.DataFrame(adjacency_matrix).to_csv(csv_path, index=False, header=False)
    print(f"    Saved GRN matrix to: {csv_path}")
    
    # Save raw matrix as numpy array
    np_filename = f"grn_run_{run_num}_seed_{seed}.npy"
    np_path = os.path.join(output_dir, np_filename)
    np.save(np_path, adjacency_matrix)
    print(f"    Saved GRN matrix to: {np_path}")
    
    # Create and save binary matrix
    binary_matrix, threshold = create_binary_matrix(adjacency_matrix)
    
    # Save binary matrix as CSV
    binary_csv_filename = f"grn_binary_run_{run_num}_seed_{seed}.csv"
    binary_csv_path = os.path.join(output_dir, binary_csv_filename)
    pd.DataFrame(binary_matrix).to_csv(binary_csv_path, index=False, header=False)
    print(f"    Saved binary GRN matrix (threshold: {threshold:.2e}) to: {binary_csv_path}")
    
    # Save binary matrix as numpy array
    binary_np_filename = f"grn_binary_run_{run_num}_seed_{seed}.npy"
    binary_np_path = os.path.join(output_dir, binary_np_filename)
    np.save(binary_np_path, binary_matrix)
    print(f"    Saved binary GRN matrix to: {binary_np_path}")
    
    return csv_path, np_path, binary_csv_path, binary_np_path, threshold

def compute_ensemble_statistics(grn_matrices, output_dir):
    """Compute ensemble statistics across all GRN matrices."""
    if not grn_matrices:
        return
    
    print("\nComputing ensemble statistics...")
    
    # Stack all matrices
    stacked_matrices = np.stack(list(grn_matrices.values()))
    
    # Compute statistics
    mean_grn = np.mean(stacked_matrices, axis=0)
    std_grn = np.std(stacked_matrices, axis=0)
    median_grn = np.median(stacked_matrices, axis=0)
    
    # Save ensemble statistics
    stats_dir = os.path.join(output_dir, 'ensemble_statistics')
    os.makedirs(stats_dir, exist_ok=True)
    
    # Save mean GRN
    mean_path = os.path.join(stats_dir, 'ensemble_mean_grn.csv')
    pd.DataFrame(mean_grn).to_csv(mean_path, index=False, header=False)
    print(f"  Saved ensemble mean GRN to: {mean_path}")
    
    # Save std GRN
    std_path = os.path.join(stats_dir, 'ensemble_std_grn.csv')
    pd.DataFrame(std_grn).to_csv(std_path, index=False, header=False)
    print(f"  Saved ensemble std GRN to: {std_path}")
    
    # Save median GRN
    median_path = os.path.join(stats_dir, 'ensemble_median_grn.csv')
    pd.DataFrame(median_grn).to_csv(median_path, index=False, header=False)
    print(f"  Saved ensemble median GRN to: {median_path}")
    
    # Compute consistency metrics
    print(f"  Ensemble consistency analysis:")
    print(f"    Mean sparsity: {np.mean([np.sum(m == 0) / m.size for m in grn_matrices.values()]):.2%}")
    print(f"    Mean non-zero elements: {np.mean([np.count_nonzero(m) for m in grn_matrices.values()]):.0f}")
    
    return mean_grn, std_grn, median_grn

def create_heatmaps(grn_matrices, binary_matrices, output_dir, prior_matrix_path=None):
    """
    Create heatmaps for all GRN matrices with consistent color scales.
    
    Args:
        grn_matrices: Dictionary of raw adjacency matrices
        binary_matrices: Dictionary of binary adjacency matrices
        output_dir: Output directory for heatmaps
        prior_matrix_path: Path to the ground truth prior matrix (optional)
    """
    print("\nCreating heatmaps...")
    
    # Create heatmaps directory
    heatmaps_dir = os.path.join(output_dir, 'heatmaps')
    os.makedirs(heatmaps_dir, exist_ok=True)
    
    # Load ground truth matrix if provided
    ground_truth_matrix = None
    if prior_matrix_path and os.path.exists(prior_matrix_path):
        try:
            ground_truth_matrix = np.loadtxt(prior_matrix_path, delimiter=',')
            print(f"  Loaded ground truth matrix from: {prior_matrix_path}")
            print(f"    Shape: {ground_truth_matrix.shape}")
            print(f"    Range: [{ground_truth_matrix.min()}, {ground_truth_matrix.max()}]")
        except Exception as e:
            print(f"  Warning: Could not load ground truth matrix: {e}")
            ground_truth_matrix = None
    
    # Use fixed color scale from -1 to 1 for raw matrices
    vmin_raw = -1.0
    vmax_raw = 1.0
    
    # For binary matrices, values are -1, 0, or 1
    vmin_binary = -1
    vmax_binary = 1
    
    # Create raw matrices heatmap
    print("  Creating raw matrices heatmap...")
    
    # Determine number of subplots needed
    num_matrices = len(grn_matrices)
    if ground_truth_matrix is not None:
        num_matrices += 1
    
    # Calculate grid dimensions
    if num_matrices <= 4:
        rows, cols = 1, 4
    elif num_matrices <= 8:
        rows, cols = 2, 4
    else:
        rows, cols = 3, 4
    
    fig_raw, axes_raw = plt.subplots(rows, cols, figsize=(20, 5*rows))
    axes_raw = axes_raw.flatten()
    
    plot_index = 0
    
    # Add ground truth matrix first if available
    if ground_truth_matrix is not None:
        ax = axes_raw[plot_index]
        # Rotate matrix 90 degrees to the right (transpose and flip)
        rotated_matrix = np.flipud(ground_truth_matrix.T)
        im = ax.imshow(rotated_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        ax.set_title('Ground Truth\nRange: [-1, 1]')
        ax.set_xlabel('Target Gene')
        ax.set_ylabel('Regulator Gene')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Regulatory Strength')
        plot_index += 1
    
    # Add ensemble matrices
    for run_name, matrix in grn_matrices.items():
        if plot_index < len(axes_raw):
            ax = axes_raw[plot_index]
            # Rotate matrix 90 degrees to the right (transpose and flip)
            rotated_matrix = np.flipud(matrix.T)
            im = ax.imshow(rotated_matrix, cmap='RdBu_r', vmin=vmin_raw, vmax=vmax_raw, aspect='equal')
            ax.set_title(f'{run_name}\nRange: [{matrix.min():.3f}, {matrix.max():.3f}]')
            ax.set_xlabel('Target Gene')
            ax.set_ylabel('Regulator Gene')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Regulatory Strength')
            plot_index += 1
    
    # Hide unused subplots
    for i in range(plot_index, len(axes_raw)):
        axes_raw[i].set_visible(False)
    
    plt.tight_layout()
    raw_heatmap_path = os.path.join(heatmaps_dir, 'all_raw_matrices_heatmap.png')
    plt.savefig(raw_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved raw matrices heatmap to: {raw_heatmap_path}")
    
    # Create binary matrices heatmap
    print("  Creating binary matrices heatmap...")
    
    # Determine number of subplots needed for binary matrices
    num_binary_matrices = len(binary_matrices)
    if ground_truth_matrix is not None:
        num_binary_matrices += 1
    
    # Calculate grid dimensions
    if num_binary_matrices <= 4:
        rows, cols = 1, 4
    elif num_binary_matrices <= 8:
        rows, cols = 2, 4
    else:
        rows, cols = 3, 4
    
    fig_binary, axes_binary = plt.subplots(rows, cols, figsize=(20, 5*rows))
    axes_binary = axes_binary.flatten()
    
    plot_index = 0
    
    # Add ground truth binary matrix first if available
    if ground_truth_matrix is not None:
        ax = axes_binary[plot_index]
        # Create binary version of ground truth (preserve signs)
        ground_truth_binary = np.sign(ground_truth_matrix)
        # Rotate matrix 90 degrees to the right (transpose and flip)
        rotated_matrix = np.flipud(ground_truth_binary.T)
        im = ax.imshow(rotated_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        ax.set_title(f'Ground Truth Binary\nEdges: {np.sum(ground_truth_binary != 0)}')
        ax.set_xlabel('Target Gene')
        ax.set_ylabel('Regulator Gene')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Edge Presence (0/1)')
        plot_index += 1
    
    # Add ensemble binary matrices
    for run_name, matrix in binary_matrices.items():
        if plot_index < len(axes_binary):
            ax = axes_binary[plot_index]
            # Rotate matrix 90 degrees to the right (transpose and flip)
            rotated_matrix = np.flipud(matrix.T)
            im = ax.imshow(rotated_matrix, cmap='RdBu_r', vmin=vmin_binary, vmax=vmax_binary, aspect='equal')
            ax.set_title(f'{run_name}\nEdges: {np.sum(matrix != 0)}')
            ax.set_xlabel('Target Gene')
            ax.set_ylabel('Regulator Gene')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Edge Presence (0/1)')
            plot_index += 1
    
    # Hide unused subplots
    for i in range(plot_index, len(axes_binary)):
        axes_binary[i].set_visible(False)
    
    plt.tight_layout()
    binary_heatmap_path = os.path.join(heatmaps_dir, 'all_binary_matrices_heatmap.png')
    plt.savefig(binary_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved binary matrices heatmap to: {binary_heatmap_path}")
    
    # Create individual heatmaps for better detail
    print("  Creating individual heatmaps...")
    individual_dir = os.path.join(heatmaps_dir, 'individual')
    os.makedirs(individual_dir, exist_ok=True)
    
    for run_name, matrix in grn_matrices.items():
        # Raw matrix heatmap
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        # Rotate matrix 90 degrees to the right (transpose and flip)
        rotated_matrix = np.flipud(matrix.T)
        im = ax.imshow(rotated_matrix, cmap='RdBu_r', vmin=vmin_raw, vmax=vmax_raw, aspect='equal')
        ax.set_title(f'{run_name} - Raw GRN Matrix\nRange: [{matrix.min():.3f}, {matrix.max():.3f}]')
        ax.set_xlabel('Target Gene')
        ax.set_ylabel('Regulator Gene')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Regulatory Strength')
        
        plt.tight_layout()
        individual_raw_path = os.path.join(individual_dir, f'{run_name}_raw_heatmap.png')
        plt.savefig(individual_raw_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    for run_name, matrix in binary_matrices.items():
        # Binary matrix heatmap
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        # Rotate matrix 90 degrees to the right (transpose and flip)
        rotated_matrix = np.flipud(matrix.T)
        im = ax.imshow(rotated_matrix, cmap='RdBu_r', vmin=vmin_binary, vmax=vmax_binary, aspect='equal')
        ax.set_title(f'{run_name} - Binary GRN Matrix\nEdges: {np.sum(matrix != 0)}')
        ax.set_xlabel('Target Gene')
        ax.set_ylabel('Regulator Gene')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Edge Presence (0/1)')
        
        plt.tight_layout()
        individual_binary_path = os.path.join(individual_dir, f'{run_name}_binary_heatmap.png')
        plt.savefig(individual_binary_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"    Saved individual heatmaps to: {individual_dir}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_grn_from_ensemble.py <ensemble_dir> [prior_matrix_path]")
        print("Example: python extract_grn_from_ensemble.py ensemble_350genes_150samples_50runs")
        print("Example with prior: python extract_grn_from_ensemble.py ensemble_350genes_150samples_50runs external_data/DASH_original/ground_truth_simulator/clean_data/edge_prior_matrix_simu_350_noise_0.05.csv")
        sys.exit(1)
    
    ensemble_dir = sys.argv[1]
    prior_matrix_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Extracting GRN matrices from ensemble: {ensemble_dir}")
    
    # Find all ensemble runs
    runs = find_ensemble_runs(ensemble_dir)
    
    if not runs:
        print(f"No completed runs with model files found in {ensemble_dir}")
        sys.exit(1)
    
    print(f"Found {len(runs)} runs with model files: {[r['run_name'] for r in runs]}")
    
    # Create output directory
    output_dir = os.path.join(ensemble_dir, 'extracted_grn_matrices')
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract GRN from each run
    grn_matrices = {}
    binary_matrices = {}
    extraction_summary = []
    
    for run in runs:
        adjacency_matrix, sparsity = extract_grn_from_model(
            run['timestamp_dir'], 
            run['run_num'], 
            run['seed']
        )
        
        if adjacency_matrix is not None:
            # Save individual GRN matrix and create binary version
            csv_path, np_path, binary_csv_path, binary_np_path, threshold = save_grn_matrix(
                adjacency_matrix, 
                run['run_num'], 
                run['seed'], 
                output_dir
            )
            
            # Create binary matrix for heatmaps
            binary_matrix, _ = create_binary_matrix(adjacency_matrix)
            
            # Store for ensemble analysis and heatmaps
            grn_matrices[f"run_{run['run_num']}"] = adjacency_matrix
            binary_matrices[f"run_{run['run_num']}"] = binary_matrix
            
            # Record summary
            extraction_summary.append({
                'Run_Number': run['run_num'],
                'Seed': run['seed'],
                'Matrix_Shape': adjacency_matrix.shape,
                'Sparsity': sparsity,
                'Non_Zero_Elements': np.count_nonzero(adjacency_matrix),
                'Binary_Threshold': threshold,
                'Binary_Edges': np.sum(binary_matrix != 0),
                'CSV_File': csv_path,
                'NPY_File': np_path,
                'Binary_CSV_File': binary_csv_path,
                'Binary_NPY_File': binary_np_path
            })
        else:
            print(f"  Failed to extract GRN from Run {run['run_num']}")
    
    # Save extraction summary
    if extraction_summary:
        summary_df = pd.DataFrame(extraction_summary)
        summary_path = os.path.join(output_dir, 'grn_extraction_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nExtraction summary saved to: {summary_path}")
        
        # Print summary
        print("\n=== GRN EXTRACTION SUMMARY ===")
        print(f"Successfully extracted GRN matrices from {len(extraction_summary)} runs")
        for summary in extraction_summary:
            print(f"  Run {summary['Run_Number']} (seed: {summary['Seed']}): "
                  f"Shape {summary['Matrix_Shape']}, "
                  f"Sparsity {summary['Sparsity']:.2%}, "
                  f"Non-zero: {summary['Non_Zero_Elements']}, "
                  f"Binary threshold: {summary['Binary_Threshold']:.2e}, "
                  f"Binary edges: {summary['Binary_Edges']}")
    
    # Create heatmaps
    if len(grn_matrices) > 0:
        create_heatmaps(grn_matrices, binary_matrices, output_dir, prior_matrix_path)
    
    # Compute ensemble statistics
    if len(grn_matrices) > 1:
        mean_grn, std_grn, median_grn = compute_ensemble_statistics(grn_matrices, output_dir)
        print(f"\nEnsemble statistics computed for {len(grn_matrices)} GRN matrices")
    else:
        print(f"\nSkipping ensemble statistics (only {len(grn_matrices)} matrix available)")
    
    print(f"\nAll GRN matrices saved to: {output_dir}")

if __name__ == "__main__":
    main()
