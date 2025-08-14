# Imports
import sys
import os
import argparse
import inspect
from datetime import datetime
import numpy as np
from tqdm import tqdm
from math import ceil
from time import perf_counter, process_time

import torch
import torch.optim as optim
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint

from datahandler import DataHandler
from PHX_base_model import ODENet, get_mask_smallest_p_proportion
from read_config import read_arguments_from_file
from visualization_inte import *

def plot_MSE(epoch_so_far, training_loss, validation_loss, true_mean_losses, true_mean_losses_init_val_based, prior_losses, img_save_dir):
    plt.figure()
    plt.plot(range(1, epoch_so_far + 1), training_loss, color = "blue", label = "Training loss")
    if len(validation_loss) > 0:
        plt.plot(range(1, epoch_so_far + 1), validation_loss, color = "red", label = "Validation loss")
    #plt.plot(range(1, epoch_so_far + 1), true_mean_losses, color = "green", label = r'True $\mu$ loss')
    if prior_losses:
        plt.plot(range(1, epoch_so_far + 1), prior_losses, color = "magenta", label = "Prior loss")
    
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.legend(loc='upper right')
    plt.ylabel("Error (MSE)")
    plt.savefig("{}/MSE_loss.png".format(img_save_dir))
    # Save all losses to CSV, including prior loss
    np.savetxt(f'{img_save_dir}/full_loss_info.csv', np.c_[training_loss, validation_loss, true_mean_losses, true_mean_losses_init_val_based, prior_losses], delimiter=',')
    plt.close()

def my_r_squared(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r_squared = 1 - ss_res / ss_tot
    return r_squared

def get_true_val_set_r2(odenet, data_handler, method, batch_type):
    data, t, target = data_handler.get_true_mu_set_pairwise(val_only = True, batch_type = batch_type)
    print(f"DEBUG: data shape: {data.shape}, t shape: {t.shape}, target shape: {target.shape}")
    
    predictions = torch.zeros(data.shape).to(data_handler.device)
    for index, (time, batch_point) in enumerate(zip(t, data)):
        predictions[index, :, :] = odeint(odenet, batch_point, time, method= method  )[1] + data_handler.init_bias_y
    
    # Select only the final time point to match target shape
    if predictions.shape[1] > 1:  # If we have multiple time points
        predictions = predictions[:, -1, :, :]  # Take the last time point
    
    print(f"DEBUG: predictions shape after fix: {predictions.shape}, target shape: {target.shape}")
    print(f"DEBUG: predictions device: {predictions.device}, target device: {target.device}")
    
    r2 = my_r_squared(predictions, target)
    mse = torch.mean((predictions - target)**2)
    return [r2.item(), mse.item()]

def random_multiply(mat_torch):
    rand_torch = torch.rand(mat_torch.size())
    out_torch = torch.zeros(mat_torch.size())

    for i in range(mat_torch.size(0)):
        for j in range(mat_torch.size(1)):
            if rand_torch[i, j] > 0.50:
                out_torch[i, j] = mat_torch[i, j] * -1 #flip
            else:
                out_torch[i, j] = mat_torch[i, j] * 1 #keep

    return out_torch

def read_prior_matrix(prior_mat_file_loc, sparse = False, num_genes = 11165, absolute = False):
    if sparse == False: 
        mat = np.genfromtxt(prior_mat_file_loc,delimiter=',')
        mat_torch = torch.from_numpy(mat)
        mat_torch = mat_torch.float()
        if absolute:
            print("I AM SWITCHING ALL EDGE SIGNS to POSITIVE!")
            mat_torch = torch.abs(mat_torch)
        return mat_torch
    else: #when scaling up >10000
        mat = np.genfromtxt(prior_mat_file_loc,delimiter=',')
        indices = torch.tensor([mat[:,0].astype(int)-1, mat[:,1].astype(int)-1])
        values = torch.tensor(mat[:,2])
        sparse_mat = torch.sparse_coo_tensor(indices, values, ( num_genes,  num_genes))
        mat_torch = sparse_mat.to_dense().float()
        return(mat_torch)

def normalize_values(my_tensor):
    """Normalizes the values of a PyTorch tensor.

    Args:
        tensor: The input tensor.

    Returns:
        A normalized tensor.
    """
    
    # Find the minimum and maximum absolute values in the tensor.
    min_val = torch.min(my_tensor)
    max_val = torch.max(my_tensor)

    # Normalize the absolute values to the range [0, 1].
    normalized_tensor = (my_tensor - min_val) / (max_val - min_val)

    return normalized_tensor

def validation(odenet, data_handler, method, explicit_time):
    data, t, target_full, n_val = data_handler.get_validation_set()
    if method == "trajectory":
        False

    init_bias_y = data_handler.init_bias_y
    #odenet.eval()
    with torch.no_grad():
        predictions = []
        targets = []
        # For now we have to loop through manually, their implementation of odenet can only take fixed time lists.
        for index, (time, batch_point, target_point) in enumerate(zip(t, data, target_full)):
            # Fix time dimensionality - extract first column if 2D
            if time.ndimension() > 1:
                time = time[:, 0]  # Take first column to make it 1D
            
            # Do prediction
            predictions.append(odeint(odenet, batch_point, time, method=method)[1])
            targets.append(target_point)

        # Calculate validation loss
        predictions = torch.cat(predictions, dim = 0).to(data_handler.device)
        targets = torch.cat(targets, dim = 0).to(data_handler.device)
        loss = torch.mean((predictions - targets)**2)
        
    return [loss, n_val]

def reset_lr(opt, verbose, old_lr):
    dir_string = "Increasing"
    group_count = 0
    for param_group in opt.param_groups:
        group_count += 1 #gene_mult!
        param_group['lr'] = old_lr 
        if group_count == 6:
            param_group['lr'] = 5*old_lr 
        if verbose:
            print(dir_string,"learning rate to: %f" % param_group['lr'])

def setOptimizerLRScheduler(opt, patience = 3):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=patience, threshold=1e-4)

def training_step(odenet, data_handler, opt, method, batch_size, batch_for_prior, prior_grad, lambda_loss):
    batch, t, target = data_handler.get_batch(batch_size)
    
    init_bias_y = data_handler.init_bias_y
    opt.zero_grad()
    predictions = torch.zeros(batch.shape).to(data_handler.device)
    for index, (time, batch_point) in enumerate(zip(t, batch)):
        predictions[index, :, :] = odeint(odenet, batch_point, time, method= method  )[1] + init_bias_y
    
    loss_data = torch.mean((predictions - target)**2) 
    
    if batch_for_prior is not None and prior_grad is not None:
        pred_grad = odenet.prior_only_forward(t, batch_for_prior)
        loss_prior = torch.mean((pred_grad - prior_grad)**2)
        composed_loss = lambda_loss*loss_data + (1- lambda_loss)*loss_prior
        composed_loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(odenet.parameters(), max_norm=1.0)
        opt.step()
        return [loss_data, loss_prior]
    else:
        loss_data.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(odenet.parameters(), max_norm=1.0)
        opt.step()
        return [loss_data, torch.tensor(0.0)]

def training_step_conditional(odenet, data_handler, opt, method, batch_size, batch_for_prior, prior_grad, lambda_loss, use_prior, use_prior_loss=False):
    if use_prior:
        return training_step(odenet, data_handler, opt, method, batch_size, batch_for_prior, prior_grad, lambda_loss)
    else:
        # Use the same function but ignore prior
        batch, t, target = data_handler.get_batch(batch_size)
        init_bias_y = data_handler.init_bias_y
        opt.zero_grad()
        predictions = torch.zeros(batch.shape).to(data_handler.device)
        for index, (time, batch_point) in enumerate(zip(t, batch)):
            predictions[index, :, :] = odeint(odenet, batch_point, time, method= method  )[1] + init_bias_y
        loss_data = torch.mean((predictions - target)**2) 
        
        # Compute prior loss for tracking even if not using it for training
        if use_prior_loss and batch_for_prior is not None and prior_grad is not None:
            with torch.no_grad():
                pred_grad = odenet.prior_only_forward(t, batch_for_prior)
                loss_prior = torch.mean((pred_grad - prior_grad)**2)
        else:
            loss_prior = torch.tensor(0.0)
            
        loss_data.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(odenet.parameters(), max_norm=1.0)
        opt.step()
        return [loss_data, loss_prior]

def _build_save_file_name(save_path, epochs):
    return '{}-{}-{}({};{})_{}_{}epochs'.format(str(datetime.now().year), str(datetime.now().month),
        str(datetime.now().day), str(datetime.now().hour), str(datetime.now().minute), save_path, epochs)

def save_model(odenet, folder, filename):
    odenet.save('{}{}.pt'.format(folder, filename))

def get_global_sparsity_metrics(odenet):
    all_weights = []
    for name, module in odenet.named_modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, 'weight') and module.weight is not None:
            # Get the actual masked weights (pruned weights will be 0)
            if hasattr(module, 'weight_mask') and module.weight_mask is not None:
                # Use masked weights if pruning mask exists
                masked_weights = module.weight.detach() * module.weight_mask.detach()
                all_weights.append(masked_weights.cpu().numpy().flatten())
            else:
                # Use regular weights if no mask
                all_weights.append(module.weight.detach().cpu().numpy().flatten())
    
    # Also include gene_multipliers if they exist
    if hasattr(odenet, 'gene_multipliers') and odenet.gene_multipliers is not None:
        all_weights.append(odenet.gene_multipliers.detach().cpu().numpy().flatten())
    
    all_weights = np.concatenate(all_weights)
    nonzero_weights = all_weights[all_weights != 0]
    total_nonzero = len(nonzero_weights)
    median_nonzero = np.median(nonzero_weights) if total_nonzero > 0 else 0.0
    min_nonzero = np.min(nonzero_weights) if total_nonzero > 0 else 0.0
    max_nonzero = np.max(nonzero_weights) if total_nonzero > 0 else 0.0
    proportion_nonzero = total_nonzero / all_weights.size
    l1_norm = np.sum(np.abs(nonzero_weights))
    return total_nonzero, median_nonzero, min_nonzero, max_nonzero, proportion_nonzero, l1_norm

def get_layerwise_nonzero_weights(odenet):
    layerwise = {}
    for name, module in odenet.named_modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, 'weight') and module.weight is not None:
            w = module.weight.detach().cpu().numpy().flatten()
            nonzero = w[w != 0]
            layerwise[name] = nonzero
    return layerwise

def save_layerwise_histograms(layerwise, epoch, when, hist_dir):
    os.makedirs(hist_dir, exist_ok=True)
    for lname, weights in layerwise.items():
        plt.figure()
        if len(weights) > 0:
            plt.hist(weights, bins=50, color='blue', alpha=0.7)
        plt.title(f"{lname} nonzero weights {when} pruning, epoch {epoch}")
        plt.xlabel("Weight value")
        plt.ylabel("Count")
        plt.tight_layout()
        fname = f"{hist_dir}/{lname.replace('.', '_')}_epoch{epoch}_{when}.png"
        plt.savefig(fname)
        plt.close()

parser = argparse.ArgumentParser('Flexible DASH Training')
parser.add_argument('--settings', type=str, default='config_simu.cfg')
parser.add_argument('--output_dir', type=str, default=None, help='Override output directory name')
parser.add_argument('--data', type=str, required=True, help='Path to data CSV')
parser.add_argument('--prior', type=str, required=True, help='Path to main prior matrix')
parser.add_argument('--secondary_prior', type=str, default=None, help='Path to secondary prior matrix (e.g., PPI)')
parser.add_argument('--use_secondary_prior', action='store_true', help='Whether to use a secondary prior matrix')
args = parser.parse_args()

# Main function
if __name__ == "__main__":
    print('Setting recursion limit to 3000')
    sys.setrecursionlimit(3000)
    print('Loading settings from file {}'.format(args.settings))
    settings = read_arguments_from_file(args.settings)
    print("[DEBUG] Loaded settings:", settings)
    print('[DEBUG] compute_prior_loss:', settings.get('compute_prior_loss', False))
    # Set random seed for reproducibility
    seed = settings.get('seed', 42)
    import torch, numpy as np, random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f'[INFO] Using random seed: {seed}')
    if settings.get('pruning', False):
        print("[INFO] Pruning is ENABLED. DASH mode active.")
    else:
        print("[INFO] Pruning is OFF. Running standard Neural ODE.")
    
    # Dynamically set cleaned_file_name based on data file
    import os
    data_filename = os.path.basename(args.data)  # Get filename from path
    cleaned_file_name = os.path.splitext(data_filename)[0]  # Remove extension
    print(f"[INFO] Using data file: {data_filename}")
    print(f"[INFO] Set cleaned_file_name to: {cleaned_file_name}")
    
    # Use output_dir from argument if provided, else from config
    output_dir = args.output_dir if args.output_dir is not None else settings.get('output_dir', 'output')
    save_file_name = _build_save_file_name(cleaned_file_name, settings.get('epochs', 200))
    output_root_dir = '{}/{}/'.format(output_dir, save_file_name)

    img_save_dir = '{}img/'.format(output_root_dir)
    interm_models_save_dir = '{}interm_models/'.format(output_root_dir)

    # Create image and model save directory
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir, exist_ok=True)
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    if not os.path.exists(interm_models_save_dir):
        os.mkdir(interm_models_save_dir)

    # Save the settings for future reference
    with open('{}/settings.csv'.format(output_root_dir), 'w') as f:
        f.write("Setting,Value\n")
        for key in settings.keys():
            f.write("{},{}\n".format(key,settings[key]))

    # Use GPU if available
    if not settings.get('cpu', False):
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        print("Trying to run on GPU -- cuda available: " + str(torch.cuda.is_available()))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Running on", device)
    else:
        print("Running on CPU")
        device = 'cpu'
    
    data_handler = DataHandler.fromcsv(args.data, device, settings.get('val_split', 0.2), normalize=settings.get('normalize_data', False), 
                                        batch_type=settings.get('batch_type', 'single'), batch_time=settings.get('batch_time', 1), 
                                        batch_time_frac=settings.get('batch_time_frac', 1.0),
                                        noise = settings.get('noise', 0.0),
                                        img_save_dir = img_save_dir,
                                        scale_expression = settings.get('scale_expression', False),
                                        log_scale = settings.get('log_scale', False),
                                        init_bias_y = settings.get('init_bias_y', 0.0))

    # Initialize prior knowledge variables
    batch_for_prior = None
    prior_grad = None
    PPI = None
    noisy_PPI = None
    noisy_prior_mat = None
    loss_lambda = 1.0
    my_current_custom_pruning_scores = {}

    # Initialize pruning scores for DASH pruning (always needed, even if pruning is disabled)
    my_current_custom_pruning_scores['net_prods.linear_out'] = torch.rand(settings.get('neurons_per_layer', 10), data_handler.dim, device=device)
    my_current_custom_pruning_scores['net_sums.linear_out'] = torch.rand(settings.get('neurons_per_layer', 10), data_handler.dim, device=device)
    my_current_custom_pruning_scores['net_alpha_combine_sums.linear_out'] = torch.rand(data_handler.dim, settings.get('neurons_per_layer', 10), device=device)
    my_current_custom_pruning_scores['net_alpha_combine_prods.linear_out'] = torch.rand(data_handler.dim, settings.get('neurons_per_layer', 10), device=device)

    # Load main prior matrix (always required)
    prior_mat = read_prior_matrix(args.prior, sparse=False, num_genes=data_handler.dim)
    prior_mat = prior_mat.to(device)  # Move to correct device
    noisy_prior_mat = prior_mat
    
    # Create PPI matrix from prior matrix (for DASH pruning)
    PPI = torch.matmul(torch.abs(prior_mat), torch.transpose(torch.abs(prior_mat), 0, 1))
    PPI = PPI / torch.sum(PPI)  # normalize PPI
    noisy_PPI = PPI
    
    # Create synthetic data for prior loss computation
    batch_for_prior = (torch.rand(10000, 1, prior_mat.shape[0], device=device) - 0.5)
    prior_grad = torch.matmul(batch_for_prior, prior_mat)  # can be any model here that predicts the derivative

    # Optional secondary prior (e.g., PPI)
    if args.use_secondary_prior and args.secondary_prior is not None:
        secondary_prior = read_prior_matrix(args.secondary_prior, sparse=False, num_genes=data_handler.dim)
        secondary_prior = secondary_prior / torch.sum(secondary_prior)  # Normalize if needed
        noisy_PPI = secondary_prior
    else:
        secondary_prior = None
        # Do NOT set noisy_PPI to None here; keep the default constructed from prior_mat
    
    print(f"[DEBUG] noisy_PPI is None: {noisy_PPI is None}")
    print(f"[DEBUG] noisy_prior_mat is None: {noisy_prior_mat is None}")

    odenet = ODENet(device, data_handler.dim, explicit_time=settings.get('explicit_time', False), neurons = settings.get('neurons_per_layer', 10), 
                    log_scale = settings.get('log_scale', False), init_bias_y = settings.get('init_bias_y', 0.0), 
                    init_sparsity = settings.get('init_sparsity', 0.95))
    odenet.float()

    if settings.get('pretrained_model', False):
        pretrained_model_file = 'output/2025-6-23(13;48)_simu_350genes_150samples_train_val_200epochs/best_val_model.pt'
        odenet.inherit_params(pretrained_model_file)

    print('Using optimizer: {}'.format(settings.get('optimizer', 'adam')))
    if settings.get('optimizer', 'adam') == 'rmsprop':
        opt = optim.RMSprop(odenet.parameters(), lr=settings.get('init_lr', 0.001), weight_decay=settings.get('weight_decay', 0.0))
    elif settings.get('optimizer', 'adam') == 'sgd':
        opt = optim.SGD(odenet.parameters(), lr=settings.get('init_lr', 0.001), weight_decay=settings.get('weight_decay', 0.0))
    elif settings.get('optimizer', 'adam') == 'adagrad':
        opt = optim.Adagrad(odenet.parameters(), lr=settings.get('init_lr', 0.001), weight_decay=settings.get('weight_decay', 0.0))
    else:
        if settings.get('pruning', False):
            # Defensive: check if odenet has the required attributes
            param_groups = []
            for layer in [
                getattr(odenet, 'net_sums', None) and getattr(odenet.net_sums, 'linear_out', None),
                getattr(odenet, 'net_prods', None) and getattr(odenet.net_prods, 'linear_out', None),
                getattr(odenet, 'net_alpha_combine_sums', None) and getattr(odenet.net_alpha_combine_sums, 'linear_out', None),
                getattr(odenet, 'net_alpha_combine_prods', None) and getattr(odenet.net_alpha_combine_prods, 'linear_out', None)
            ]:
                if isinstance(layer, torch.nn.Module):
                    if hasattr(layer, 'weight') and layer.weight is not None:
                        param_groups.append({'params': layer.weight})
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        param_groups.append({'params': layer.bias})
            # Add gene multipliers with higher learning rate
            if hasattr(odenet, 'gene_multipliers'):
                param_groups.append({'params': odenet.gene_multipliers, 'lr': 5*settings.get('init_lr', 0.001)})
            opt = optim.Adam(param_groups, lr=settings.get('init_lr', 0.001), weight_decay=settings.get('weight_decay', 0.0))
        else:
            opt = optim.Adam(odenet.parameters(), lr=settings.get('init_lr', 0.001), weight_decay=settings.get('weight_decay', 0.0))
    
    with open('{}/network.txt'.format(output_root_dir), 'w') as net_file:
        net_file.write(odenet.__str__())
        net_file.write('\n\n\n')
        net_file.write(inspect.getsource(ODENet.forward))
        net_file.write('\n')
        if settings.get('pruning', False):
            net_file.write('DASH Training with Pruning and Prior Knowledge')
            net_file.write('\n')
            net_file.write('PHX lambda (weight given to data loss) = {}'.format(loss_lambda))
            net_file.write('\n')
            net_file.write('causal lottery!')
            net_file.write('\n')
            net_file.write('doing PPI mask + T mask')
            net_file.write('\n')
            net_file.write('.....')
            net_file.write('Considering multipliers for final layer pruning')   
            net_file.write('.....')
            net_file.write('\n')
            net_file.write('pruning score lambda (PPI, Motif) = ({}, {})'.format(0.05, 0.01))
            net_file.write('\n')
            net_file.write('Initial hit = {} at epoch {}, then prune {} every {} epochs'.format(0.7, 3, 0.1, 10))
        else:
            net_file.write('Regular Neural ODE Training (No Pruning, No Prior Knowledge)')
            net_file.write('\n')
            net_file.write('Training with data loss only')
    
    # Init plot
    if settings.get('viz', False):
        visualizer = Visualizator1D(data_handler, odenet, settings)

    # Training loop
    epoch_times = []
    total_time = 0
    training_loss = []
    validation_loss = []  # Initialize validation_loss list
    prior_losses = []
    true_mean_losses = []
    true_mean_losses_init_val_based = []

    min_loss = 0
    if settings.get('batch_type', 'single') == 'single':
        iterations_in_epoch = ceil(data_handler.train_data_length / settings.get('batch_size', 32))
    elif settings.get('batch_type', 'single') == 'trajectory':
        iterations_in_epoch = data_handler.train_data_length
    else:
        iterations_in_epoch = ceil(data_handler.train_data_length / settings.get('batch_size', 32))
    
    tot_epochs = settings.get('epochs', 200)
    rep_epochs = [5, 10, 15, 30, 40, 50, 75, 100, 120, 150, 180, 200,220, 240, 260, 280, 300, 350, tot_epochs]
    viz_epochs = rep_epochs
    rep_epochs_train_losses = []
    rep_epochs_val_losses = []
    rep_epochs_mu_losses = []
    rep_epochs_time_so_far = []
    rep_epochs_so_far = []
    consec_epochs_failed = 0
    epochs_to_fail_to_terminate = 40
    
    scheduler = setOptimizerLRScheduler(opt)
            
    if settings.get('viz', False):
        with torch.no_grad():
            visualizer.visualize()
            visualizer.plot()
            visualizer.save(img_save_dir, 0)
    
    start_time = perf_counter()

    # Add these lists to store per-epoch metrics
    sparsity_total_nonzero = []
    sparsity_median_nonzero = []
    sparsity_min_nonzero = []
    sparsity_max_nonzero = []
    sparsity_proportion_nonzero = []
    sparsity_l1_norm = []

    # After reading settings, ensure it is a dict
    if settings is None:
        settings = {}

    # Create comprehensive metrics file with header
    comprehensive_metrics_file = f'{output_root_dir}comprehensive_metrics.csv'
    header = "epoch,train_loss,val_loss,true_mse,true_r2,prior_loss,total_nonzero,median_nonzero,min_nonzero,max_nonzero,proportion_nonzero,l1_norm,time_hrs\n"
    with open(comprehensive_metrics_file, 'w') as f:
        f.write(header)

    for epoch in range(1, tot_epochs + 1):
        print()
        print("[Running epoch {}/{}]".format(epoch, settings.get('epochs', 200)))

        # Pruning logic (only if pruning is enabled)
        if settings.get('pruning', False):
            masking_start_epoch = 3
            initial_hit_perc = 0.70
            num_epochs_till_mask = 10
            prune_perc = 0.10
            pruning_score_lambda_PPI = 0.05 
            pruning_score_lambda_motif = 0.01
            
            print(f"[DEBUG] Epoch {epoch}: checking pruning condition")
            print(f"[DEBUG] masking_start_epoch = {masking_start_epoch}, epoch == masking_start_epoch = {epoch == masking_start_epoch}")
            print(f"[DEBUG] epoch % num_epochs_till_mask = {epoch % num_epochs_till_mask}")
            
            if (epoch == masking_start_epoch) or (epoch == masking_start_epoch +0 ) or (epoch >= masking_start_epoch and epoch % num_epochs_till_mask == 0):
                print(f"[DEBUG] Pruning condition met for epoch {epoch}!")
                if epoch == masking_start_epoch or epoch == masking_start_epoch + 0:
                    prune_this_epoch = initial_hit_perc
                    print(f"[DEBUG] Using initial hit percentage: {prune_this_epoch}")
                else:
                    prune_this_epoch = prune_perc
                    print(f"[DEBUG] Using regular prune percentage: {prune_this_epoch}")

                total_pruned = 0
                total_params = 0

                for name, module in odenet.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        if name == 'net_sums.linear_out':
                            if module.weight is not None:
                                current_NN_weights_abs = abs(module.weight.detach())
                                current_NN_weights_abs = current_NN_weights_abs/torch.sum(current_NN_weights_abs)
                        elif name == 'net_prods.linear_out':
                            if module.weight is not None:
                                current_NN_weights_abs = torch.exp(module.weight.detach())
                                current_NN_weights_abs = current_NN_weights_abs/torch.sum(current_NN_weights_abs)
                        elif name in ['net_alpha_combine_sums.linear_out','net_alpha_combine_prods.linear_out' ]:
                            if module.weight is not None and hasattr(odenet, 'gene_multipliers') and odenet.gene_multipliers is not None:
                                current_NN_weights_abs = abs(module.weight.detach())
                                current_gene_mult_ReLU = torch.relu(odenet.gene_multipliers.detach().t())
                                current_NN_weights_abs = current_NN_weights_abs * current_gene_mult_ReLU
                                current_NN_weights_abs = current_NN_weights_abs/torch.sum(current_NN_weights_abs)
                        if name in ['net_sums.linear_out','net_prods.linear_out'] and ((epoch == masking_start_epoch) or epoch % num_epochs_till_mask == 0):
                            print(f"[DEBUG] Pruning first layer: {name}, noisy_PPI is None: {noisy_PPI is None}")
                            mask_curr = my_current_custom_pruning_scores[name]
                            if mask_curr is not None and noisy_PPI is not None:
                                S_S_transpose_inv = torch.inverse(torch.matmul(mask_curr, torch.transpose(mask_curr,0,1)))
                                S_PPI = torch.matmul(mask_curr, noisy_PPI)
                                S_mask_best_guess = torch.matmul(S_S_transpose_inv, S_PPI)
                                updated_score = pruning_score_lambda_PPI * torch.abs(S_mask_best_guess) + (1 - pruning_score_lambda_PPI) * current_NN_weights_abs
                                prune.l1_unstructured(module, name='weight', amount=prune_this_epoch, importance_scores = updated_score)
                                my_current_custom_pruning_scores[name] = updated_score
                                print(f"[DEBUG] Pruned {name} with {prune_this_epoch*100:.1f}% using PPI. Nonzero weights: {(module.weight != 0).sum().item()}")
                            else:
                                print(f"[DEBUG] Skipped pruning {name} (mask_curr or noisy_PPI is None)")
                        elif name in['net_alpha_combine_sums.linear_out', 'net_alpha_combine_prods.linear_out'] and ((epoch == masking_start_epoch +0 ) or (epoch % num_epochs_till_mask == 0)):
                            print(f"[DEBUG] Pruning second layer: {name}, noisy_prior_mat is None: {noisy_prior_mat is None}")
                            if name == 'net_alpha_combine_sums.linear_out':
                                incoming_mask_curr = my_current_custom_pruning_scores['net_sums.linear_out']
                            else:
                                incoming_mask_curr = my_current_custom_pruning_scores['net_prods.linear_out']
                            if incoming_mask_curr is not None and noisy_prior_mat is not None:
                                T_tranpose_S_transpose = torch.transpose(torch.matmul(incoming_mask_curr,abs(noisy_prior_mat)),0,1)
                                S_S_transpose_inv = torch.inverse(torch.matmul(incoming_mask_curr, torch.transpose(incoming_mask_curr,0,1)))
                                C_mask_best_guess = torch.matmul(T_tranpose_S_transpose, S_S_transpose_inv)
                                updated_score = pruning_score_lambda_motif * torch.abs(C_mask_best_guess)  + (1 - pruning_score_lambda_motif) * current_NN_weights_abs
                                prune.l1_unstructured(module, name='weight', amount=prune_this_epoch, importance_scores = updated_score)
                                my_current_custom_pruning_scores[name] = updated_score
                                print(f"[DEBUG] Pruned {name} with {prune_this_epoch*100:.1f}% using motif. Nonzero weights: {(module.weight != 0).sum().item()}")
                            else:
                                print(f"[DEBUG] Skipped pruning {name} (incoming_mask_curr or noisy_prior_mat is None)")
                        if module.weight is not None:
                            total_params += module.weight.nelement()
                            total_pruned += torch.sum(module.weight == 0)
                
                print("Updated mask based on prior! Current perc pruned: {:.2%}, num pruned: {}".format(total_pruned/total_params, total_pruned))
                
                if prune_perc > 0 or initial_hit_perc > 0:
                    reset_lr(opt, True, settings.get('init_lr', 0.001))
                    scheduler = setOptimizerLRScheduler(opt)

        # Compute and store global sparsity metrics AFTER pruning (if applicable)
        total_nonzero, median_nonzero, min_nonzero, max_nonzero, proportion_nonzero, l1_norm = get_global_sparsity_metrics(odenet)
        print(f"[DEBUG] Epoch {epoch} sparsity calculation: total_nonzero={total_nonzero}, proportion_nonzero={proportion_nonzero:.3f}")
        sparsity_total_nonzero.append(total_nonzero)
        sparsity_median_nonzero.append(median_nonzero)
        sparsity_min_nonzero.append(min_nonzero)
        sparsity_max_nonzero.append(max_nonzero)
        sparsity_proportion_nonzero.append(proportion_nonzero)
        sparsity_l1_norm.append(l1_norm)

        start_epoch_time = perf_counter()
        iteration_counter = 1
        data_handler.reset_epoch()
        this_epoch_total_train_loss = 0
        this_epoch_total_prior_loss = 0
        
        if settings.get('pruning', False):
            print("current loss_lambda =", loss_lambda)
        
        if settings.get('verbose', False):
            pbar = tqdm(total=iterations_in_epoch, desc="Training loss:")
        while not data_handler.epoch_done:
            start_batch_time = perf_counter()
            
            # Always use training_step directly (like original working version)
            loss_list = training_step(odenet, data_handler, opt, settings.get('method', 'dopri5'), settings.get('batch_size', 32), 
                                    batch_for_prior, prior_grad, loss_lambda)
            loss = loss_list[0]
            prior_loss = loss_list[1]
            
            this_epoch_total_train_loss += loss.item()
            this_epoch_total_prior_loss += prior_loss.item()
            iteration_counter += 1

            if settings.get('verbose', False):
                if settings.get('pruning', False):
                    pbar.set_description("Training loss, Prior loss: {:.2E}, {:.2E}".format(loss.item(), prior_loss.item()))
                else:
                    pbar.set_description("Training loss: {:.2E}".format(loss.item()))
        
        epoch_times.append(perf_counter() - start_epoch_time)

        #Epoch done, now handle training loss
        train_loss = this_epoch_total_train_loss/iterations_in_epoch
        training_loss.append(train_loss)
        # Always store prior loss (like original working version)
        prior_losses.append(this_epoch_total_prior_loss/iterations_in_epoch)
        
        mu_loss = get_true_val_set_r2(odenet, data_handler, settings.get('method', 'dopri5'), settings.get('batch_type', 'single'))
        true_mean_losses.append(mu_loss[1])
        true_mean_losses_init_val_based.append(mu_loss[0])
                
        if epoch == 1:
                min_train_loss = train_loss
        else:
            if train_loss < min_train_loss:
                min_train_loss = train_loss
                true_loss_of_min_train_model =  mu_loss[1]

        if settings.get('verbose', False):
            pbar.close()

        # Handle validation loss
        if data_handler.n_val > 0:
            val_loss_list = validation(odenet, data_handler, settings.get('method', 'dopri5'), settings.get('explicit_time', False))
            val_loss = val_loss_list[0]
            validation_loss.append(val_loss)  # Append validation loss for this epoch
            if epoch == 1:
                min_val_loss = val_loss
                true_loss_of_min_val_model = mu_loss[1]
                print('Model improved, saving current model')
                best_vaL_model_so_far = odenet
                save_model(odenet, output_root_dir, 'best_val_model')
            else:
                if val_loss < min_val_loss:
                    consec_epochs_failed = 0
                    min_val_loss = val_loss
                    true_loss_of_min_val_model =  mu_loss[1]
                    print('Model improved, saving current model')
                    save_model(odenet, output_root_dir, 'best_val_model')
                else:
                    consec_epochs_failed = consec_epochs_failed + 1

            print("Validation loss {:.5E}, using {} points".format(val_loss, val_loss_list[1]))
            scheduler.step(val_loss)
        else:
            validation_loss.append(0.0)  # Append 0 if no validation data

        print("Overall training loss {:.5E}".format(train_loss))
        print("True MSE of val traj (pairwise): {:.5E}".format(mu_loss[1]))
        print("True R^2 of val traj (pairwise): {:.2%}".format(mu_loss[0]))

        # Save comprehensive metrics every epoch (using sparsity calculated at beginning of epoch)
        current_time_hrs = (perf_counter() - start_time) / 3600
        comprehensive_metrics = np.array([
            epoch,
            train_loss,
            val_loss if data_handler.n_val > 0 else 0.0,
            mu_loss[1],  # true MSE
            mu_loss[0],  # true R2
            prior_losses[-1] if prior_losses else 0.0,
            sparsity_total_nonzero[-1],  # Use the sparsity calculated at beginning of epoch
            sparsity_median_nonzero[-1],
            sparsity_min_nonzero[-1],
            sparsity_max_nonzero[-1],
            sparsity_proportion_nonzero[-1],
            sparsity_l1_norm[-1],
            current_time_hrs
        ]).reshape(1, -1)
        
        with open(comprehensive_metrics_file, 'a') as f:
            np.savetxt(f, comprehensive_metrics, delimiter=',', fmt='%.6e')
        
        print(f"Comprehensive metrics saved for epoch {epoch}")
            
        if (settings.get('viz', False) and epoch in viz_epochs) or (settings.get('viz', False) and epoch in rep_epochs) or (consec_epochs_failed == epochs_to_fail_to_terminate):
            print("Saving plot")
            with torch.no_grad():
                visualizer.visualize()
                visualizer.plot()
                visualizer.save(img_save_dir, epoch)
        
        if (epoch in rep_epochs) or (consec_epochs_failed == epochs_to_fail_to_terminate):
            print()
            rep_epochs_so_far.append(epoch)
            print("Epoch=", epoch)
            rep_time_so_far = (perf_counter() - start_time)/3600
            print("Time so far= ", rep_time_so_far, "hrs")
            rep_epochs_time_so_far.append(rep_time_so_far)
            print("Best training (MSE) so far= ", min_train_loss)
            rep_epochs_train_losses.append(min_train_loss)
            if data_handler.n_val > 0:
                print("Best validation (MSE) so far = ", min_val_loss)
                rep_epochs_val_losses.append(min_val_loss)
                rep_epochs_mu_losses.append(true_loss_of_min_val_model)
            else:
                print("True loss of best training model (MSE) = ", 0)
            print("Saving MSE plot...")
            plot_MSE(epoch, training_loss, validation_loss, true_mean_losses, true_mean_losses_init_val_based, prior_losses, img_save_dir)    

            print("Saving losses..")
            if data_handler.n_val > 0:
                L = [rep_epochs_so_far, rep_epochs_time_so_far, rep_epochs_train_losses, rep_epochs_val_losses, rep_epochs_mu_losses]
                np.savetxt('{}rep_epoch_losses.csv'.format(output_root_dir), np.transpose(L), delimiter=',')    
            
            # Save current sparsity metrics to a separate file
            current_sparsity_data = np.array([
                epoch, 
                sparsity_total_nonzero[-1] if sparsity_total_nonzero else 0,
                sparsity_median_nonzero[-1] if sparsity_median_nonzero else 0,
                sparsity_min_nonzero[-1] if sparsity_min_nonzero else 0,
                sparsity_max_nonzero[-1] if sparsity_max_nonzero else 0,
                sparsity_proportion_nonzero[-1] if sparsity_proportion_nonzero else 0,
                sparsity_l1_norm[-1] if sparsity_l1_norm else 0
            ]).reshape(1, -1)
            
            # Append to sparsity tracking file
            sparsity_file = f'{output_root_dir}sparsity_tracking.csv'
            if epoch == rep_epochs[0]:  # First reporting epoch
                # Create header and save first row
                header = "epoch,total_nonzero,median_nonzero,min_nonzero,max_nonzero,proportion_nonzero,l1_norm\n"
                with open(sparsity_file, 'w') as f:
                    f.write(header)
                    np.savetxt(f, current_sparsity_data, delimiter=',', fmt='%.6e')
            else:
                # Append to existing file
                with open(sparsity_file, 'a') as f:
                    np.savetxt(f, current_sparsity_data, delimiter=',', fmt='%.6e')
            
            print(f"Sparsity metrics saved for epoch {epoch}")

        # Save layer-wise histograms before/after pruning epochs (or at the same epochs for base)
        pruning_epochs = [3] + list(range(10, settings.get('epochs', 200)+1, 10))
        if epoch in pruning_epochs:
            layerwise_before = get_layerwise_nonzero_weights(odenet)
            save_layerwise_histograms(layerwise_before, epoch, 'before', '{}histograms/'.format(output_root_dir))
            # ...pruning logic happens here...
            layerwise_after = get_layerwise_nonzero_weights(odenet)
            save_layerwise_histograms(layerwise_after, epoch, 'after', '{}histograms/'.format(output_root_dir))

        if consec_epochs_failed==epochs_to_fail_to_terminate:
            print("Went {} epochs without improvement; terminating.".format(epochs_to_fail_to_terminate))
            break

    total_time = perf_counter() - start_time
    
    save_model(odenet, output_root_dir, 'final_model')

    print("Saving times")
    np.savetxt('{}epoch_times.csv'.format(output_root_dir), epoch_times, delimiter=',')

    # Save all losses and sparsity metrics to CSV
    all_metrics = np.c_[training_loss, validation_loss, true_mean_losses, true_mean_losses_init_val_based, prior_losses,
                        sparsity_total_nonzero, sparsity_median_nonzero, sparsity_min_nonzero, sparsity_max_nonzero,
                        sparsity_proportion_nonzero, sparsity_l1_norm]
    np.savetxt(f'{img_save_dir}/full_loss_info.csv', all_metrics, delimiter=',')

    print("DONE!") 