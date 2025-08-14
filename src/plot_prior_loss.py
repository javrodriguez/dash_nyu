import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Get CSV file path from command line argument
if len(sys.argv) != 2:
    print("Usage: python plot_prior_loss.py <path_to_full_loss_info.csv>")
    print("Example: python plot_prior_loss.py output/2025-6-23(18;43)_simu_350genes_150samples_train_val_200epochs/img/full_loss_info.csv")
    sys.exit(1)

csv_path = sys.argv[1]

# Check if file exists
if not os.path.exists(csv_path):
    print(f"Error: File '{csv_path}' not found!")
    sys.exit(1)

# Read the CSV file
print(f"Reading data from: {csv_path}")
data = pd.read_csv(csv_path, header=None)

# Check if we have 4 or 5 columns (prior loss might be added)
if data.shape[1] == 4:
    data.columns = ['training_loss', 'validation_loss', 'true_mean_losses', 'r_squared_values']
    has_prior_loss = False
elif data.shape[1] == 5:
    data.columns = ['training_loss', 'validation_loss', 'true_mean_losses', 'r_squared_values', 'prior_losses']
    has_prior_loss = True
else:
    print(f"Unexpected number of columns: {data.shape[1]}")
    exit()

# Create figure with subplots
if has_prior_loss:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
else:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

epochs = range(1, len(data) + 1)

# Find the best validation epoch (lowest MSE) if validation loss exists
best_val_epoch = None
if data['validation_loss'].sum() > 0:
    best_val_epoch = data['validation_loss'].idxmin() + 1  # +1 because epochs start at 1

# First subplot: Training and Validation Loss
ax1.plot(epochs, data['training_loss'], color='blue', label='Training loss', linewidth=2)

# Plot validation loss if it exists and is not all zeros
if data['validation_loss'].sum() > 0:
    ax1.plot(epochs, data['validation_loss'], color='red', label='Validation loss', linewidth=2)
    
    # Add vertical line at best validation epoch
    best_val_loss = data['validation_loss'].min()
    ax1.axvline(x=best_val_epoch, color='black', linestyle='--', alpha=0.7, linewidth=2)
    ax1.annotate(f'Best Val\nEpoch {best_val_epoch}\nMSE: {best_val_loss:.2e}', 
                xy=(best_val_epoch, best_val_loss), 
                xytext=(best_val_epoch + 5, best_val_loss * 1.5),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=10, color='black', ha='left')

ax1.set_yscale('log')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Error (MSE)', fontsize=12)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title('Training and Validation Loss Over Epochs\n(All Time Points)', fontsize=14, pad=10)

# Second subplot: True Mean Losses
ax2.plot(epochs, data['true_mean_losses'], color='green', label='True Mean Losses', linewidth=2)
if best_val_epoch is not None:
    best_true_mean_loss = data.loc[best_val_epoch-1, 'true_mean_losses']
    ax2.axvline(x=best_val_epoch, color='black', linestyle='--', alpha=0.7, linewidth=2)
    ax2.annotate(f'Best Val\nEpoch {best_val_epoch}\nMSE: {best_true_mean_loss:.2e}', 
                xy=(best_val_epoch, best_true_mean_loss), 
                xytext=(best_val_epoch + 5, best_true_mean_loss * 1.5),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=10, color='black', ha='left')
ax2.set_yscale('log')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('True Mean Losses (MSE)', fontsize=12)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_title('True Mean Losses Over Epochs\n(Final Time Point Only)', fontsize=14, pad=10)

# Third subplot: R² Values
ax3.plot(epochs, data['r_squared_values'], color='orange', label='R² values', linewidth=2)
if best_val_epoch is not None:
    best_r2_value = data.loc[best_val_epoch-1, 'r_squared_values']
    ax3.axvline(x=best_val_epoch, color='black', linestyle='--', alpha=0.7, linewidth=2)
    ax3.annotate(f'Best Val\nEpoch {best_val_epoch}\nR²: {best_r2_value:.3f}', 
                xy=(best_val_epoch, best_r2_value), 
                xytext=(best_val_epoch + 5, best_r2_value * 0.9),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=10, color='black', ha='left')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('R² Values', fontsize=12)
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_title('R² Values Over Epochs\n(Final Time Point Only)', fontsize=14, pad=10)

# Fourth subplot: Prior Loss (if available) or empty
if has_prior_loss:
    ax4.plot(epochs, data['prior_losses'], color='magenta', label='Prior loss', linewidth=2)
    if best_val_epoch is not None:
        best_prior_loss = data.loc[best_val_epoch-1, 'prior_losses']
        ax4.axvline(x=best_val_epoch, color='black', linestyle='--', alpha=0.7, linewidth=2)
        ax4.annotate(f'Best Val\nEpoch {best_val_epoch}\nLoss: {best_prior_loss:.2e}', 
                    xy=(best_val_epoch, best_prior_loss), 
                    xytext=(best_val_epoch + 5, best_prior_loss * 1.5),
                    arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                    fontsize=10, color='black', ha='left')
    ax4.set_yscale('log')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Prior Loss', fontsize=12)
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Prior Loss Over Epochs\n(Synthetic Data + Prior Matrix)', fontsize=14, pad=10)
else:
    ax4.text(0.5, 0.5, 'Prior Loss Data\nNot Available\n\nModify plot_MSE function\nto save prior_losses\nto CSV file', 
             ha='center', va='center', transform=ax4.transAxes, fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax4.set_title('Prior Loss Over Epochs\n(Synthetic Data + Prior Matrix)', fontsize=14, pad=10)

# Adjust layout
plt.tight_layout()

# Determine output path for the plot
plot_dir = os.path.dirname(csv_path)
plot_base = os.path.splitext(os.path.basename(csv_path))[0]
plot_path = os.path.join(plot_dir, plot_base + '.png')

# Save the plot
plt.savefig(plot_path)
print(f"Plot saved to: {plot_path}")

if best_val_epoch is not None:
    print(f"Best validation epoch: {best_val_epoch}")
    print(f"  - Validation MSE: {data.loc[best_val_epoch-1, 'validation_loss']:.2e}")
    print(f"  - True Mean Loss: {data.loc[best_val_epoch-1, 'true_mean_losses']:.2e}")
    print(f"  - R² Value: {data.loc[best_val_epoch-1, 'r_squared_values']:.3f}")
    if has_prior_loss:
        print(f"  - Prior Loss: {data.loc[best_val_epoch-1, 'prior_losses']:.2e}")
if not has_prior_loss:
    print("\nNOTE: Prior loss data is not available in the CSV file.")
    print("To include prior loss, modify the plot_MSE function in train_DASH_simu.py")
    print("to save prior_losses to the CSV file and re-run the training.") 