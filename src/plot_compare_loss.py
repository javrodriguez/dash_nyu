import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

from matplotlib.colors import to_rgba
from matplotlib.ticker import ScalarFormatter
from adjustText import adjust_text

def darken(color, factor=0.6):
    rgba = np.array(to_rgba(color))
    return tuple(rgba[:3] * factor) + (rgba[3],)

# Parse arguments
no_annotate = False
args = sys.argv[1:]
if '--no-annotate' in args:
    no_annotate = True
    args.remove('--no-annotate')

if len(args) < 2:
    print("Usage: python plot_compare_loss.py <csv1> <csv2> [name1] [name2] [--no-annotate]")
    sys.exit(1)

csv1 = args[0]
csv2 = args[1]
name1 = args[2] if len(args) > 2 else "Base"
name2 = args[3] if len(args) > 3 else "DASH"

# Read both CSVs
def read_loss_csv(path):
    data = pd.read_csv(path, header=0)
    if data.shape[1] == 4:
        data.columns = ['training_loss', 'validation_loss', 'true_mean_losses', 'r_squared_values']
        has_prior_loss = False
    elif data.shape[1] == 5:
        data.columns = ['training_loss', 'validation_loss', 'true_mean_losses', 'r_squared_values', 'prior_losses']
        has_prior_loss = True
    elif data.shape[1] == 11:
        data.columns = [
            'training_loss', 'validation_loss', 'true_mean_losses', 'r_squared_values', 'prior_losses',
            'sparsity_total_nonzero', 'sparsity_median_nonzero', 'sparsity_min_nonzero', 'sparsity_max_nonzero',
            'sparsity_proportion_nonzero', 'sparsity_l1_norm'
        ]
        has_prior_loss = True
    elif data.shape[1] == 13:
        data.columns = [
            'epoch', 'training_loss', 'validation_loss', 'true_mean_losses', 'r_squared_values', 'prior_losses',
            'sparsity_total_nonzero', 'sparsity_median_nonzero', 'sparsity_min_nonzero', 'sparsity_max_nonzero',
            'sparsity_proportion_nonzero', 'sparsity_l1_norm', 'time_hrs'
        ]
        # For plotting, drop 'epoch' and 'time_hrs'
        data = data.drop(columns=['epoch', 'time_hrs'])
        has_prior_loss = True
    else:
        raise ValueError(f"Unexpected number of columns: {data.shape[1]}")
    return data, has_prior_loss

data1, has_prior1 = read_loss_csv(csv1)
data2, has_prior2 = read_loss_csv(csv2)

if has_prior1 != has_prior2:
    print("Warning: Only one run has prior loss. The prior loss plot will only show available data.")

epochs1 = range(1, len(data1) + 1)
epochs2 = range(1, len(data2) + 1)

# Find best validation epochs
idx1 = data1['validation_loss'].idxmin()
idx2 = data2['validation_loss'].idxmin()
best_val_epoch1 = int(np.asarray(idx1).item()) + 1 if data1['validation_loss'].sum() > 0 else None
best_val_epoch2 = int(np.asarray(idx2).item()) + 1 if data2['validation_loss'].sum() > 0 else None

# Set up colors
colors = {
    'train':    {'train_1': 'blue', 'train_2': darken('blue')},
    'val':      {'train_1': 'red', 'train_2': darken('red')},
    'mean':     {'train_1': 'green', 'train_2': darken('green')},
    'r2':       {'train_1': 'orange', 'train_2': darken('orange')},
    'prior':    {'train_1': 'magenta', 'train_2': darken('magenta')},
}

fig, ((ax1, ax2), (ax3, ax4), (ax5, _)) = plt.subplots(3, 2, figsize=(15, 18))

# Helper function to clamp annotation positions
def clamp(val, vmin, vmax):
    return max(vmin, min(val, vmax))

# Training and Validation Loss
ax1.plot(epochs1, data1['training_loss'], color=colors['train']['train_1'], label=f'Training loss ({name1})', linewidth=2)
ax1.plot(epochs2, data2['training_loss'], color=colors['train']['train_2'], label=f'Training loss ({name2})', linewidth=2)
if data1['validation_loss'].sum() > 0:
    ax1.plot(epochs1, data1['validation_loss'], color=colors['val']['train_1'], label=f'Validation loss ({name1})', linewidth=2)
if data2['validation_loss'].sum() > 0:
    # Ensure DASH validation loss is plotted with a distinct color and on top
    ax1.plot(epochs2, data2['validation_loss'], color=colors['val']['train_2'], label=f'Validation loss ({name2})', linewidth=2, zorder=10)
ax1.set_yscale('log')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Error (MSE)', fontsize=12)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title('Training and Validation Loss Over Epochs', fontsize=14, pad=10)

# True Mean Losses
ax2.plot(epochs1, data1['true_mean_losses'], color=colors['mean']['train_1'], label=f'True Mean Losses ({name1})', linewidth=2)
ax2.plot(epochs2, data2['true_mean_losses'], color=colors['mean']['train_2'], label=f'True Mean Losses ({name2})', linewidth=2)
ax2.set_yscale('log')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('True Mean Losses (MSE)', fontsize=12)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_title('True Mean Losses Over Epochs', fontsize=14, pad=10)

# R² Values
ax3.plot(epochs1, data1['r_squared_values'], color=colors['r2']['train_1'], label=f'R² values ({name1})', linewidth=2)
ax3.plot(epochs2, data2['r_squared_values'], color=colors['r2']['train_2'], label=f'R² values ({name2})', linewidth=2)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('R² Values', fontsize=12)
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_title('R² Values Over Epochs', fontsize=14, pad=10)

# Prior Loss
if has_prior1 or has_prior2:
    if has_prior1:
        ax4.plot(epochs1, data1['prior_losses'], color=colors['prior']['train_1'], label=f'Prior loss ({name1})', linewidth=2)
    if has_prior2:
        ax4.plot(epochs2, data2['prior_losses'], color=colors['prior']['train_2'], label=f'Prior loss ({name2})', linewidth=2)
    ax4.set_yscale('log')
    # Add more y-tick labels at the extremes
    prior_losses_all = []
    if has_prior1:
        prior_losses_all.extend(data1['prior_losses'].values)
    if has_prior2:
        prior_losses_all.extend(data2['prior_losses'].values)
    if len(prior_losses_all) > 0:
        min_loss = np.nanmin(prior_losses_all)
        max_loss = np.nanmax(prior_losses_all)
        # Get current ticks and add min/max if not present
        ticks = list(ax4.get_yticks())
        ticks += [min_loss, max_loss]
        ticks = np.unique([t for t in ticks if t > 0])  # Remove non-positive for log scale
        ax4.set_yticks(ticks)
        ax4.get_yaxis().set_major_formatter(ScalarFormatter())
        # Set y-axis limits relative to min and max
        ax4.set_ylim(min_loss * 0.5, max_loss * 1.5)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Prior Loss', fontsize=12)
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Prior Loss Over Epochs', fontsize=14, pad=10)
else:
    ax4.text(0.5, 0.5, 'Prior Loss Data\nNot Available', ha='center', va='center', transform=ax4.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax4.set_title('Prior Loss Over Epochs', fontsize=14, pad=10)

# Sparsity plot (proportion_nonzero)
if 'sparsity_proportion_nonzero' in data1.columns and 'sparsity_proportion_nonzero' in data2.columns:
    ax5.plot(epochs1, data1['sparsity_proportion_nonzero'], color='black', label=f'Sparsity ({name1})', linewidth=2)
    ax5.plot(epochs2, data2['sparsity_proportion_nonzero'], color='gray', label=f'Sparsity ({name2})', linewidth=2)
    # Add secondary y-axis for number of nonzero parameters
    ax5b = ax5.twinx()
    if 'sparsity_total_nonzero' in data1.columns:
        ax5b.plot(epochs1, data1['sparsity_total_nonzero'], color='black', linestyle=':', label=f'Nonzero Params ({name1})', linewidth=2)
    if 'sparsity_total_nonzero' in data2.columns:
        ax5b.plot(epochs2, data2['sparsity_total_nonzero'], color='gray', linestyle=':', label=f'Nonzero Params ({name2})', linewidth=2)
    ax5b.set_ylabel('Number of Nonzero Parameters', fontsize=12)
    # Combine legends
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5b.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Proportion Nonzero', fontsize=12)
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Sparsity (Proportion Nonzero) Over Epochs', fontsize=14, pad=10)
else:
    ax5.text(0.5, 0.5, 'Sparsity Data\nNot Available', ha='center', va='center', transform=ax5.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax5.set_title('Sparsity (Proportion Nonzero) Over Epochs', fontsize=14, pad=10)

# Weight Value Range plot (bottom right)
ax_weight_range = _  # The unused subplot
if ('sparsity_min_nonzero' in data1.columns and 'sparsity_max_nonzero' in data1.columns and
    'sparsity_min_nonzero' in data2.columns and 'sparsity_max_nonzero' in data2.columns):
    # Restore to original: plot min, max, and median without log transformation
    ax_weight_range.fill_between(epochs2, data2['sparsity_min_nonzero'], data2['sparsity_max_nonzero'],
                                alpha=0.3, color='#1f77b4', label=f'{name2} Range')
    ax_weight_range.plot(epochs2, data2['sparsity_min_nonzero'], color='#1f77b4', linewidth=1, alpha=0.7)
    ax_weight_range.plot(epochs2, data2['sparsity_max_nonzero'], color='#1f77b4', linewidth=1, alpha=0.7)
    ax_weight_range.fill_between(epochs1, data1['sparsity_min_nonzero'], data1['sparsity_max_nonzero'],
                                alpha=0.3, color='#ff7f0e', label=f'{name1} Range')
    ax_weight_range.plot(epochs1, data1['sparsity_min_nonzero'], color='#ff7f0e', linewidth=1, alpha=0.7)
    ax_weight_range.plot(epochs1, data1['sparsity_max_nonzero'], color='#ff7f0e', linewidth=1, alpha=0.7)
    # Median lines
    if 'sparsity_median_nonzero' in data1.columns:
        ax_weight_range.plot(epochs1, data1['sparsity_median_nonzero'], color='#ff7f0e', linestyle='--', linewidth=2, label=f'{name1} Median')
    if 'sparsity_median_nonzero' in data2.columns:
        ax_weight_range.plot(epochs2, data2['sparsity_median_nonzero'], color='#1f77b4', linestyle='--', linewidth=2, label=f'{name2} Median')
    # Add vertical lines for best validation epochs
    if best_val_epoch1 is not None:
        ax_weight_range.axvline(x=best_val_epoch1, color='#ff7f0e', linestyle=':', alpha=0.7, linewidth=2)
    if best_val_epoch2 is not None:
        ax_weight_range.axvline(x=best_val_epoch2, color='#1f77b4', linestyle=':', alpha=0.7, linewidth=2)
    ax_weight_range.set_xlabel('Epoch')
    ax_weight_range.set_ylabel('Weight Value')
    ax_weight_range.set_title('Weight Value Range')
    ax_weight_range.legend()
    ax_weight_range.grid(True, alpha=0.3)
else:
    ax_weight_range.text(0.5, 0.5, 'Weight Range Data\nNot Available', ha='center', va='center', transform=ax_weight_range.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax_weight_range.set_title('Weight Value Range')

plt.tight_layout()
fig.canvas.draw()  # Ensure axis limits are finalized

# Add best epoch annotations after axis limits are finalized
texts_ax1 = []
texts_ax2 = []
texts_ax3 = []
texts_ax4 = []
texts_ax5 = []
if best_val_epoch1 is not None and not no_annotate:
    # ax1: Training/Validation Loss
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    best_val_loss = data1['validation_loss'].min()
    ann_x = clamp(best_val_epoch1 + 5, xlim[0], xlim[1])
    ann_y = clamp(best_val_loss * 1.5, ylim[0], ylim[1])
    texts_ax1.append(ax1.annotate(f'{name1}\nBest Val\nEpoch {best_val_epoch1}\nMSE: {best_val_loss:.2e}',
                xy=(best_val_epoch1, best_val_loss),
                xytext=(ann_x, ann_y),
                arrowprops=dict(arrowstyle='->', color=colors['val']['train_1'], alpha=0.7),
                fontsize=10, color=colors['val']['train_1'], ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8)))
    # ax2: True Mean Losses
    xlim = ax2.get_xlim()
    ylim = ax2.get_ylim()
    best_true_mean_loss = float(data1.loc[best_val_epoch1-1, 'true_mean_losses'])
    ann_x = clamp(best_val_epoch1 + 5, xlim[0], xlim[1])
    ann_y = clamp(best_true_mean_loss * 1.5 if best_true_mean_loss != 0 else 1, ylim[0], ylim[1])
    texts_ax2.append(ax2.annotate(f'{name1}\nBest Val\nEpoch {best_val_epoch1}\nMSE: {best_true_mean_loss:.2e}',
                xy=(best_val_epoch1, best_true_mean_loss),
                xytext=(ann_x, ann_y),
                arrowprops=dict(arrowstyle='->', color=colors['mean']['train_1'], alpha=0.7),
                fontsize=10, color=colors['mean']['train_1'], ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8)))
    # ax3: R2
    xlim = ax3.get_xlim()
    ylim = ax3.get_ylim()
    best_r2_value = float(data1.loc[best_val_epoch1-1, 'r_squared_values'])
    ann_x = clamp(best_val_epoch1 + 5, xlim[0], xlim[1])
    ann_y = clamp(best_r2_value * 0.9 if best_r2_value != 0 else 1, ylim[0], ylim[1])
    texts_ax3.append(ax3.annotate(f'{name1}\nBest Val\nEpoch {best_val_epoch1}\nR²: {best_r2_value:.3f}',
                xy=(best_val_epoch1, best_r2_value),
                xytext=(ann_x, ann_y),
                arrowprops=dict(arrowstyle='->', color=colors['r2']['train_1'], alpha=0.7),
                fontsize=10, color=colors['r2']['train_1'], ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8)))
    # ax4: Prior Loss
    if has_prior1:
        xlim = ax4.get_xlim()
        ylim = ax4.get_ylim()
        best_prior_loss = float(data1.loc[best_val_epoch1-1, 'prior_losses'])
        ann_x = clamp(best_val_epoch1 + 5, xlim[0], xlim[1])
        ann_y = clamp(best_prior_loss * 1.5 if best_prior_loss != 0 else 1, ylim[0], ylim[1])
        texts_ax4.append(ax4.annotate(f'{name1}\nBest Val\nEpoch {best_val_epoch1}\nLoss: {best_prior_loss:.2e}',
                    xy=(best_val_epoch1, best_prior_loss),
                    xytext=(ann_x, ann_y),
                    arrowprops=dict(arrowstyle='->', color=colors['prior']['train_1'], alpha=0.7),
                    fontsize=10, color=colors['prior']['train_1'], ha='left',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8)))
    # ax5: Sparsity
    if 'sparsity_proportion_nonzero' in data1.columns:
        xlim = ax5.get_xlim()
        ylim = ax5.get_ylim()
        sparsity1 = data1.loc[best_val_epoch1-1, 'sparsity_proportion_nonzero']
        ann_x = clamp(best_val_epoch1 + 5, xlim[0], xlim[1])
        ann_y = clamp(min(sparsity1 + 0.1, 1.0), ylim[0], ylim[1])
        texts_ax5.append(ax5.annotate(f'{name1}\nBest Val\nEpoch {best_val_epoch1}\nSparsity: {sparsity1:.3f} ({sparsity1*100:.1f}%)',
                    xy=(best_val_epoch1, sparsity1),
                    xytext=(ann_x, ann_y),
                    arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                    fontsize=10, color='black', ha='left',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8)))
if best_val_epoch2 is not None and not no_annotate:
    # ax1: Training/Validation Loss
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    best_val_loss2 = data2['validation_loss'].min()
    ann_x = clamp(best_val_epoch2 + 5, xlim[0], xlim[1])
    ann_y = clamp(best_val_loss2 * 1.5, ylim[0], ylim[1])
    texts_ax1.append(ax1.annotate(f'{name2}\nBest Val\nEpoch {best_val_epoch2}\nMSE: {best_val_loss2:.2e}',
                xy=(best_val_epoch2, best_val_loss2),
                xytext=(ann_x, ann_y),
                arrowprops=dict(arrowstyle='->', color=colors['val']['train_2'], alpha=0.7),
                fontsize=10, color=colors['val']['train_2'], ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8)))
    # ax2: True Mean Losses
    xlim = ax2.get_xlim()
    ylim = ax2.get_ylim()
    best_true_mean_loss2 = float(data2.loc[best_val_epoch2-1, 'true_mean_losses'])
    ann_x = clamp(best_val_epoch2 + 5, xlim[0], xlim[1])
    ann_y = clamp(best_true_mean_loss2 * 1.5 if best_true_mean_loss2 != 0 else 1, ylim[0], ylim[1])
    texts_ax2.append(ax2.annotate(f'{name2}\nBest Val\nEpoch {best_val_epoch2}\nMSE: {best_true_mean_loss2:.2e}',
                xy=(best_val_epoch2, best_true_mean_loss2),
                xytext=(ann_x, ann_y),
                arrowprops=dict(arrowstyle='->', color=colors['mean']['train_2'], alpha=0.7),
                fontsize=10, color=colors['mean']['train_2'], ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8)))
    # ax3: R2
    xlim = ax3.get_xlim()
    ylim = ax3.get_ylim()
    best_r2_value2 = float(data2.loc[best_val_epoch2-1, 'r_squared_values'])
    ann_x = clamp(best_val_epoch2 + 5, xlim[0], xlim[1])
    ann_y = clamp(best_r2_value2 * 0.9 if best_r2_value2 != 0 else 1, ylim[0], ylim[1])
    texts_ax3.append(ax3.annotate(f'{name2}\nBest Val\nEpoch {best_val_epoch2}\nR²: {best_r2_value2:.3f}',
                xy=(best_val_epoch2, best_r2_value2),
                xytext=(ann_x, ann_y),
                arrowprops=dict(arrowstyle='->', color=colors['r2']['train_2'], alpha=0.7),
                fontsize=10, color=colors['r2']['train_2'], ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8)))
    # ax4: Prior Loss
    if has_prior2:
        xlim = ax4.get_xlim()
        ylim = ax4.get_ylim()
        best_prior_loss2 = float(data2.loc[best_val_epoch2-1, 'prior_losses'])
        ann_x = clamp(best_val_epoch2 + 5, xlim[0], xlim[1])
        ann_y = clamp(best_prior_loss2 * 1.5 if best_prior_loss2 != 0 else 1, ylim[0], ylim[1])
        texts_ax4.append(ax4.annotate(f'{name2}\nBest Val\nEpoch {best_val_epoch2}\nLoss: {best_prior_loss2:.2e}',
                    xy=(best_val_epoch2, best_prior_loss2),
                    xytext=(ann_x, ann_y),
                    arrowprops=dict(arrowstyle='->', color=colors['prior']['train_2'], alpha=0.7),
                    fontsize=10, color=colors['prior']['train_2'], ha='left',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8)))
    # ax5: Sparsity
    if 'sparsity_proportion_nonzero' in data2.columns:
        xlim = ax5.get_xlim()
        ylim = ax5.get_ylim()
        sparsity2 = data2.loc[best_val_epoch2-1, 'sparsity_proportion_nonzero']
        ann_x = clamp(best_val_epoch2 + 5, xlim[0], xlim[1])
        ann_y = clamp(min(sparsity2 + 0.1, 1.0), ylim[0], ylim[1])
        texts_ax5.append(ax5.annotate(f'{name2}\nBest Val\nEpoch {best_val_epoch2}\nSparsity: {sparsity2:.3f} ({sparsity2*100:.1f}%)',
                    xy=(best_val_epoch2, sparsity2),
                    xytext=(ann_x, ann_y),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                    fontsize=10, color='gray', ha='left',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8)))

# Use adjustText to automatically adjust annotation positions per axes
if texts_ax1:
    adjust_text(texts_ax1, expand_text=(1.5, 2.0), expand_points=(1.5, 2.0), ax=ax1)
if texts_ax2:
    adjust_text(texts_ax2, expand_text=(1.5, 2.0), expand_points=(1.5, 2.0), ax=ax2)
if texts_ax3:
    adjust_text(texts_ax3, expand_text=(1.5, 2.0), expand_points=(1.5, 2.0), ax=ax3)
if texts_ax4:
    adjust_text(texts_ax4, expand_text=(1.5, 2.0), expand_points=(1.5, 2.0), ax=ax4)
if texts_ax5:
    adjust_text(texts_ax5, expand_text=(1.5, 2.0), expand_points=(1.5, 2.0), ax=ax5)

# Save the plot in the same directory as the first CSV
plot_dir = os.path.dirname(csv1)
plot_path = os.path.join(plot_dir, 'compare_loss.png')
plt.savefig(plot_path)
print(f"Plot saved to: {plot_path}")

# Print best validation epochs
if best_val_epoch1 is not None:
    print(f"{name1} - Best validation epoch: {best_val_epoch1}")
    print(f"  - Validation MSE: {data1.loc[best_val_epoch1-1, 'validation_loss']:.2e}")
    print(f"  - True Mean Loss: {data1.loc[best_val_epoch1-1, 'true_mean_losses']:.2e}")
    print(f"  - R² Value: {data1.loc[best_val_epoch1-1, 'r_squared_values']:.3f}")
    if has_prior1:
        print(f"  - Prior Loss: {data1.loc[best_val_epoch1-1, 'prior_losses']:.2e}")
if best_val_epoch2 is not None:
    print(f"{name2} - Best validation epoch: {best_val_epoch2}")
    print(f"  - Validation MSE: {data2.loc[best_val_epoch2-1, 'validation_loss']:.2e}")
    print(f"  - True Mean Loss: {data2.loc[best_val_epoch2-1, 'true_mean_losses']:.2e}")
    print(f"  - R² Value: {data2.loc[best_val_epoch2-1, 'r_squared_values']:.3f}")
    if has_prior2:
        print(f"  - Prior Loss: {data2.loc[best_val_epoch2-1, 'prior_losses']:.2e}") 