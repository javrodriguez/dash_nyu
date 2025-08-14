import matplotlib.pyplot as plt

def save_figure(fig, filename, dpi=300):
    """Save a matplotlib figure to a file"""
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig) 