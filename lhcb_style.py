import matplotlib.pyplot as plt

def apply_lhcb_style():
    custom_style = {
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 18,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'axes.linewidth': 2,
        'axes.grid': True,
        'grid.linewidth': 2,
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'xtick.major.size': 10,
        'xtick.major.width': 2,
        'ytick.major.size': 10,
        'ytick.major.width': 2,
        'xtick.minor.size': 5,
        'xtick.minor.width': 1,
        'ytick.minor.size': 5,
        'ytick.minor.width': 1,
        'legend.fontsize': 16,
        'legend.frameon': False,
        'legend.borderaxespad': 0,
        'legend.loc': 'best',
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'figure.figsize': (10, 7),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.edgecolor': 'black',
        'axes.facecolor': 'white',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'text.usetex': False,
    }

    plt.rcParams.update(custom_style)

# Apply the custom style


