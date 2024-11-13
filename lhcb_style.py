import matplotlib.pyplot as plt

def apply_lhcb_style():
    # Define the LHCb style for matplotlib rcParams
    lhcb_style = {
        # Use a similar font family and size
        'font.family': 'serif',
        'font.sans-serif': ['Times New Roman'],
        'font.size': 18,  # Approximate size matching ROOT's lhcbTSize (0.06)

        # Set line and marker style
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'lines.marker': 'o',

        # Set figure and axis background colors to white
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        
        # Set axis titles and labels font properties
        'axes.titlesize': 2,  # Title size in points
        'axes.labelsize': 200, # Label size in points
        'axes.titleweight': 'bold',

        # Tick settings, to include ticks on all sides
        'xtick.top': True,
        'ytick.right': True,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 5,
        'ytick.major.size': 5,

        # Configure legend
        'legend.frameon': False,  # Remove legend border
        'legend.fontsize': 14,
        'legend.loc': 'best',

        # Adjust padding to reflect ROOT style margins
        'axes.titlepad': 10,
        'axes.labelpad': 8,

        # Grid style similar to ROOT
        'grid.color': 'gray',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        
        # Configure histogram/axes divisions
        'xtick.major.pad': 8,  # Padding for x-axis major ticks
        'ytick.major.pad': 8,  # Padding for y-axis major ticks
        'axes.grid': True,
        'axes.linewidth': 2,  # Set axis line thickness
    }

    # Apply the LHCb style to matplotlib
    plt.rcParams.update(lhcb_style)
    print("LHCb style applied to matplotlib.")

