import functools
from pathlib import Path
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import json
from scipy.optimize import curve_fit
from scipy.special import erf
import mplhep
import mpl_toolkits.mplot3d as plt3d
#mplhep.style.use(mplhep.style.LHCb2)


# Define the Gaussian function
def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def gaussian_convolved_tophat(x, amp, mean, stddev, width):
    return amp * (erf((x - mean + width / 2) / (np.sqrt(2) * stddev)) - erf((x - mean - width / 2) / (np.sqrt(2) * stddev))) / 2

def gaussian_convolved_tophat_fixed_width(x, amp, mean, stddev):
    width = 0.55
    return gaussian_convolved_tophat(x, amp, mean, stddev, width)


def load_data(fname):
    _, pulse_min = np.loadtxt(fname, delimiter=",", unpack=True)
    return pulse_min


def extract_xy_from_filename(filename):
    """
    Extract the x and y positions from filenames with pattern like 
    'histogram_F5_80.97x_87.95y.txt'.
    Returns x and y as floats.
    """
    match = re.search(r'_(\d+\.\d+)x_(\d+\.\d+)y\.txt', filename)
    if match:
        x, y = map(float, match.groups())
        return x, y
    return None, None


# Define the data directory
path_string = "C:/Users/lexda/local_pmt_info/characterisation/laser_sweeps/2d_sweep/89_hour_2D_week_end_scan"
DATA_DIR = Path(path_string)
directory = DATA_DIR / 'hist'

# Create directory and its parents if they don't exist
os.makedirs(directory, exist_ok=True)

# Initialize variables
positions_x = []
positions_y = []
residuals_sums = []
gaussian_fits = {}
fields = []

# Loop over fields
for field in range(5, 9):
    files_with_positions = []
    for filepath in directory.glob(f'histogram_F{field}_*.txt'):
        x, y = extract_xy_from_filename(filepath.name)
        if x is not None and y is not None:
            files_with_positions.append(((x, y), filepath))
    files_with_positions.sort()  # Ascending order by x, y

    print(f"Found {len(files_with_positions)} files for field F{field}")

    for (x, y), filepath in files_with_positions:
        print(f"Processing file {filepath} for position x: {x}, y: {y}")
        hist_data = pd.read_csv(filepath, delimiter=',', names=['bins', 'counts'], skiprows=1)

        x_data = np.asarray(hist_data['bins']) * 1e12  # Convert to picoseconds
        y_data = np.asarray(hist_data['counts'])

        max_y_index = np.argmax(y_data)
        peak_x_value = x_data[max_y_index]

        # Shift x_data so that the peak is at 0
        x_data = x_data - peak_x_value

        # Gaussian fitting
        popt, _ = curve_fit(gaussian, x_data, y_data, p0=[200, 0., 0.1])
        amp, mean, stddev = popt

        fitted_gaussian = gaussian(x_data, amp, mean, stddev)
        num_stddevs = 2 # Define the range, e.g., 2 standard deviations around the mean
        lower_bound = mean - num_stddevs * abs(stddev)
        
        #lower_bound =  -5.40

        # Create a mask for the tails: True where bins are outside the central region
        tails_mask = (x_data < lower_bound) 

        # Compute residuals and other statistics
        residuals = y_data - fitted_gaussian
        residuals_sum = np.sum(residuals[tails_mask])
        

        # Append results
        positions_x.append(x)
        positions_y.append(y)
        residuals_sums.append(residuals_sum)
        fields.append(f'F{field}')

        gaussian_fits[f'F{field}_{x}_{y}'] = {
            'amplitude': amp,
            'mean': mean,
            'stddev': stddev
        }


# Save results to CSV
residuals_df = pd.DataFrame({
    'x_position': positions_x,
    'y_position': positions_y,
    'residuals_sum': residuals_sums,
    'field': fields
})

residuals_df.to_csv('residuals_sums_data_with_xy.csv', index=False)

# Save Gaussian fits to JSON
with open('gaussian_fits_with_xy.json', 'w') as f:
    json.dump(gaussian_fits, f, indent=4)

# Plot residuals
combined_x_positions = []
combined_y_positions = []
combined_residuals_sums = []
plt.figure(figsize=(14, 8))
fig = plt.figure(figsize=(10, 6))
for field in range(7, 8):
    # Select data corresponding to this field
    field_mask = np.array([f == f'F{field}' for f in fields])
    selected_x_positions = np.array(positions_x)[field_mask]
    selected_y_positions = np.array(positions_y)[field_mask]
    selected_residuals_sums = np.array(residuals_sums)[field_mask]
    

    combined_x_positions.extend(np.array(positions_x)[field_mask])
    combined_y_positions.extend(np.array(positions_y)[field_mask])
    combined_residuals_sums.extend(np.array(residuals_sums)[field_mask])


    x_step_size = 0.2
    y_step_size = 0.2
    x_bins = np.arange(selected_x_positions.min(), selected_x_positions.max() + x_step_size, x_step_size)
    y_bins = np.arange(selected_y_positions.min(), selected_y_positions.max() + y_step_size, y_step_size)
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2
    x,y = np.meshgrid(x_centers, y_centers)
    H, xedges, yedges = np.histogram2d(selected_x_positions, selected_y_positions, bins=(x_bins, y_bins), weights=selected_residuals_sums)
    
    ax = fig.add_subplot(111, projection='3d')
    X,Y = np.meshgrid(x_centers, y_centers)
    ax.plot_surface(X, Y, H.T, cmap='viridis')
    
    #plt.scatter(selected_x_positions, selected_residuals_sums, label=f'Channel {field - 4}')
    


    

plt.xlabel('X Position')
plt.ylabel('Y Position')

plt.title('Sum of Residuals vs Position in X and Y')


combined_x_positions = np.array(combined_x_positions)
combined_y_positions = np.array(combined_y_positions)
combined_residuals_sums = np.array(combined_residuals_sums)

#plt.figure(figsize=(10, 6))
# plt.hist2d(
#     combined_x_positions,
#     combined_y_positions,
#     bins=[
#         np.arange(combined_x_positions.min(), combined_x_positions.max() + x_step_size, x_step_size),
#         np.arange(combined_y_positions.min(), combined_y_positions.max() + y_step_size, y_step_size)
#     ],
#     weights=combined_residuals_sums,
#     cmap='viridis',
   
# )


plt.legend()
plt.show()
