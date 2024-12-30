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
#import mplhep
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
target_x_positions = [83.17]
target_y_positions = [91.15]
colors = ['red', 'blue', 'green', 'purple']

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

        if x in target_x_positions and y in target_y_positions:
            
            bin_centers = x_data
            heights = y_data


            bin_width = bin_centers[1] - bin_centers[0]  # Assume uniform bin width
            bin_edges = np.concatenate([[bin_centers[0] - bin_width / 2],
                                        bin_centers + bin_width / 2])

            # Step 3: Create data to mimic histogram input
            # For each bin, repeat the bin center value 'height' times
            hist_data = np.repeat(bin_centers, heights.astype(int))

            plt.hist(hist_data, bins=200, color=colors[field - 5], histtype='step', linewidth=1, label=f'Field {field} at x: {x}, y: {y}')

            #plt.plot(x_data, fitted_gaussian, label='Fit')
            plt.xlabel('[pWb]')
            plt.ylabel('Counts')
            plt.yscale('log')
            plt.ylim(1, 1e4)
            plt.xlim(-20, 1.5)
            plt.title(f'Field {field} at x: {x}, y: {y}')
            plt.legend()
            
        

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


# Define a tolerance for floating point comparison (optional)
tolerance = 0.01  # Adjust based on your precision needs

# Method 1: Using isin for exact matches
filtered_df_exact = residuals_df[
    residuals_df['x_position'].isin(target_x_positions) &
    residuals_df['y_position'].isin(target_y_positions)
]

print(f"{filtered_df_exact} output")


# Save Gaussian fits to JSON
with open('gaussian_fits_with_xy.json', 'w') as f:
    json.dump(gaussian_fits, f, indent=4)

# Plot residuals
combined_x_positions = []
combined_y_positions = []
combined_residuals_sums = []
plt.figure(figsize=(14, 8))
fig = plt.figure(figsize=(10, 6))
for field in range(5, 9):
    # Select data corresponding to this field
    field_mask = np.array([f == f'F{field}' for f in fields])
    selected_x_positions = np.array(positions_x)[field_mask]
    selected_y_positions = np.array(positions_y)[field_mask]
    selected_residuals_sums = np.array(residuals_sums)[field_mask]
    

    combined_x_positions.extend(np.array(positions_x)[field_mask])
    combined_y_positions.extend(np.array(positions_y)[field_mask])
    combined_residuals_sums.extend(np.array(residuals_sums)[field_mask])
    selected_x_positions = np.where(selected_y_positions > 91.15, selected_x_positions + 0.55, selected_x_positions)


    # x_step_size = 0.2
    # y_step_size = 0.2
    # x_bins = np.arange(selected_x_positions.min(), selected_x_positions.max() + x_step_size, x_step_size)
    # y_bins = np.arange(selected_y_positions.min(), selected_y_positions.max() + y_step_size, y_step_size)
    # x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    # y_centers = (y_bins[:-1] + y_bins[1:]) / 2
    # x,y = np.meshgrid(x_centers, y_centers)
    # H, xedges, yedges = np.histogram2d(selected_x_positions, selected_y_positions, bins=(x_bins, y_bins), weights=selected_residuals_sums)
    
    # ax = fig.add_subplot(111, projection='3d')
    # X,Y = np.meshgrid(x_centers, y_centers)
    # ax.plot_surface(X, Y, H.T, cmap='viridis')
    
    plt.scatter(selected_x_positions, selected_residuals_sums, label=f'Channel {field - 4}')
    

plt.xlabel('X Position [mm]')
plt.ylabel('Y Position [mm]')

plt.title('Sum of Residuals vs Position in X and Y')


combined_x_positions = np.array(combined_x_positions)
combined_y_positions = np.array(combined_y_positions)
combined_residuals_sums = np.array(combined_residuals_sums)



data_df = pd.read_csv(r"residuals_sums_data_with_xy.csv", delimiter=',', names=['x_position','y_position','residuals_sum','field'], skiprows=1)


print(data_df.head())

#plt.colorbar()




x_center =  83.17
filtered_data = data_df[data_df['y_position'] == x_center]
plt.figure(figsize=(14, 8))
plt.scatter(filtered_data['x_position'], filtered_data['residuals_sum'], s=10)
print(data_df.head())

plt.legend()
plt.show()
