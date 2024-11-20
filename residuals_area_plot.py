
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
from scipy.signal import chirp, find_peaks, peak_widths, argrelextrema 
from scipy.special import erf
import mplhep
mplhep.style.use(mplhep.style.LHCb2)


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

def parse_xy(f):
    return tuple((float(k) for k in f.stem.split("_")[1:]))

def extract_position_from_filename(filename):
    # Extract position from filename, e.g., 'histogram_F<field>_<position>.txt'
    match = re.search(r'_(\d+\.\d+)\.txt', filename)
    return float(match.group(1)) if match else None


path_string = "C:/Users/lexda/Desktop/Crosstalk_data_zip/week_28_07_23/measurement_93_95_0.1_step_10000s/"

DATA_DIR = Path(path_string)
directory = DATA_DIR / 'hist'

# Create directory and its parents if they don't exist
os.makedirs(directory, exist_ok=True)

positions = []
residuals_sums = []
gaussian_fits = {}
fields = []
mean = []


for field in range(5, 9):
    files_with_positions = []
    for filepath in directory.glob(f'histogram_F{field}_*.txt'):
        position = extract_position_from_filename(filepath.name)
        if position is not None:
            files_with_positions.append((position, filepath))
    files_with_positions.sort()  # Ascending order by position

    print(f"Found {len(files_with_positions)} files for field F{field}")

    for position, filepath in files_with_positions:
      # Step size of 0.1
        # Format the position to one decimal place for the filename
        position_str = f'{position:.2f}'
        hist_data = pd.read_csv(filepath, delimiter=',', names=['bins', 'counts'], skiprows=1)
        
        
        x_data = np.asarray(hist_data['bins']) * 1e12  # Convert to picoseconds
        y_data = np.asarray(hist_data['counts'])
        
        max_y_index = np.argmax(y_data)
        peak_x_value = x_data[max_y_index]

        # Shift x_data so that the peak is at 0
        x_data = x_data - peak_x_value

        popt, _ = curve_fit(gaussian, x_data, y_data ,p0=[200, 0., 0.1])

        # Extract fitted parameters (amplitude, mean, standard deviation)
        amp, mean, stddev = popt

        # Compute the fitted Gaussian values at the bin positions
        fitted_gaussian = gaussian(hist_data['bins']*1e12, *popt)
        total_area = amp * stddev * np.sqrt(2 * np.pi)
        total_sum = np.sum(hist_data['counts'])
        print("total sum", total_sum)
        print("amp:", amp)
        print(stddev)
        print("Total area under the Gaussian (analytical):", total_area)

        fitted_gaussian = gaussian(x_data, amp, mean, stddev)
        numerical_area = np.trapz(fitted_gaussian, x=x_data)  # Trapezoidal rule

        print("Area under the fitted Gaussian (numerical integration):", numerical_area)

        # Subtract the fit from the original data to get the residuals
        residuals = hist_data['counts'] - fitted_gaussian

        # Define the region around the mean that will be considered "central"
        
        num_stddevs = 5 # Define the range, e.g., 2 standard deviations around the mean
        lower_bound = mean - num_stddevs * abs(stddev)
        upper_bound = mean + num_stddevs * abs(stddev)
        #lower_bound =  -5.40

        # Create a mask for the tails: True where bins are outside the central region
        tails_mask = (x_data < lower_bound) #| (hist_data['bins']*1e12 > upper_bound)
        x_values_for_gaussian = np.linspace(lower_bound, upper_bound, 10000)
        fitted_gaussian_10000 = gaussian(x_values_for_gaussian, amp, mean, stddev)
        


        print("Area under the fitted Gaussian_1000 (numerical integration):", numerical_area)
        

        plt.plot(x_data, hist_data['counts'], label='Histogram Data')
       

        plt.plot(x_values_for_gaussian, fitted_gaussian_10000, label='Fitted Gaussian', linestyle='--')
        print()
        # Plot the residuals only at the edges (tails)
        #plt.plot(x_data[tails_mask], residuals[tails_mask], 
        #        label='Residuals (Tails)', linestyle=':', color='red')

        residuals_sum = np.sum(residuals[tails_mask])
        
        mu = -np.log(numerical_area/total_sum)
        print("mu=",mu  )

        gaussian_fits[f'F{field}_{position_str}'] = {
            'amplitude': amp,
            'mean': mean,
            'stddev': stddev
        }
    
        

        # Set the plot labels and limits
        #plt.xlim(-3.5, 21)
        plt.xlabel('pwb [Vs]')
        plt.ylabel('counts')
        plt.yscale('log')
        plt.ylim(1, 10e3)
        positions.append(position)
        residuals_sums.append(residuals_sum)
        fields.append(f'F{field}')
        #plt.legend()

residuals_df = pd.DataFrame({
    'position': positions,
    'residuals_sum': residuals_sums,
    'field': fields
    
})
residuals_df.to_csv('residuals_sums_data.csv', index=False)

with open('gaussian_fits.json', 'w') as f:
    json.dump(gaussian_fits, f, indent=4)


plt.figure(figsize=(14, 8))

initial_guesses = {
    'F5': [2000, 81.2, 0.85], 
    'F6': [1500, 81.6, 0.85],
    'F7': [1500, 82.0, 0.85],
    'F8': [2000, 82.5, 0.85]
}

for field in range(5, 9):
    # Select data corresponding to this field
    field_mask = np.array([f == f'F{field}' for f in fields])
    selected_positions = np.array(positions)[field_mask]
    selected_residuals_sums = np.array(residuals_sums)[field_mask]
    plt.scatter(selected_positions, selected_residuals_sums, marker='o', s=10, linestyle='-', label=f'channle {field-4}')

    high_residuals_mask = selected_residuals_sums > 100
    high_residuals_positions = selected_positions[high_residuals_mask]

    # Print the positions
    print(f"Positions with residual sums above 300 for field F{field}: {high_residuals_positions}")


    p0 = initial_guesses[f'F{field}']

    #popt, _ = curve_fit(gaussian_convolved_tophat_fixed_width, selected_positions, selected_residuals_sums, p0=p0)
    #amp, mean, stddev = popt
    # Plot the fitted Gaussian curve
    #x_fit = np.linspace(selected_positions.min(), selected_positions.max(), 1000)
    #y_fit = gaussian_convolved_tophat_fixed_width(x_fit, *popt)
    #plt.plot(x_fit, y_fit, linestyle='--', label=f'Fitted Gaussian F{field}')
    fwhm_gaussian = 2.355 * stddev
    print(f"FWHM of the Gaussian for field F{field}: {fwhm_gaussian:.2f}")
    #print(f"Width of the tophat for field F{field}: {0.55:.2f}")
    



# Label and title for the final plot
#plt.xlim(58.95, 59.95)
#plt.vlines(77.85, 0, 1000, linestyles ="dashed", colors ="k")
#plt.vlines(77.30, 0, 1000, linestyles ="dashed", colors ="k")
plt.xlabel('Position')
plt.ylabel('Sum of Residuals')
plt.title('Sum of Residuals vs Position for different channles')



plt.legend()
plt.show()