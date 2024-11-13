
import functools
from pathlib import Path
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import chirp, find_peaks, peak_widths, argrelextrema 
from scipy.special import erf




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


def extract_numbers_from_filename(filename):
    numbers = re.findall(r'-?\d+\.\d+|-?\d+', filename)
    return [float(num) for num in numbers]


path_string = "C:/Users/lexda/local_pmt_info/characterisation/laser_sweeps/600v_cathode_1420mcp_0.01_steps_over_3mm_180_int"

DATA_DIR = Path(path_string)
start_postion = extract_numbers_from_filename(path_string)



directory = DATA_DIR / 'hist'

# Create directory and its parents if they don't exist
os.makedirs(directory, exist_ok=True)

positions = []
residuals_sums = []
fields = []
mean = []


for field in range(5, 6):
    for position in np.arange(81.25, 81.26): #np.arange(81.25, 81.25, 0.01):  # Step size of 0.1
        # Format the position to one decimal place for the filename
        position_str = f'{position:.2f}'
        filename = DATA_DIR / f'hist/histogram_F{field}_{position_str}.txt'
        hist_data = pd.read_csv(filename, delimiter=',', names=['bins', 'counts'], skiprows=1)
        
        x_data = np.asarray(hist_data['bins']) * 1e12  # Convert to picoseconds
        y_data = np.asarray(hist_data['counts'])
       
        popt, _ = curve_fit(gaussian, x_data, y_data)

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
        #lower_bound =  -2.40

        # Create a mask for the tails: True where bins are outside the central region
        tails_mask = (hist_data['bins']*1e12 < lower_bound) #| (hist_data['bins']*1e12 > upper_bound)
        x_values_for_gaussian = np.linspace(lower_bound, upper_bound, 10000)
        fitted_gaussian_10000 = gaussian(x_values_for_gaussian, amp, mean, stddev)
        numerical_area = np.trapz(fitted_gaussian_10000, x=x_values_for_gaussian)  # Trapezoidal rule

        print("Area under the fitted Gaussian_1000 (numerical integration):", numerical_area)
        # Plot the original histogram
        plt.plot(hist_data['bins']*1e12, hist_data['counts'], label=' oscilloscopes Area Data')
        plt.plot(x_values_for_gaussian, fitted_gaussian_10000, label='Fitted Gaussian', linestyle='--')
        print()
        # Plot the residuals only at the edges (tails)
        plt.plot(hist_data['bins'][tails_mask]*1e12, residuals[tails_mask], 
                label='Residuals (Tails)', linestyle=':', color='red', linewidth=2)

        residuals_sum = np.sum(residuals[tails_mask])
        
        mu = -np.log(numerical_area/total_sum)
        print("mu=",mu  )
    
        

        # Set the plot labels and limits
        #plt.xlim(-3.5, 21)
        plt.xlabel('pwb [Vs]')
        plt.ylabel('counts')
        plt.yscale('log')
        plt.ylim(1, 800e3)
        plt.title(f'F{field} at {position:.2f} mm')
        plt.legend()
        positions.append(position)
        residuals_sums.append(residuals_sum)
        fields.append(f'F{field}')

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
    plt.plot(selected_positions, selected_residuals_sums, marker='o', linestyle='-', label=f'F{field}')

    high_residuals_mask = selected_residuals_sums > 120
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
plt.ylabel('Sum of Residuals in Tails')
plt.title('Sum of Residuals vs Position for F5 to F8')



plt.legend()
plt.show()