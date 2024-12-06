import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import json
from scipy.optimize import curve_fit
from lhcb_style import apply_lhcb_style
from matplotlib.colors import LogNorm
import mplhep
pd.set_option('display.max_rows', None)
mplhep.style.use(mplhep.style.LHCb2)

def gaussian_with_offset(x, amp, mean, stddev, offset):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) + offset

def gaus(x, A, mu, sigma):
    return A / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu)**2 / (2 * sigma**2)) 



# Load data from CSV file
data_path = "C:/Users/lexda/local_pmt_info/characterisation/process_data_for_conference/Gain_data_channel_5_8_0.01_1420v.csv"
data = pd.read_csv(data_path, delimiter=',',  names=['Position','data','channle'], skiprows=1)

# Load the Gaussian fit parameters from a JSON file
json_path = "C:/Users/lexda/local_pmt_info/characterisation/process_data_for_conference/fit_parameters_Gain_data_channel_5_8_0.01_1420v.json"
with open(json_path, 'r') as f:
    gaussian_params = json.load(f)


for i in range(5, 9):
    field_data = data[data['channle'] == f"F{i}"]
    positions = field_data['Position']
    y_data = field_data['data']
    plt.scatter(positions, y_data, s=10, label=f'channle {i}')

    params = gaussian_params[f"F{i}"]
    amp, mean, stddev = params

    x_values = np.linspace(positions.min(), positions.max(), 1000)
    y_values = gaus(x_values, amp, mean, stddev)

    # Plot the Gaussian curve
    plt.plot(x_values, y_values)



plt.xlabel('Position')
plt.ylabel('average min pluse height')
plt.title('')
plt.legend()

plt.figure()
hist_1 = pd.read_csv("C:/Users/lexda/local_pmt_info/characterisation/process_data_for_conference/voltage_mins/C2-waveform-200-avg.csv", delimiter=',', names=['bins', 'count'], skiprows=6)
print (hist_1)

# Plot the shifted data
plt.plot(hist_1['bins']*1e9, hist_1['count']*1e3, label='wavefrom', color='red')

# Add labels and legend
plt.xlabel("time [ns]" )
plt.ylabel("voltage [mV]")
plt.legend()
#plt.show()


# process gain and counts data for plotting
# Load data from CSV file
data_path = "C:/Users/lexda/local_pmt_info/characterisation/gain_results/gain_data_1460_v_mcp_0.1_step_90_int.csv"
gain_data = pd.read_csv(data_path, delimiter=',',  names=['Position','gain','channle'], skiprows=1)

data_path = "C:/Users/lexda/PycharmProjects/Photek_lab/residuals_sums_data.csv"
counts_data = pd.read_csv(data_path, delimiter=',',  names=['Position','counts','channle'], skiprows=1)
merged_data = pd.merge(gain_data, counts_data, on='Position', suffixes=('_gain', '_counts'))

high_residuals_mask = merged_data["counts"] > 100 


high_residuals_positions_merged = merged_data[high_residuals_mask]

print (high_residuals_positions_merged)  
counts = high_residuals_positions_merged['counts']

print(counts)
counts_greater_than_300 = counts[counts > 100]
print("Counts greater than 300:")
print(counts_greater_than_300)
gain = high_residuals_positions_merged['gain']
position = high_residuals_positions_merged['Position']

plt.figure()
plt.hist2d(position,gain, bins=[len(position), 50], weights=counts, cmap='viridis', norm=LogNorm(vmin=350, vmax=414))  
plt.colorbar(label='Counts')
plt.xlabel('Position')
plt.ylabel('Gain')
plt.show()








# Show plot
#plt.show()