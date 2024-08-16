import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
from joblib import Parallel, delayed


def read_data_hist(filename):
    data = pd.read_csv(filename, comment='#', names=["time", "hits"], skiprows=5, dtype={"time": float, "hits": float})
    return data['time'], data['hits']

def process_file(file_name,i):
    time, voltage = read_data_hist(f"{file_name}--{i:05}.txt")
    threshold = 100
    # Invert the voltage to find dips as peaks
    voltage = -voltage
    # Find the first significant peak (dip in original data)
    peaks, _ = find_peaks(voltage, height=0.0035, distance=350, prominence=0.001)

    peak_heights_dict = {i: height for i, height in enumerate( voltage[peaks]) if height < threshold}
    peak_times = time[peaks].astype(float).tolist()
    peak_times = [time[i].astype(float) for i in peaks if voltage[i] < threshold]
    return peak_times, len(peak_times), peak_heights_dict

def looking_at_peaks(file_name,j,k):
    waveform_voltages = []
    wavefrom_times = []
    for i in range(j, k):
        time, voltage = read_data_hist(f"{file_name}--{i:05}.txt")

        voltage = -voltage
        # Find the first significant peak (dip in original data)
        peaks, _ = find_peaks(voltage, height=0.0035, distance=220, prominence=0.001)
        peak_time = time[peaks]
        peak_positions = voltage[peaks]


        plt.plot(time, voltage)
        plt.plot(peak_time, peak_positions, 'x')
        wavefrom_times.append(time)
        waveform_voltages.append(-voltage)
    plt.show()
    return waveform_voltages, wavefrom_times
#file_name = "Ion_feedback_data/waveform/ion--waveform__back_up/C4--alex gas--700v-mcp--waveform"
file_name = "Ion_feedback_data/waveform_10k/ion-10k-wave-form/C4--alex gas--700v-mcp-10k--waveform"
#file_name = "Ion_feedback_data/10k-min-laser-bi/10k-min-laser-bi/C4--alex gas--700v-mcp-10k-min-laser--waveform--binary"
plt.figure(figsize=(20, 8))
waveform_voltages, waveform_times = looking_at_peaks(file_name, 0, 10000)
waveform_times = np.array(waveform_times).flatten()
waveform_voltages = np.array(waveform_voltages).flatten()
norm = LogNorm(vmin=1, vmax=500)
# Create a 2D histogram of the data
plt.hist2d(waveform_times*1e9, waveform_voltages, bins=[250,100],norm=norm, cmap='viridis')
plt.ylim(-0.015, 0.005)
plt.xlim(0, 200)
plt.colorbar(label='Count')
plt.xlabel('Time [ns]')
plt.ylabel('Voltage')
plt.title('Persistence Plot')
plt.figure()

results = Parallel(n_jobs=-1)(delayed(process_file)(file_name, i) for i in range(0, 10000))

hist_peaks = [item for sublist in [result[0] for result in results] for item in sublist]
num_peaks = [result[1] for result in results]
peak_heights_dict = [result[2] for result in results]


first_peak = []
ion_peaks = []
for i in range(len(hist_peaks)):
    if hist_peaks[i] > 0.2e-7:
        ion_peaks.append(hist_peaks[i])
    else:
        first_peak.append(hist_peaks[i])

first_peak = np.array(first_peak)
ion_peaks = np.array(ion_peaks)

plt.figure(figsize=(10, 6))
mean_first_peak = np.mean(first_peak)
hist_vals, bin_edges, _ = plt.hist((ion_peaks-mean_first_peak)*1e9, bins=300)
hist_peaks, _ = find_peaks(hist_vals, height=25, distance=10)
peak_positions = bin_edges[hist_peaks]
peak_heights = hist_vals[hist_peaks]
plt.plot(peak_positions, peak_heights, 'x')
plt.hist((first_peak-mean_first_peak)*1e9, bins=300)
print(f"ion_peaks {peak_positions} ns")
plt.xlabel('Time (ns)')
plt.ylabel('Counts')
plt.title('Events vs. Time')

plt.figure(figsize=(10, 6))
#ions = [0.5, 1, 2, 4, 8, 18, 37]
#plt.plot(ions,peak_positions)
plt.figure()
bin_edges = np.arange(0, np.max(num_peaks)+1, 0.5)

plt.hist(num_peaks, bins=bin_edges, align ="left", alpha=0.5, label='First Detected Dip in Voltage Signal')
#bin_centers = bin_edges[:-1]+0.5
plt.xlabel('number of peaks')
plt.ylabel('Frequency')
plt.title('Histogram of Number of Peaks')
plt.yscale('log')
value = 2

# Use list comprehension to create a new list with numbers greater than the value
greater_numbers = [num for num in num_peaks if num > value]

no_peaks = [num for num in num_peaks if num == 0]
no_ions = [num for num in num_peaks if num == 1]
ions = [num for num in num_peaks if num > 1]
no_peak_size = len(no_peaks)
no_ion_size = len(no_ions)
ion_size = len(ions)

# The size (count) of numbers greater than the value is the length of the new list
size = len(greater_numbers)
print (f"number of no peaks {no_peak_size}, number of no ions {no_ion_size}, ions {ion_size}, number of greater numbers {size} "
       f"therefor number mutiple peaks in reluation to events ....")
#plt.xticks(bin_centers)
plt.figure(figsize=(10, 6))
for i in range(np.max(num_peaks)):
    zero_key_values = [d[i] for d in peak_heights_dict if i in d]
    plt.hist(zero_key_values, bins=100, alpha=0.5, label=f'{i} peaks')
    plt.xlabel('peak height')
    plt.ylabel('Counts')
    plt.xlim(0.001, 0.015)
plt.legend()
plt.show()

