import lecroyparser
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from joblib import Parallel, delayed
import numpy as np
from scipy.ndimage import gaussian_filter1d

def read_data_hist(filename):
    data = lecroyparser.ScopeData(filename)
    return data.x, data.y


def process_file(file_name, i):
    time, voltage = read_data_hist(f"{file_name}--{i:05}.trc")
    threshold = 0.014
    # Invert the voltage to find dips as peaks
    voltage = gaussian_filter1d(voltage, sigma=20)
    voltage = -voltage
    # Find the first significant peak (dip in original data)
    peaks, _ = find_peaks(voltage, height=0.0017, distance=350, prominence=0.001)

    peak_heights_dict = {i: height for i, height in enumerate(voltage[peaks]) if height < threshold}
    peak_times = [time[i].astype(float) for i in peaks if voltage[i] < threshold]
    return peak_times, len(peak_times), peak_heights_dict


def looking_at_peaks(file_name, j, k):
    waveform_voltages = []
    wavefrom_times = []
    plt.figure()
    for i in range(j, k):
        time, voltage = read_data_hist(f"{file_name}--{i:05}.trc")

        voltage = -voltage
        voltage = gaussian_filter1d(voltage, sigma=20)
        # Find the first significant peak (dip in original data)
        peaks, _ = find_peaks(voltage, height=0.0017, distance=220, prominence=0.001)
        peak_time = time[peaks]
        peak_positions = voltage[peaks]


        plt.plot(time, voltage)
        plt.plot(peak_time, peak_positions, 'x')
        wavefrom_times.append(time)
        waveform_voltages.append(-voltage)
        plt.xlim(0, 0.6e-7)
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.title('Waveforms with Peaks')
    plt.savefig('binary_plot.png')


file_name = "/mnt/c/Users/lexda/PycharmProjects/Photek_lab/Ion_feedback/Ion_feedback_data/wave-form-100k/C4--alex gas--700v-mcp-100k--waveform--binary"

results = Parallel(n_jobs=-1)(delayed(process_file)(file_name, i) for i in range(0, 10000))
looking_at_peaks(file_name, 0, 100)

hist_peaks = [item for sublist in [result[0] for result in results] for item in sublist]
num_peaks = [result[1] for result in results]
peak_heights_dict = [result[2] for result in results]

first_peak = []
ion_peaks = []
for i in range(len(hist_peaks)):
    if hist_peaks[i] > 0.12e-7:
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
plt.yscale('log')
plt.savefig('binary_hist_plot.png')