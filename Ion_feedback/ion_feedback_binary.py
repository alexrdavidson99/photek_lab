import lecroyparser
import matplotlib.pyplot as plt
#import pandas as pd
from scipy.signal import find_peaks
from joblib import Parallel, delayed
import numpy as np
import os
#from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import LogNorm




def read_data_hist(filename):
    data = lecroyparser.ScopeData(filename)
    return data.x, data.y

def process_file(file_name):
    time, voltage = read_data_hist(file_name)
    threshold = 100
    # Invert the voltage to find dips as peaks
    #voltage = gaussian_filter1d(voltage, sigma=20)
    voltage = -voltage
    # Find the first significant peak (dip in original data)
    peaks, _ = find_peaks(voltage, height=0.008,distance=50, prominence=0.008)
    peak_heights = voltage[peaks]

    peak_heights_dict = {i: height for i, height in enumerate(voltage[peaks]) if height < threshold}
    peak_times = [time[i].astype(float) for i in peaks if voltage[i] < threshold]

    return peak_times, len(peak_times), peak_heights_dict


def looking_at_peaks(file_names, i,j):
    waveform_voltages = []
    waveform_times = []
    plt.figure()



    for file_name in file_names[i:j]:
        time, voltage = read_data_hist(file_name)


        voltage = -voltage
        #voltage = gaussian_filter1d(voltage, sigma=20)
        # Find the first significant peak (dip in original data)
        peaks, _ = find_peaks(voltage, height=0.008,distance=50, prominence=0.008)
        peak_time = time[peaks]
        peak_positions = voltage[peaks]
        print (peak_time)
        peak_times = [time[i].astype(float) for i in peaks]

        if len(peak_times) > 1 : #< 2e-9:
            #plt.plot((time-peak_times[0])*1e9, voltage)
            waveform_voltages.append(voltage)
            waveform_times.append((time-peak_times[0])*1e9)
            #plt.plot((peak_time-peak_times[0])*1e9, peak_positions, 'x')
        if len(peak_times) > 5:
            plt.plot((time) * 1e9, voltage)
            plt.plot((peak_time) * 1e9, peak_positions, 'x')
        #plt.xlim(-15, 50)

    plt.xlabel('Time [ns]')
    plt.ylabel('Voltage')
    plt.title('Waveforms with Peaks')
    plt.savefig('binary_plot.png')
    plt.figure()
    waveform_times = np.array(waveform_times).flatten()
    waveform_voltages = np.array(waveform_voltages).flatten()
    print (f"waveform times {waveform_times}")
    norm = LogNorm(vmin=1, vmax=500)
    # Create a 2D histogram of the data
    plt.hist2d(waveform_times, -waveform_voltages, bins=[500, 100], norm=norm, cmap='viridis')
    plt.colorbar(label='Count')
    plt.xlabel('Time [ns]')
    plt.ylabel('Voltage')
    plt.title('Persistence Plot')
    plt.figure()


#file_name = "C:/Users/lexda/PycharmProjects/Photek_lab/Ion_feedback/Ion_feedback_data/wave-form-100k/"
#file_name = "C:/Users/lexda/PycharmProjects/Photek_lab/Ion_feedback/Ion_feedback_data/13150210-min-10-bi-ml/"
#file_name = "C:/Users/lexda/PycharmProjects/Photek_lab/Ion_feedback/Ion_feedback_data/10k-just-ion-13150210-10ns"
#file_name = "C:/Users/lexda/PycharmProjects/Photek_lab/Ion_feedback/Ion_feedback_data/13150210-2.47-1.67-0.93-10k-10ns" # height=0.0008,distance=50, prominence=0.0008
#file_name = "C:/Users/lexda/PycharmProjects/Photek_lab/Ion_feedback/Ion_feedback_data/new-trig-240-l-old-pmt"
file_name = "C:/Users/lexda/PycharmProjects/Photek_lab/Ion_feedback/Ion_feedback_data/13150210-min-10-bi-ml"



trc_files = [os.path.join(file_name, f) for f in os.listdir(file_name) if f.endswith('.trc')]
#trc_files = [os.path.join(file_name, f) for f in os.listdir(file_name) if f.endswith('.trc')][:7]

results = Parallel(n_jobs=-1)(delayed(process_file)(file_name) for file_name in trc_files[0:10000])
looking_at_peaks(trc_files, 0, 1000)

hist_peaks = [item for sublist in [result[0] for result in results] for item in sublist]
num_peaks = [result[1] for result in results]
peak_heights_dict = [result[2] for result in results]
value = 1
greater_numbers = [num for num in num_peaks if num > value]

no_peaks = [num for num in num_peaks if num == 0]
no_ions = [num for num in num_peaks if num == 1]
ions = [num for num in num_peaks if num > 2]
no_peak_size = len(no_peaks)
no_ion_size = len(no_ions)
ion_size = len(ions)

#one_peak_times = [hist_peaks[i] for i in range(len(num_peaks)) if num_peaks[i] == 1]

# Plot histogram of these times
plt.figure(figsize=(10, 6))
#plt.hist(np.array(one_peak_times) * 1e9, bins=100)
plt.xlabel('Time (ns)')
plt.ylabel('Counts')
plt.title('Histogram of Times with Exactly One Peak')



plt.figure(figsize=(10, 6))
bin_edges = np.arange(0, np.max(num_peaks)+1, 0.5)

plt.hist(num_peaks, bins=bin_edges, align ="left", alpha=0.5, label='First Detected Dip in Voltage Signal')
#bin_centers = bin_edges[:-1]+0.5
plt.xlabel('number of peaks')
plt.ylabel('Frequency')
plt.title('Histogram of Number of Peaks')

time_differences = []
for peak_times in [result[0] for result in results]:
    if len(peak_times) > 1:
        first_peak_time = peak_times[0]
        for subsequent_peak_time in peak_times[1:]:
            time_differences.append(subsequent_peak_time - first_peak_time)

# Plot histogram of time differences
plt.figure(figsize=(10, 6))
time_differences_array = np.array(time_differences)*1e9
#hist_vals, bin_edges, _ = plt.hist(time_differences_array, bins=1000)
bin_width_ns = 0.33
hist_vals, bin_edges, _ = plt.hist(time_differences_array, bins=np.arange(min(time_differences_array), max(time_differences_array), bin_width_ns),
         align='left', alpha=0.5, label='time differences')



hist_peaks, _ = find_peaks(hist_vals, height=100, distance=7, prominence=8)

#peak_bin_centers = (bin_edges[hist_peaks] + bin_edges[hist_peaks + 1]) / 2
peak_bin_centers = bin_edges[hist_peaks] # (bin_width_ns )
peak_positions = bin_edges[hist_peaks]
peak_heights = hist_vals[hist_peaks]
print (f" peak bin centers {peak_bin_centers}")
plt.plot(peak_bin_centers, peak_heights, 'x')

#plt.yscale('log')
plt.xlim(0,50 )
print(np.array(time_differences) * 1e9)
dif = np.array(time_differences) * 1e9
np.savetxt('time_differences.csv', dif, delimiter=',')
plt.xlabel('Time Difference (ns)')
plt.ylabel('Counts')
plt.title('Histogram of Time Differences After First Peak')
plt.savefig('time_differences_hist.png')
print(f"number of time differences {len(time_differences)}")





plt.figure()
# The size (count) of numbers greater than the value is the length of the new list
size = len(greater_numbers)
print (f"number of no peaks {no_peak_size}, number of no ions {no_ion_size}, ions {ion_size}, number of greater numbers {size} "
       f"therefor number mutiple peaks in reluation to events ....")

print(f"peak positions {peak_positions}")
print(f"peak heights {peak_heights}")
first_peak = []
ion_peaks = []

plt.show()

    # for i in range(len(hist_peaks)):
    #     if hist_peaks[i] > 0.08e-7:
    #         ion_peaks.append(hist_peaks[i])
    #     else:
    #         first_peak.append(hist_peaks[i])
    #
    # first_peak = np.array(first_peak)
    # ion_peaks = np.array(ion_peaks)
    # total_photoelectrons = len(hist_peaks)
    # total_ions = len(ion_peaks)
    #
    # print(f"Total photoelectrons: {total_photoelectrons} total ions: {total_ions}")
    #
    # plt.figure(figsize=(10, 6))
    # mean_first_peak = np.mean(first_peak)
    # hist_vals, bin_edges, _ = plt.hist((ion_peaks-mean_first_peak)*1e9, bins=100)
    # hist_peaks, _ = find_peaks(hist_vals, height=25, distance=10)
    # print (f"feedback time {(ion_peaks-mean_first_peak)*1e9}")
    # peak_positions = bin_edges[hist_peaks]
    # peak_heights = hist_vals[hist_peaks]
    # plt.plot(peak_positions, peak_heights, 'x')
    # plt.hist((first_peak-mean_first_peak)*1e9, bins=500)
    # print(f"ion_peaks {peak_positions} ns")
    # plt.xlabel('Time (ns)')
    # plt.ylabel('Counts')
    # plt.title('Events vs. Time')
    # #plt.yscale('log')
    # plt.xlim(0, 50)
    # plt.show()
    # plt.savefig('binary_hist_plot.png')