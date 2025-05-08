import lecroyparser
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from joblib import Parallel, delayed
import numpy as np
import os
import re
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import LogNorm
from scipy.signal import butter, filtfilt
from numpy.fft import fft, fftfreq

def high_pass_filter(data, cutoff, fs, order=1):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def read_data_hist(filename):
    data = lecroyparser.ScopeData(filename)
    print(f"filename {filename}")
    return data.x, data.y

def process_file(file_name):
    time, voltage = read_data_hist(file_name)
    threshold = 1000

    pattern = r'--(\d+)'  # Regular expression to extract the number after '--'
    file_number = re.search(pattern, file_name)
    
    #fs = 1 / (time[1] - time[0])  # Sampling frequency
    #cutoff = 1  # Cutoff frequency in Hz (20 GHz)
    #voltage = high_pass_filter(voltage, cutoff, fs)
    # Invert the voltage to find dips as peaks
    voltage = -voltage
    voltage = gaussian_filter1d(voltage, sigma=5)
    
    # Find the first significant peak (dip in original data)
    peaks, _ = find_peaks(voltage, height=0.001,distance=1600, prominence=0.005)
    peak_heights = voltage[peaks]



    peak_heights_dict = {i: height for i, height in enumerate(voltage[peaks]) if height < threshold}
    peak_times = [time[i].astype(float) for i in peaks if voltage[i] < threshold]

    return peak_times, len(peak_times), peak_heights_dict, file_number.group(1)


def looking_at_peaks(file_names, i,j):
    waveform_voltages = []
    waveform_times = []
    
    
    plt.figure()
    for file_name in file_names[i:j]:
        
        time_c2, voltage_c2 = read_data_hist(file_name)
        voltage_c2 = -voltage_c2
        peaks_c2, _ = find_peaks(voltage_c2, height=0.001,distance=1600, prominence=0.005
                                )
        peak_time = time_c2[peaks_c2]
        plt.plot((time_c2)*1e9, voltage_c2)
        plt.plot((time_c2[peaks_c2]) * 1e9, voltage_c2[peaks_c2], 'x')

            #smothing the data
            #voltage_hp = savgol_filter(voltage, window_length=850, polyorder=6)
            #voltage_guass = gaussian_filter1d(voltage_hp, sigma=5)
            # Find the first significant peak (dip in original data)
            

            #if len(peak_times) ==1 : #< 2e-9:
            #    plt.plot((time_c2-peak_times[0])*1e9, voltage_c2)
            #    waveform_voltages.append(voltage_c2)
            #    waveform_times.append((time_c2-peak_times[0])*1e9)
            #    plt.plot(peak_time*1e9, peak_positions, 'x')
            #    plt.plot(time_c2 * 1e9, voltage_c2)

            # if len(peak_times) >=2 : #< 2e-9:
            #     plt.plot((time_c2-peak_times[0])*1e9, voltage_c2)
            #     waveform_voltages.append(voltage_c2)
            #     waveform_times.append((time_c2-peak_times[0])*1e9)
            #     plt.plot((peak_time-peak_times[0])*1e9, peak_positions, 'x')
            #     plt.plot((time_c2-peak_times[0]) * 1e9, voltage_c2)
                
            #     ion += 1

        
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
file_name = "C:/Users/lexda/local_pmt_info/characterisation/ion_feedback/ion-torch"




trc_files = [os.path.join(file_name, f) for f in os.listdir(file_name) if f.endswith('.trc')]
#trc_files = [os.path.join(file_name, f) for f in os.listdir(file_name) if f.endswith('.trc')][:7]

results = Parallel(n_jobs=-1)(delayed(process_file)(file_name) for file_name in trc_files[0:1057])
looking_at_peaks(trc_files, 860, 867)

hist_peaks = [item for sublist in [result[0] for result in results] for item in sublist]
num_peaks = [result[1] for result in results]
peak_heights_dict = [result[2] for result in results]

file_numbers = [result[3] for result in results]
value = 2
greater_numbers = [num for num in num_peaks if num > value]

no_peaks = [num for num in num_peaks if num == 0]
no_ions = [num for num in num_peaks if num == 1]
ions = [num for num in num_peaks if num >= 2]
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
first_peak_times = []
file_numbers = []

for result in results:  # Iterate over each result tuple
    peak_times = result[0]  # Extract peak_times from the tuple
    if len(peak_times) > 0:
        #first_peak_times.append(float(peak_times[0]))  # Convert to raw float
        file_numbers.append(result[3])  # Append the file number

for peak_times in [result[0] for result in results]:
    if len(peak_times) > 0:
        first_peak_times.append(peak_times[0])
        
    if len(peak_times) > 1:
        first_peak_time = peak_times[0]
        for subsequent_peak_time in peak_times[1:]:
            time_differences.append(subsequent_peak_time - first_peak_time)

# Plot histogram of time differences
plt.figure(figsize=(10, 6))
time_differences_array = np.array(time_differences)*1e9
hist_vals, bin_edges, _ = plt.hist(time_differences_array, bins=100)
bin_width_ns = 0.33
print(f"test {time_differences_array}")
#hist_vals, bin_edges, _ = plt.hist(time_differences_array, bins=np.arange(min(time_differences_array), max(time_differences_array), bin_width_ns),
#         align='left', alpha=0.5, label='time differences')



hist_peaks, _ = find_peaks(hist_vals, height=100, distance=7, prominence=8)

#peak_bin_centers = (bin_edges[hist_peaks] + bin_edges[hist_peaks + 1]) / 2
peak_bin_centers = bin_edges[hist_peaks] # (bin_width_ns )
peak_positions = bin_edges[hist_peaks]
peak_heights = hist_vals[hist_peaks]
print (f" peak bin centers {peak_bin_centers}")
plt.plot(peak_bin_centers, peak_heights, 'x')

#plt.yscale('log')
np.savetxt('time_differences.csv', time_differences_array, delimiter=',')
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
#df = pd.DataFrame({'file_numbers':file_numbers,'first_peak_times':first_peak_times})

peak_data = []

for result in results:
    peak_times = result[0]  # list of peak times for this file
    file_number = result[3]  # extracted from filename using regex
    peak_data.append({
        'file_number': file_number,
        'peak_times': peak_times
    })

# Create DataFrame
df_peak_times = pd.DataFrame(peak_data)

# Optional: Convert lists to strings for cleaner CSV output
df_peak_times['peak_times'] = df_peak_times['peak_times'].apply(lambda x: ','.join([f"{t:.10e}" for t in x]))

# Save to CSV
#df_peak_times.to_csv('peak_times_by_file.csv', index=False)

#df.to_csv('ion_feedback_ion_times.csv', index=False)