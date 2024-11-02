import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import simps,trapezoid

# Variables
int_time = 1800.0
serial = "A12 TORCH"
folder_path = Path("C:/Users/lexda/Desktop/QE/post-pot-gain/J-CD0/")
base_folder_path = Path("C:/Users/lexda/Desktop/QE/post-pot-gain/J-CD0/")

# Calibration function
def calibrate(fname):
    q = 1.68e6
    cal_phd = np.loadtxt(fname, delimiter=",", skiprows=2)
    pk_chan = cal_phd[cal_phd[:, 1].argmax(), 0]
    print(f"Peak channel: {pk_chan}")

    m = q / pk_chan
    calibrated = cal_phd[:, 0] * m
    print(f"Calibrated values: {calibrated}")
    
    return calibrated

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


# Get the base name of the folder
base_name = folder_path.name
print(base_name)

# Recursively find all CSV files matching the pattern *SIG_*.csv
sig_files = list(base_folder_path.rglob("*SIG_200_1450*.csv"))
print(sig_files)

# Optional: Just print the file names without full path
csv_file_names = [file.name for file in sig_files]
print(csv_file_names)



# Initialize arrays for data
dn_rate = np.zeros(len(sig_files))
sig_rate = np.zeros(len(sig_files))
voltages = np.zeros(len(sig_files))

gain_norm = {}

# Process each SIG file
for i, sig_f in enumerate(sig_files):
    # Extract voltage from file name
    v = sig_f.stem.split("_")[2]
    print(f"Voltage: {v}")
    
    voltages[i] = float(v)

    dn_f = sig_f.with_name(sig_f.name.replace("SIG", "DN"))
    print(f"DN file: {dn_f}")

    # Extract gain from file name
    gain = sig_f.stem.split("_")[-3]
    print(f"Gain: {gain}")

    # Locate the calibration file
    cal_f = list(folder_path.glob("IPD_CAL*.csv"))[0]
    cal = calibrate(cal_f)

    # Correct for inconsistent bin sizes between different IPD gain values
    if gain not in gain_norm and len(gain_norm) == 0:
        gain_norm[gain] = (1, cal[-1])
        first_gain = gain

        print(f"First gain: {first_gain}")

    elif gain not in gain_norm:
        print(f"test {gain_norm[first_gain]}")
        first_norm = gain_norm[first_gain]

        cur_bin_size = cal[-1] / len(cal)
        old_bin_size = first_norm[1] / len(cal)
        scale_factor = old_bin_size / cur_bin_size
        
        gain_norm[gain] = (scale_factor, cal[-1])
    
    print(f"Gain normalization: {gain_norm}")

    # Load SIG and DN data
    chans, sig = np.loadtxt(sig_f, delimiter=",", skiprows=2, unpack=True)
    sig *= gain_norm[gain][0]
    sig_rate[i] = sig.sum() / int_time

    _, dn = np.loadtxt(dn_f, delimiter=",", skiprows=2, unpack=True)
    dn *= gain_norm[gain][0]
    dn_rate[i] = dn.sum() / int_time

    # Subtract DN from SIG
    sub = sig - dn
    sub[sub < 0] = 0
    

    # Plot the results
    plt.plot(cal, sub, label=f"{float(v)} V {sig_f.parent.name}")

# Plotting and saving the figure
plt.legend()
plt.xlabel("Gain (10$^6$ electrons)")
plt.ylabel("N events")
plt.title(f"MAPMT-256 TORCH {serial} - PHD vs. MCP Voltage")
plt.savefig("PHD.pdf")
plt.savefig("PHD.png", dpi=600)


#looking at IPD gain at multiple points along x 

#base_folder_path = Path("C:/Users/lexda/Downloads/dropper_chain_2.728V/")
base_folder_path = Path("C:/Users/lexda/local_pmt_info/characterisation/2024-09-25 LPG650 PHD/J_KL1/sum_test")
# Recursively find all CSV files matching the pattern *SIG_*.csv
sig_files = list(base_folder_path.rglob("*SIG*.txt"))
print(sig_files)

counts = []
position = []
counts_dn = []
counts_sig = []
plt.figure()

for i, sig_f in enumerate(sig_files):
    parts = sig_f.stem.split("_")
    #y_pos = float(parts[2])

    # Format the number to two decimal places
    #formatted_number = f"{number:.2f}"

    #position.append(y_pos)
    chans, sig = np.loadtxt(sig_f, delimiter="\t", skiprows=2, unpack=True)
    sig *= gain_norm[gain][0]
    
    counts_sig.append(sum(sig)/int_time)

    
    dn_f = sig_f.with_name(sig_f.name.replace("SIG", "DN"))
    chans_df, df = np.loadtxt(dn_f, delimiter="\t", skiprows=2, unpack=True)
    df *= gain_norm[gain][0]
    counts_dn.append(sum(df)/int_time)
    
   
    print(sum(sig)/5)
    #counts.append(sum(sig)/int_time- sum(df)/int_time)
    sub = sig - df
    sub[sub < 0] = 0
    smoothed_cal = moving_average(cal, 20)
    smoothed_df = moving_average(sub[:-1], 20)
    #numeric_value = float(parts[2])
    area = simps(sub, dx=cal[1] - cal[0])

    print(f"Area: {area}")
    area = trapezoid(sub, dx=cal[1] - cal[0])
    print(f"Area: {area}")
    counts.append(area)
    #if -10 <= numeric_value <= 10:
    #    None
        #plt.plot(smoothed_cal, smoothed_df, label=f"{float(v)} V {sig_f.name}")
        
    plt.plot(cal, sub[:-1], label=f"{float(v)} V {sig_f.name}")
    plt.xlabel("Gain (10$^6$ electrons)")
    plt.ylabel("N events")
    plt.xlim(0, 2e6)
   


position = np.array(position)
counts = np.array(counts)
counts_dn = np.array(counts_dn)
counts_sig = np.array(counts_sig)
print(position)
ind = position.argsort()
position = position[ind]
print(position)
counts = counts[ind]
counts_dn = counts_dn[ind]
counts_sig = counts_sig[ind]
plt.legend()
plt.figure()
plt.plot(position,counts, label = "subtracted")
plt.plot(position,counts_dn, label = "DN")
plt.xlabel("Pixel number")
plt.ylabel("Counts")
#plt.hlines(8000,-4.6,1.4,colors='r',linestyles='dashed')
#plt.text(0,8000,'length of connector 6.6mm  ',color='r')
#plt.plot(position,counts_sig, label = "SIG")
plt.legend()
plt.show()