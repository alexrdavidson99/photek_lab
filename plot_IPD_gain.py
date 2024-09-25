import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Variables
int_time = 60.0
serial = "A12 TORCH"
folder_path = Path("C:/Users/lexda/Desktop/QE/post-pot-gain/J-CD0/")
base_folder_path = Path("C:/Users/lexda/Desktop/QE/post-pot-gain/J-CD0/")

# Get the base name of the folder
base_name = folder_path.name
print(base_name)

# Recursively find all CSV files matching the pattern *SIG_*.csv
sig_files = list(base_folder_path.rglob("*SIG_200_1450*.csv"))
print(sig_files)

# Optional: Just print the file names without full path
csv_file_names = [file.name for file in sig_files]
print(csv_file_names)

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
    elif gain not in gain_norm:
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