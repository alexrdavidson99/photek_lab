import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
#from scipy.integrate import simps,trapezoid
import mplhep
mplhep.style.use(mplhep.style.LHCb2)
from anode_layout_plot import layout_of_anode
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit
# Variables
int_time = 1800.0
serial = "A12 TORCH"
folder_path = Path("C:/Users/lexda/Desktop/QE/post-pot-gain/J-CD0/")
base_folder_path = Path("C:/Users/lexda/Desktop/QE/post-pot-gain/J-CD0/")

def polyia(x, A, mu, sigma, alpha):
    return A * np.exp(-((x - mu)**2 / (2 * sigma**2))) * (1 + alpha * (x - mu))
def gaus(x, A, mu, sigma):
    return A / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu)**2 / (2 * sigma**2))




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
    print(f"Calibration file: {cal_f}")
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









# base_folders = [
#     "C:/Users/lexda/local_pmt_info/characterisation/2024-09-25 LPG650 PHD/J_OP1/dropper_chain_2.728V",
#     "C:/Users/lexda/local_pmt_info/characterisation/2024-09-25 LPG650 PHD/J_CD0/dropper_chain_2.728V",
#     "C:/Users/lexda/local_pmt_info/characterisation/2024-09-25 LPG650 PHD/J_GH1/dropper_chain_2.728V",
#     "C:/Users/lexda/local_pmt_info/characterisation/2024-09-25 LPG650 PHD/J_KL0/dropper_chain_2.728V",
#     "C:/Users/lexda/local_pmt_info/characterisation/2024-09-25 LPG650 PHD/J_KL1/dropper_chain_2.728V",
    

# ]


#base_folders = ["C:/Users/lexda/local_pmt_info/characterisation/rate-cap/full-pix-gain/"]
base_folders = ["C:/Users/lexda/local_pmt_info/characterisation/2024-09-25 LPG650 PHD/J_KL0/22_04_2025/"]
# Recursively find all CSV files matching the pattern *SIG_*.csv #200_cathode_1340_MCP_3mins_int_35db
#sig_files = list(base_folder_path.glob("*1500v_MCP*.txt"))
sig_files = list(base_folder_path.rglob("*sig.csv"))
print(sig_files)

counts = []
position = []
counts_dn = []
counts_sig = []
color_mapping = {
        "OP1": "blue",
        "CD0": "orange",
        "GH1": "cyan",
        "IK0": "green",
        "IK1": "red",
    }


plt.figure()
for base_folder in base_folders:
    base_folder_path = Path(base_folder)
    print (base_folder_path)    
    sig_files = list(base_folder_path.rglob("*sig.csv"))
    for i, sig_f in enumerate(sig_files):
        parts = sig_f.stem.split("_")
         
        # Find the index of 'volts' then get the number just before it
        if "pix" in parts:
            volts_index = parts.index("KL1")
            voltage_value = parts[volts_index - 1]
        else:
        #    volts_index = parts.index("dropper")
            voltage_value = parts[1]
            #y_pos = float(parts[2])

        # Format the number to two decimal places
        #formatted_number = f"{number:.2f}"

        #position.append(y_pos)
        chans, sig = np.loadtxt(sig_f, delimiter=",", skiprows=1, unpack=True)
        sig *= gain_norm[gain][0]
        
        total_events = np.sum(sig)
        mean_gain = np.sum(chans * sig) / total_events  # Weighted mean
        mean_gain = np.round(mean_gain)
        print(f"Mean gain: {mean_gain}")
        print(f"Mean gain: {cal[int(mean_gain)]}")

        counts_sig.append(sum(sig)/int_time)


        dn_f = sig_f.with_name(sig_f.name.replace("sig", "dn"))
        chans_df, df = np.loadtxt(dn_f, delimiter=",", skiprows=1, unpack=True)
        df *= gain_norm[gain][0]
        counts_dn.append(sum(df)/int_time)


        print(sum(sig)/5)
        counts.append(sum(sig)/int_time- sum(df)/int_time)
        sub = sig 
        sub[sub < 0] = 0
       
        sub_peak = np.max(sub)
        nom_sub = sub
        list_name = sig_f.name.split("_")
        print(list_name)
        popt, _ = curve_fit(gaus, cal,  nom_sub[:-1], p0=[10e4, 1e6, 2e6])
        fit_x = np.linspace(0, 2.5e6, 1000)
        fit_y = gaus(fit_x, *popt)
        print(f"popt: {popt}")
        #plt.plot(fit_x, fit_y, label=f"{voltage_value} {list_name[0]}")
        
        print(f"Votage {voltage_value} color: {color_mapping.get(voltage_value, 'black')}")
        plt.plot(cal, nom_sub[:-1], lw=1, label=f"{voltage_value}")
            #color=color_mapping.get(voltage_value, "black"))
        
        plt.xlabel("Gain (electrons)")
        plt.ylabel("Normalised counts ")
        plt.title(f"Normalised PHD vs. Gain for 2by48 pixels anode sections")
        plt.yscale("log")
        #plt.xlim(0, 2.5e6)
        plt.legend()


plt.gca().xaxis.set_label_coords(0.5, -0.1)  # Adjusts x label
plt.gca().yaxis.set_label_coords(-0.1, 0.5) 

# Add an inset for the layout of anode
#inset_ax = inset_axes(plt.gca(), width="30%", height="30%", loc='upper right')  # Adjust size and position
fig_anode, ax_anode = layout_of_anode()

# Transfer the contents of layout_of_anode to the inset axes
#for line in ax_anode.get_lines():
#    inset_ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color())
#inset_ax.set_aspect(aspect='equal')
#inset_ax.axis('off') 

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
 # Adjusts y label
plt.show()