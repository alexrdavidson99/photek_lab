import pandas as pd
import re
import os
from scipy.signal import find_peaks
import scipy
from itertools import cycle
import numpy as np
#from ion_plot import electron_trajectory
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import mplhep
import os
mplhep.style.use(mplhep.style.LHCb2)






def read_data_hist(filename):
    data = pd.read_csv(filename, comment='#', names=["time", "hits"], skiprows=5, delimiter=',')
    data = data[data["hits"] < 0]
    return data['time'], data['hits']

def plot_hist(hit_data,voltage):
    hist_vals, bin_edges, _ =  plt.hist(-pd_data_hits*1e9, bins=100, label =f"{cathode_voltage} V", alpha=0.5)
    hist_peaks, _ = find_peaks(hist_vals, height=100,prominence=100)
    peak_positions = bin_edges[hist_peaks]
    #plt.plot(peak_positions, hist_vals[hist_peaks], "x", color="red")
    print(peak_positions)

def plot_saved_hist(hist_time, counts,time):
    hist_time = (hist_time.to_numpy())*1e9
    plt.bar(hist_time, counts, width=(hist_time[1] - hist_time[0]), align='center', label=f"{time}")
    plt.xlabel('Time')
    plt.ylabel('Counts')
    #plt.title('Afterpulse distribution')
    plt.legend()

file_string = "C:/Users/lexda/local_pmt_info/characterisation/ion_feedback/hist/"
file_list = [#"afternoon/F2--ion-torch-with-iamge-d--hist-5.5k-700cath-1450-mcp-1500-back.txt", 
             #"afternoon/F2--ion-torch-with-iamge-d--hist-5.5k-700cath-1450-mcp.txt",
            # "day_2/F3--ion-torch-with-iamge-d-700cath-1450-mcp-1500-back-day-2.txt",
             #"afternoon/F2--ion-torch-with-iamge-d--hist-5.5k-150cath-1450-mcp.txt",
              #"ion-cathode-sweep/F2--ion-torch-with-iamge-d--hist-1.5k-100cath-1450-mcp-1500-back.txt",
              "ion-cathode-sweep/F2--ion-torch-with-iamge-d--hist-1.5k-200cath-1450-mcp-1500-back.txt",
              #"ion-cathode-sweep/F2--ion-torch-with-iamge-d--hist-1.5k-300cath-1450-mcp-1500-back.txt",
              #"ion-cathode-sweep/F2--ion-torch-with-iamge-d--hist-1.5k-400cath-1450-mcp-1500-back.txt",
              "ion-cathode-sweep/F2--ion-torch-with-iamge-d--hist-1.5k-500cath-1450-mcp-1500-back.txt",
             #   "ion-cathode-sweep/F2--ion-torch-with-iamge-d--hist-1.5k-600cath-1450-mcp-1500-back.txt",
            "ion-cathode-sweep/F2--ion-torch-with-iamge-d--hist-1.5k-700cath-1450-mcp-1500-back.txt",
            # "long_width/F2--ion-torch-with-iamge-d--hist-1.5k-700cath-1450-mcp-1500-back-long--1800ns.txt",
            # "long_width/F2--ion-torch-with-iamge-d--hist-1.5k-700cath-1450-mcp-1500-back-long--740ns.txt",
             ]
             #"F2--ion-torch-with-iamge-d--hist-5.5k.txt",]

file_list_hist = [#"afternoon/F1--ion-torch-with-iamge-d--hist-5.5k-700cath-1450-mcp-1500-back.txt", 
             #"afternoon/F1--ion-torch-with-iamge-d--hist-5.5k-700cath-1450-mcp.txt",
             #"day_2/F3--ion-torch-with-iamge-d-700cath-1450-mcp-1500-back-day-2.txt",
             #"afternoon/F2--ion-torch-with-iamge-d--hist-5.5k-150cath-1450-mcp.txt",
             # "ion-cathode-sweep/F1--ion-torch-with-iamge-d--hist-1.5k-100cath-1450-mcp-1500-back.txt",
             #"ion-cathode-sweep/F1--ion-torch-with-iamge-d--hist-1.5k-300cath-1450-mcp-1500-back.txt",
             # "ion-cathode-sweep/F1--ion-torch-with-iamge-d--hist-1.5k-400cath-1450-mcp-1500-back.txt",
             #"ion-cathode-sweep/F1--ion-torch-with-iamge-d--hist-1.5k-700cath-1450-mcp-1500-back.txt",
             "long_width/F1--ion-torch-with-iamge-d--hist-1.5k-700cath-1450-mcp-1500-back-long--1800ns.txt",
             "long_width/F1--ion-torch-with-iamge-d--hist-1.5k-700cath-1450-mcp-1500-back-long--740ns.txt",
             ]
                 

for file in file_list:
    pd_data_time, pd_data_hits  = read_data_hist(file_string + file) 
    pattern = r'(\d+)(?=cath)'
    cathode_voltage_str = re.findall(pattern, file)
    cathode_voltage = int(cathode_voltage_str[0])
   
    print(cathode_voltage)
    plot_hist(pd_data_hits, cathode_voltage)

csv_path = r"c:\Users\lexda\PycharmProjects\Photek_lab\tof_bounds_by_mass_voltage.csv"
# for 13150210 
#csv_path = r"c:\Users\lexda\PycharmProjects\Photek_lab\tof_bounds_by_mass_voltage_335um.csv"
# old_pmt = r"C:/Users/lexda/PycharmProjects/Photek_lab/time_differences_hist_13150210_works.csv"
# old_pmt_data = pd.read_csv(old_pmt)
# plt.figure(figsize=(14, 7))
# plt.bar(old_pmt_data['bin_center_ns'], old_pmt_data['count'], width=(old_pmt_data['right_edge_ns'] - old_pmt_data['left_edge_ns']), align='center', alpha=0.5, label="13150210 PMT")   

df_bounds = pd.read_csv(csv_path)

# choose the normalized mass to match (change if needed)
mass_to_match = 18.0  # H2+ and H2O+


# mass_labels = {
#     1.0:  r'$\mathrm{H^+}$', 
#     4.0:  r'$\mathrm{He^+}$',
#     18.0: r'$\mathrm{H_2O^+}$',
#     37.0: r'Heavy ion',  # example fragment (adjust as needed)
#     73.0: r'Heavy ion',  # example fragment (adjust as needed)
# }
#mass_colors = {1.0: "#ff0000", 4.0: "#ffa500", 18.0: "#0000ff", 37.0: "#615994", 73.0: "#800080"}
#mass_colors = {
#     1.0: "#1D56E5",   # deep orange-red
#     4.0: "#348ABD",   # soft blue
#     18.0: "#997AE7",  # lavender
#     37.0: "#615994",  # neutral gray
#     73.0: "#800080",  # warm yellow-orange
# }



ymin = 0
ymax = 400

#vol_colors = {200: "#f17354"}
# for m in mass_to_match:
#     for volt, color in vol_colors.items():
#         row = df_bounds[(df_bounds["Voltage (V)"] == volt) & (df_bounds["Mass number"] == m)]
#         ion_label = mass_labels.get(m, f"m={m}")
#         ion_color = mass_colors.get(m, f"m={m}")
#         if not row.empty:
#             tof_min = float(row["TOF_min (ns)"].values[0])
#             tof_max = float(row["TOF_max (ns)"].values[0])
#             plt.vlines([tof_min, tof_max], ymin=ymin, ymax=ymax, colors=ion_color,
#                        linestyles='dashed', label=fr'Time range for {ion_label}')
#plt.xlim(-1, 49)

# draw min/max TOF lines for each voltage
vol_colors = {200: "#5ba7ff", 500: "#ff8d40", 700: "#61dfe6"}
for volt, color in vol_colors.items():
    row = df_bounds[(df_bounds["Voltage (V)"] == volt) & (df_bounds["Mass number"] == mass_to_match)]
    
    if not row.empty:
        tof_min = float(row["TOF_min (ns)"].values[0])
        tof_max = float(row["TOF_max (ns)"].values[0])
        plt.vlines([tof_min, tof_max], ymin=ymin, ymax=ymax, colors=color,
                   linestyles='dashed', label=fr'Time range $\mathrm{{H_2O^+}}$ {volt}V')

plt.xlabel("Time [ns]")
plt.ylabel("Counts")
#plt.title("Afterpulse distribution for different cathode-MCP gap voltages")

save_dir = r"C:/Users/lexda/PycharmProjects/Photek_lab/Ion_feedback/plots"


os.makedirs(save_dir, exist_ok=True)

#plt.yscale('log')
plt.legend(fontsize=16.5)
plt.savefig(os.path.join(save_dir, "afterpulse_cathode_sweep.png"), dpi=300)
plt.figure()
for file in file_list_hist:
    filename = file_string + file
    #pd_data_time, pd_data_hits  = read_data_hist(filename) 
    file_name_without_extension = os.path.splitext(filename)[0]
    file_split = file_name_without_extension.split("-")
    time = next((part for part in file_split if "ns" in part), None)
    data = pd.read_csv(filename, comment='#', names=["time", "hits"], skiprows=5, delimiter=',')
    if file in ["afternoon/F1--ion-torch-with-iamge-d--hist-5.5k-700cath-1450-mcp-1500-back.txt", 
                "afternoon/F1--ion-torch-with-iamge-d--hist-5.5k-700cath-1450-mcp.txt", 
                "ion-cathode-sweep/F1--ion-torch-with-iamge-d--hist-1.5k-700cath-1450-mcp-1500-back.txt",]:
        data["time"] = -data["time"]


    pos_time_data = data[data["time"] > 0]
    pd_data_time, pd_data_hits  = pos_time_data['time'], pos_time_data['hits']

    

    plot_saved_hist(pd_data_time, pd_data_hits,time)
  

plt.show()