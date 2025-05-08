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
    plt.title('afterpulse distribution')
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
            #  "ion-cathode-sweep/F2--ion-torch-with-iamge-d--hist-1.5k-500cath-1450-mcp-1500-back.txt",
             #   "ion-cathode-sweep/F2--ion-torch-with-iamge-d--hist-1.5k-600cath-1450-mcp-1500-back.txt",
            #"ion-cathode-sweep/F2--ion-torch-with-iamge-d--hist-1.5k-700cath-1450-mcp-1500-back.txt",
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

plt.xlabel("Time [ns]")
plt.ylabel("counts")
plt.title("Afterpulse distribution for different cathode-MCP gap voltages")
#plt.yscale('log')
plt.legend()
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