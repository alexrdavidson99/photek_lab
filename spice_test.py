from PyLTSpice import RawRead
import mplhep
mplhep.style.use(mplhep.style.LHCb2)
import os
from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import chirp, find_peaks, peak_widths

def plot_waveform(time, current,):
    # Load the waveform data

  # Adjust the range as needed
    # Plot the waveform
   
    #plt.plot(time* 1e9, current* 1e3, label='Waveform', color='red')
    #fitter
    # Fit a Gaussian to the waveform data
    #plt.plot(x_fit * 1e9, y_fit * 1e3, label='Gaussian Fit', color='blue')
    #find peak 
    peaks, _ = find_peaks(-current, height=0.0069)  # Invert current to find negative peaks
    print(f"Peaks found at indices: {peaks}, Peak values: {current[peaks]}")
    #plt.plot(time[peaks]* 1e9, current[peaks]* 1e3, "x", label='Peaks', color='blue')
    # normaise to peak
    norm_current = current / np.abs(current[peaks]).max()
    time_shifted = time - time[peaks][0]  # Shift time so that the first peak is at t=0
    plt.plot(time_shifted* 1e9, norm_current, label='Experimental Data', color='orange')
    #plt.plot(time* 1e9, current* 1e3, label='Waveform', color='red')
    # Add labels and legend
    plt.xlabel('Time (ns)')
    plt.ylabel('voltage (mV)')
    plt.title('Normalized Waveform Data')
    plt.xlim(-1, 20)
    plt.legend(loc='lower right', fontsize=24)
    plt.grid()
    plt.show()





LTR = RawRead("C:/Users/lexda/PycharmProjects/Induced_charge_ramo_sim/cross_talk_spice.raw")

V_list  = ['V(n1)', 'V(n2)', 'V(n001)', 'V(n002)', 'V(n004)']
v_names = ['Next to Neighbour 1', 'Neighbour 1 ', 'Center', 'Neighbour 2', 'Next to Neighbour 2']
#clour_list = ['red', 'red', 'green', 'red', 'red']
clour_list = ["#ff0000", '#8bdb68', '#38761d', '#8bdb68', '#ff0000']
clour_list = ["#ff0000", '#ff0000', '#38761d', '#ff0000', '#ff0000']

for v in V_list:
    v_line = LTR.get_trace(v)  # Gets the voltage at node n1"
    x = LTR.get_trace('time')  # Gets the time axis
    steps = LTR.get_steps()
    for step in range(len(steps)):
        # print(steps[step])
        plt.plot(x.get_wave(step)*1e9, v_line.get_wave(step)*1e3, label=f'{v_names[V_list.index(v)]} ', color=clour_list[V_list.index(v)]) 
plt.xlabel("Time (ns)")
plt.locator_params(axis='x', nbins=2.5)
plt.ylabel("Voltage (mV)")
#plt.title("SPICE Simulation of Crosstalk from Anode Capacitively Coupling & Image Charge ", fontsize=18)
plt.xlim(0, 2)
plt.legend(loc='lower right', fontsize=22)
#plt.savefig("C:/Users/lexda/PhD_thesis/65f4c29cf553e8d908bd542a/figures/Sim_chapter/Picture_pd_25.png", dpi=300)

v_line = LTR.get_trace('V(n001)')  # Gets the voltage at node n1"
x = LTR.get_trace('time')  # Gets the time axis
steps = LTR.get_steps()

before_anode = pd.read_csv("C:/Users/lexda/PycharmProjects/Induced_charge_ramo_sim/induced_current/summed_results/summed_induced_current_127__center_120_1500v_in__z_120_sumed.csv", delimiter=',', names=['time', 'current'] )

time = before_anode['time']*1e9  # to ns
current = before_anode['current']*1e3  # to mA
voltage = current*50  # to mV across 50 ohm

plt.figure()
for step in range(len(steps)):
    # print(steps[step])
    time = x.get_wave(step)*1e9
    voltage = v_line.get_wave(step)*1e3
    peaks, _ = find_peaks(-voltage, height=0.0069)
    norm_voltage = voltage / np.abs(voltage[peaks]).max()
    print(f"Peaks at: {time[peaks]}, Peak values: {voltage[peaks]}")
    plt.plot(time-time[peaks][0], norm_voltage, label=f'Ramo Theorem Induced Current')
    #plt.plot(x.get_wave(step)*1e9, v_line.get_wave(step)*1e3, label=f'center pixel after anode') 


#plt.plot(time, voltage, label='center pixel before anode', linestyle='--')
  # order a legend

waveform_path = Path("C:/Users/lexda/local_pmt_info/characterisation/process_data_for_conference/voltage_mins/C2-waveform-200-avg.csv")
waveform_data = pd.read_csv(waveform_path, delimiter=',', names=['time', 'voltage'], skiprows=6)
waveform_data_time_window = waveform_data[(waveform_data['time'] >= -2e-9) & (waveform_data['time'] <= 0)] 
time = waveform_data_time_window ['time'].to_numpy()   
current = waveform_data_time_window ['voltage'].to_numpy()  

# shift time to align peaks
# find peak in current 


plot_waveform(time, current)






plt.show()