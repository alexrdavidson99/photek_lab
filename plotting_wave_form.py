import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
#from scipy.integrate import simps,trapezoid
import mplhep
mplhep.style.use(mplhep.style.LHCb2)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit



def gaus(x, A, mu, sigma):
    return A / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def rise_time(time,current):
    """
    Calculate the 10-90% rise time of a signal.
    
    Args:
        data (pd.Series): The signal data.
    
    Returns:
        float: The 10-90% rise time in nanoseconds.
    """
    current = -current  # Invert the current signal if needed
    # Calculate the 10% and 90% levels
    print(f"Current min: {current.min()}, Current max: {current.max()}")
    ten_percent =  0.1 * (current.max() )
    ninety_percent = 0.9 * (current.max() )
    
    # Find the indices where the signal crosses these levels
    ten_index = np.where(current >= ten_percent)[0][0]
    print(f"10% index: {ten_index}, 10% value: {current[ten_index]} , at time: {time[ten_index]}")
    ninety_index = np.where(current >= ninety_percent)[0][0]
    print(f"90% index: {ninety_index}, 90% value: {current[ninety_index]} , at time: {time[ninety_index]}")
    
        # Calculate the time difference
    rise_time_val = time[ninety_index] - time[ten_index]
    return rise_time_val, time[ten_index], time[ninety_index]



def plot_waveform(time, current, rise_upper, rise_lower):
    # Load the waveform data

  # Adjust the range as needed
    # Plot the waveform
    plt.figure(figsize=(12, 8))
    plt.plot(time* 1e9, current* 1e3, label='Waveform', color='red')
    #fitter
    # Fit a Gaussian to the waveform data
    popt, pcov = curve_fit(gaus, time, current, p0=[current.min(), -1e-9, 1e-10])
    x_fit = np.linspace(time.min(), time.max(), 1000)
    y_fit = gaus(x_fit, *popt)  
    #plt.plot(x_fit * 1e9, y_fit * 1e3, label='Gaussian Fit', color='blue')
    plt.vlines(rise_lower * 1e9, ymin=current.min() * 1e3, ymax=current.max() * 1e3, color='green', linestyle='--', label='10% Rise Time')
    plt.vlines(rise_upper * 1e9, ymin=current.min() * 1e3, ymax=current.max() * 1e3, color='orange', linestyle='--', label='90% Rise Time')
    print(f"Fitted parameters: {popt}")
    print(f"FWHM: {2 * np.sqrt(2 * np.log(2)) * popt[2] * 1e9:.2f} ns")  # Convert to ns
    # Add labels and legend
    plt.xlabel('Time (ns)')
    plt.ylabel('voltage (mV)')
    plt.title('Waveform Data')
    plt.legend()
    plt.grid()
    plt.show()



waveform_path = Path("C:/Users/lexda/local_pmt_info/characterisation/process_data_for_conference/voltage_mins/C2-waveform-200-avg.csv")
waveform_data = pd.read_csv(waveform_path, delimiter=',', names=['time', 'voltage'], skiprows=6)
waveform_data_time_window = waveform_data[(waveform_data['time'] >= -2e-9) & (waveform_data['time'] <= 0)] 
time = waveform_data_time_window ['time'].to_numpy()   
current = waveform_data_time_window ['voltage'].to_numpy()  



rise_time_value, rise_lower, rise_upper = rise_time(time, current)
print(f"Rise time (10-90%): {rise_time_value*1e9:.2f} ns")

plot_waveform(time, current,rise_upper, rise_lower)