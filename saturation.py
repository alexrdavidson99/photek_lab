import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
#from scipy.integrate import simps,trapezoid
import mplhep
mplhep.style.use(mplhep.style.LHCb2)
from anode_layout_plot import layout_of_anode
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit
import pandas as pd
#import matplotlib.pyplot as plt
# Variables

gain_data = pd.read_csv("C:/Users/lexda/local_pmt_info/UTF-A1181001_Gain_CSV_0.326cm2.csv", sep=',', comment="#" ,skiprows=10)

print(gain_data.head())
print(gain_data.columns)
gain_column_name = 'GAIN.1'
voltage_column_name = 'Voltage (V)'  # Get the name of the 11th column
gain_values = gain_data[gain_column_name][:14].astype(float).to_numpy()  # Convert to float and numpy array
voltage_values = gain_data[voltage_column_name][:14].astype(float).to_numpy()


plt.scatter(voltage_values[:14], gain_values[:14], label='Gain vs Voltage')    
print(gain_values[:14])
print(voltage_values[:14])
plt.xlabel('Voltage (V)')
plt.ylabel('Gain')
plt.title('Gain vs Voltage for A1181001 (0.326 cm2)')
plt.yscale('log')
plt.legend()
plt.grid()
plt.show()





