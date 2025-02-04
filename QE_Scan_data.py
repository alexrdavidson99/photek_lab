import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm , Normalize
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import mplhep
mplhep.style.use(mplhep.style.LHCb2)

def qe_vs_x(scan_data, specific_y,x_max,date):
        filtered_data = scan_data[scan_data['Y'] == specific_y]
        x_values = filtered_data['X'].values
        subtraction_values = filtered_data['Subtraction'].values
        print(subtraction_values)
        plt.plot(x_values, subtraction_values, label=date)
        plt.xlabel('y[mm]')
        plt.ylabel('Photocurrent')
        plt.title(f'Photocurrent vs y at x = {specific_y-34.125} mm')

        return 

scan_data = pd.read_csv("C:/Users/lexda/Downloads/qe-scan-0.25-a11240313_alex_405nm.csv",
                         sep=',', comment="#" , names =[ 'X', 'Y', 'Timestamp', 'Dark', 'Signal', 'Subtraction'])


step_size = 0.5



#scan_data = pd.read_csv("C:/Users/lexda/Downloads/A3241111_405nm.csv",
#                          sep=',', comment="#" , names =[ 'X', 'Y', 'Timestamp', 'Dark', 'Signal', 'Subtraction'])
#print(scan_data)


scan_data = pd.read_csv("C:/Users/lexda/local_pmt_info/QE_A3241111/TMC-SQ_TMC-SQ_405nm.csv",
                          sep=',', comment="#" , names =[ 'X', 'Y', 'Timestamp', 'Dark', 'Signal', 'Subtraction'])
print(scan_data)

x_min, x_max = scan_data['X'].min(), scan_data['X'].max()
print(x_min, x_max)

y_min, y_max = scan_data['Y'].min(), scan_data['Y'].max()
specific_y = 32.25+0.25*50

qe_vs_x(scan_data, specific_y, x_max, '2024-26-09')
plt.legend()

print(y_max, y_min)
print(y_min)
x_bins = int((x_max - x_min) / step_size)
print(x_bins)

y_bins = int((y_max - y_min) / step_size)
print(y_bins)

x_centered = np.abs(scan_data['X'] - x_max + step_size / 2)
y_centered = np.abs(scan_data['Y'] - y_max + step_size / 2)
#plt.figure(figsize=(5.3, 3.24))
plt.figure(figsize=(13, 9))
norm = Normalize(vmin=0, vmax=0.16)
plt.hist2d(y_centered , x_centered, weights=scan_data['Subtraction'], bins=[x_bins, y_bins], cmap=plt.cm.viridis,norm=LogNorm(vmin=0.01, vmax=0.53))
plt.colorbar(label='photocurrent')
plt.xlabel('x[mm]', fontsize=30)
plt.ylabel('y[mm]')
plt.title('QE distribution ', fontsize=30)
plt.xticks(np.arange(0, 60, step=10))
plt.yticks(np.arange(0, 60, step=10))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.gca().yaxis.set_major_locator(MultipleLocator(10))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
title = f'Photocurrent vs x and y'
plt.fontsize = 300
plt.savefig('Photocurrent_vs_x_and_y.png')

#plt.axhline(y=specific_y - y_max + step_size / 2, color='red', linestyle='--', linewidth=2, label='1D Sample Position')
plt.show()





#plt.figure(figsize=(16, 8))
x = scan_data['X'].values
y = scan_data['Y'].values
z = scan_data['Subtraction'].values
#levels = 3000
#plt.tricontourf(x, y, z, levels=levels, cmap='plasma')

#plt.colorbar(label='|E|')
#plt.xlabel('x')
#plt.ylabel('y')

