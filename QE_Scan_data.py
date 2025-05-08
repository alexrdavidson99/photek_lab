import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm , Normalize
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import mplhep
mplhep.style.use(mplhep.style.LHCb2)

def qe_vs_axis(scan_data, specific_c,y_min,x_min, axis,date):
    if axis == 'x':
        specific_y_off_set = specific_c + y_min
        filtered_data = scan_data[scan_data['Y'] == specific_y_off_set]

        x_values = filtered_data['X'].values
        subtraction_values = filtered_data['Subtraction'].values
        #norm_sub = (subtraction_values - np.min(subtraction_values)) / (np.max(subtraction_values) - np.min(subtraction_values))
        print(subtraction_values)
        abs = (x_values - x_min)
        plt.plot(abs, subtraction_values, label=date)
        plt.xlabel('[mm]')
        
    elif axis == 'y':
        specific_x_off_set = specific_c + x_min
        filtered_data = scan_data[scan_data['X'] == specific_x_off_set]
        y_values = filtered_data['Y'].values
        subtraction_values = filtered_data['Subtraction'].values
        #norm_sub = (subtraction_values - np.min(subtraction_values)) / (np.max(subtraction_values) - np.min(subtraction_values))
        print(y_values)
        print(subtraction_values)
        abs = (y_values - y_min)
        plt.plot(abs, subtraction_values, label=date)
        plt.xlabel('[mm]')
    else:
        raise ValueError("Axis must be 'x' or 'y'")
    #plt.ylim(0.12,0.22)
    plt.ylabel('Photocurrent')
    plt.title(f'Photocurrent vs  coordinate at 2 wavelengths')

    return


def qe_vs_x(scan_data, specific_y,y_min,x_min,date):
        specific_y_off_set = specific_y + y_min
        filtered_data = scan_data[scan_data['Y'] == specific_y_off_set]
        x_values = filtered_data['X'].values
        subtraction_values = filtered_data['Subtraction'].values
        print(subtraction_values)
        abs = np.abs(x_values - x_min)
        plt.plot(abs, subtraction_values, label=date)
        plt.xlabel('y[mm]')
        plt.ylabel('Photocurrent')
        plt.title(f'Photocurrent vs y at x = {specific_y-34.125} mm')

        return 

scan_data = pd.read_csv("C:/Users/lexda/Downloads/qe-scan-0.25-a11240313_alex_405nm.csv",
                         sep=',', comment="#" , names =[ 'X', 'Y', 'Timestamp', 'Dark', 'Signal', 'Subtraction'])


#scan_data = pd.read_csv("C:/Users/lexda/local_pmt_info/QE_A3241111/A3241111_405nm.csv",
#                          sep=',', comment="#" , names =[ 'X', 'Y', 'Timestamp', 'Dark', 'Signal', 'Subtraction'])





#qe_vs_x(scan_data, specific_y, y_min, x_min, '2024-11-21')

#scan_data = pd.read_csv("C:/Users/lexda/local_pmt_info/QE_A3241111/A3241111_405nm.csv",
#                          sep=',', comment="#" , names =[ 'X', 'Y', 'Timestamp', 'Dark', 'Signal', 'Subtraction'])
x_min, x_max = scan_data['X'].min(), scan_data['X'].max()
y_min, y_max = scan_data['Y'].min(), scan_data['Y'].max()

step_size = 0.5
specific_y = 26
specific_x = 21

x_bins = int((x_max - x_min) / step_size)
y_bins = int((y_max - y_min) / step_size)


x_centered = np.abs(scan_data['X'] - x_max + step_size / 2)
y_centered = np.abs(scan_data['Y'] - y_max + step_size / 2)


#qe_vs_x(scan_data, specific_y, y_min, x_min, '2025-01-10')
qe_vs_axis(scan_data, specific_y,y_min,x_min, 'y', '2025-01-10 - 650nm constant Y')
qe_vs_axis(scan_data, specific_x,y_min,x_min, 'x', '2025-01-10 - 650nm constant X')
#qe_vs_x(scan_data, specific_y, y_min, x_min, '2025-01-10')
#scan_data = pd.read_csv("C:/Users/lexda/local_pmt_info/QE_A3241111/TMC-SQ_TMC-SQ_405nm.csv",
#                          sep=',', comment="#" , names =[ 'X', 'Y', 'Timestamp', 'Dark', 'Signal', 'Subtraction'])

x_min, x_max = scan_data['X'].min(), scan_data['X'].max()
y_min, y_max = scan_data['Y'].min(), scan_data['Y'].max()

step_size = 0.5
specific_y = 26.5+4.692
specific_x = 21.727+3.3+3

x_bins = int((x_max - x_min) / step_size)
y_bins = int((y_max - y_min) / step_size)


x_centered = np.abs(scan_data['X'] - x_max + step_size / 2)
y_centered = np.abs(scan_data['Y'] - y_max + step_size / 2)


#qe_vs_x(scan_data, specific_y, y_min, x_min, '2025-01-10')
qe_vs_axis(scan_data, specific_y,y_min,x_min, 'y', '2025-01-10 - 405nm constant Y')
qe_vs_axis(scan_data, specific_x,y_min,x_min, 'x', '2025-01-10- 405nm constant X')

qe_vs_axis(scan_data, specific_y-9.9,y_min,x_min, 'y', '2025-01-10 - 405nm constant Y')
qe_vs_axis(scan_data, specific_x-3.3,y_min,x_min, 'x', '2025-01-10- 405nm constant X')




plt.legend()



print(y_max, y_min)
print(y_min)
x_bins = int((x_max - x_min) / step_size)
print(x_bins)

y_bins = int((y_max - y_min) / step_size)
print(y_bins)

x_centered = scan_data['X'] - x_min + step_size / 2
y_centered = scan_data['Y'] - y_min + step_size / 2
#plt.figure(figsize=(5.3, 3.24))
plt.figure(figsize=(13, 9))
norm = Normalize(vmin=0.4, vmax=0.7)
plt.hist2d(y_centered , x_centered, weights=scan_data['Subtraction'], bins=[x_bins, y_bins], cmap=plt.cm.viridis, norm=norm)#LogNorm(vmin=0.01, vmax=0.2))

plt.colorbar(label='photocurrent')
plt.xlabel('x[mm]', fontsize=30)
plt.ylabel('y[mm]')
plt.title('QE distribution for 0.326cm^2 mask', fontsize=30)
plt.xticks(np.arange(0, 60, step=10))
plt.yticks(np.arange(0, 60, step=10))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
plt.gca().yaxis.set_major_locator(MultipleLocator(10))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

plt.fontsize = 300
plt.savefig('Photocurrent_vs_x_and_y.png')

plt.axhline(y=specific_y, color='red', linestyle='--', linewidth=2, label='1D Sample Position')
plt.axvline(x=specific_x, color='red', linestyle='--', linewidth=2, label='1D Sample Position')

plt.axhline(y=specific_y+9.9, color='red', linestyle='--', linewidth=2, label='1D Sample Position')
plt.axvline(x=specific_x-3.3, color='red', linestyle='--', linewidth=2, label='1D Sample Position')
plt.show()


