import pandas as pd
from scipy.signal import find_peaks
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

def calculate_time(anode_gap, cathode_gap, q_m):
    """
    Calculate the time it takes for an electron to travel a given distance under
    constant acceleration due to an electric field.

    Parameters:
    distance (float): The distance traveled by the electron (meters).

    Returns:
    float: The time taken (seconds).
    """
    q_m_electron = 1.76e11  # Charge-to-mass ratio of the electron (C/kg)

    # Electric field strengths (V/m)
    anode_field = 1900/anode_gap
    cathode_field = 200 / cathode_gap

    # Acceleration due to the electric field (m/s^2)
    cathode_acc = q_m * cathode_field
    e_cathode_acc = q_m_electron * cathode_field
    anode_acc = anode_field * q_m_electron

    # Time to travel the distance (s)
    time_to_cathode = np.sqrt((2 * cathode_gap) / cathode_acc)
    time_to_mcp = np.sqrt((2 * cathode_gap) / e_cathode_acc)
    time_to_anode = np.sqrt((2 * anode_gap) / anode_acc) # electron time to anode (s)
    print (F"mcp to anode {time_to_anode}")
    time = time_to_cathode  + time_to_mcp   + 0.4e-9  + time_to_anode    # Time (seconds)

    return time


anode_gap = 5.15e-3
cathode_gap = 0.5e-3 # between 0.2 mm and 0.5 mm calculated gap 0.295 mm
q_m_hydrogen = 9.58e7

Q_m = 9.58e7
ions = [Q_m, Q_m / 5, Q_m / 8, Q_m / 10, Q_m / 12, Q_m / 18,  Q_m / 132]

#divisors = list(range(1, 134))
#ions = [Q_m / d for d in divisors]

times = [calculate_time(anode_gap, cathode_gap, i) * 1e9 for i in ions]
plot_time = np.array(times/times[0])

for index, time_ns in enumerate(times):
    print(f"{index + 1}: {time_ns} ns")

ions_list = ['H+', 'He+','Li/Be+?', 'Li/Be+?', 'C+', 'H20+', 'Ga+']
colors = cycle(['black', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan'])

# Plot each ion's time as a vertical line with a unique color
#for k, color in zip(range(len(times)), colors):
#    plt.vlines(times[k], 0, 5000, colors=color, linestyles='solid', label=f'{ions_list[k]}')

data_t0 = pd.read_csv('C:/Users/lexda/Desktop/ion_feedback/F4--alex gas--00000.txt', comment='#', names=["1_to_10000", "events"], skiprows=1)
events_t0 = data_t0['events']
print(len(events_t0))

plt.hist((events_t0-7.4e-9)*1e9, bins=200)
plt.text(0.8, 0.3, 'preliminary data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

data   = pd.read_csv('C:/Users/lexda/Desktop/ion_feedback/F3--alex gas--00000.txt', comment='#', names=["1_to_10000", "events"], skiprows=1)
events = data['events']
print(len(events))
hist_vals, bin_edges, _ =plt.hist((events-7.4e-9)*1e9, bins=300)
plt.yscale('log')

# Find peaks in the second histogram
peaks, _ = find_peaks(hist_vals, height=400)  # You can adjust the height parameter as needed

# Extract peak positions and heights
peak_positions = bin_edges[peaks]
peak_heights = hist_vals[peaks]

# Plot peak positions and heights

plt.xlim(-10, 80)
#plt.ylim(0, 2000)
plt.xlabel('Time [ns]')
plt.ylabel('Events')
plt.title('Events vs. Time')
plt.legend()
plt.savefig('ion_feedback.pdf', dpi=300)


nom_peaks = peak_positions/peak_positions[0]
ions_mass_number = [1, 5, 8, 10, 12, 18, 70]

# Plot the peak positions/initial peak as a function of the ion mass number
plt.figure(figsize=(10, 6))
plot_peak_positions = peak_positions/peak_positions[0]
plt.plot(ions_mass_number,plot_peak_positions, 'o-', color='blue' , label='data')
plt.xlabel('Ion Charge (Q_m)')
plt.ylabel('Time [ns]')
plt.title('Time of peak vs. Ion Charge')

# Plot the time guess/initial time as a function of the ion mass number
plt.plot(ions_mass_number,plot_time, 'o-', color='orange' , label='Python guess')
plt.xlabel('Ion Charge mass ratio (Q_m)')
plt.ylabel('Time/time_h')
plt.title('Time ratio from initial hydrogen peak  to other peaks vs. Ion Charge mass ratio')
plt.legend()

plt.savefig('norm_peaks_v_mass_number.pdf', dpi=300)
