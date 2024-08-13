import pandas as pd
from scipy.signal import find_peaks
import scipy
from itertools import cycle
import numpy as np
#from ion_plot import electron_trajectory
import matplotlib.pyplot as plt


def step_energy( v0, acc, time,m):
    '''
    Energy after time step starting from emission velocity (ev)
    938.272 MeV/c2, electron mass is 511 keV/c2
    '''
    #m = 511e3 # in eV
    c = scipy.constants.speed_of_light 
    return 0.5*m*(np.dot(v0 + acc*time, v0 + acc*time))/(c**2)

def calculate_time(anode_gap, cathode_gap, mcp_gap, q_m,m):
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
    mcp_field = 1000 / mcp_gap
    print (F"mcp field {mcp_field}")

    # Acceleration due to the electric field (m/s^2)
    cathode_acc = q_m * cathode_field
    e_cathode_acc = q_m_electron * cathode_field
    anode_acc = anode_field * q_m_electron
    e_mcp_acc = mcp_field* q_m_electron
    mcp_acc = mcp_field* q_m

    # Time to travel the distance (s)
    time_in_mcp = np.sqrt((2 * mcp_gap) / mcp_acc)
    print (F"time in mcp {time_in_mcp*1e9} ns")


    energy = step_energy(0, mcp_acc, time_in_mcp,m)
    #t = electron_trajectory(mcp_gap, q_m, m, energy)
    #print (F"tof ion mcp {t} s")
    
    print (F"energy in mcp {energy} ev")

    time_to_cathode = np.sqrt((2 * cathode_gap) / cathode_acc)
    print (F"mcp to cathode {time_to_cathode*1e9} ns")
    time_to_mcp = np.sqrt((2 * cathode_gap) / e_cathode_acc)
    print (F"cathode to mcp {time_to_mcp*1e9} ns")
    time_to_anode = np.sqrt((2 * anode_gap) / anode_acc) # electron time to anode (s)
    print (F"mcp to anode {time_to_anode}")
    time = time_to_cathode  + time_to_mcp   + 0.4e-9  + time_to_anode    # Time (seconds)

    return time, time_in_mcp

def read_data(filename):
    data = pd.read_csv(filename, comment='#', names=["1_to_10000", "events"], skiprows=6, dtype={"1_to_10000": int, "events": float})
    print(f"number captured events..{len(data['events'])}")
    return data['events'].reset_index(drop=True)
def read_data_hist(filename):
    data = pd.read_csv(filename, comment='#', names=["time", "hits"], skiprows=1, dtype={"time": float, "hits": int})
    return data['time'], data['hits']


def find_offset(data):
    hist_vals, bin_edges, _ = plt.hist(data, bins=50)
    peaks, _ = find_peaks(hist_vals, height=450)
    peak_positions = bin_edges[peaks]
    peak_heights = hist_vals[peaks]
    print(f"peaks t0 {peak_positions * 1e9} ns")
    return peak_positions, peak_heights

anode_gap = 5.15e-3
cathode_gap = 0.431e-3 # between 0.2 mm and 0.5 mm calculated gap 0.295 mm
mcp_gap = 460e-6 # 460 um photonis mcp value from datasheet
mcp_bais_angle = 8 # degrees
q_m_hydrogen = 9.58e7 

Q_m = 9.58e7
ions = [Q_m, Q_m / 5, Q_m / 8, Q_m / 10, Q_m / 12, Q_m / 18,  Q_m /56]
ion_mass_in_ev = [938.272e6, 938.272e6*5, 7485.3e6, 9370.4e6, 11244.5e6, 16866.8e6,  52667.2e6]

#divisors = list(range(1, 134))
#ions = [Q_m / d for d in divisors]

events_t0 = read_data('./Ion_feedback_data/F4--alex gas--ion-300k--00000.txt') # C3--alex gas--ion-300k--00000
print(len(events_t0))
peak_positions_t0, peak_heights_t0 = find_offset(events_t0)
plt.scatter(peak_positions_t0, peak_heights_t0, color='red', label='Data')

plt.figure(figsize=(10, 6))
plt.hist((events_t0-peak_positions_t0)*1e9, bins=300)
plt.yscale('log')

events_intital_data = read_data('./Ion_feedback_data/F4--alex gas--00000.txt')
peak_positions_t0_intial, peak_heights_t0_intial = find_offset(events_intital_data)


times, times_in_mcp = zip(*[(time * 1e9, time_in_mcp * 1e9) for time, time_in_mcp in [calculate_time(anode_gap, cathode_gap, mcp_gap, qm, mass) for qm, mass in zip(ions, ion_mass_in_ev)]])
plot_time = np.array(times/times[0])

for index, time_ns in enumerate(times_in_mcp):
    print(f"mass {ion_mass_in_ev[index]/1e6:.1f} MeV/c^2: {time_ns} ns")

ions_list = ['H+', 'He+','Li/Be+?', 'Li/Be+?', 'C+', 'H20+', 'Ga+']
colors = cycle(['black', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan'])

# Plot each ion's time as a vertical line with a unique color
for k, color in zip(range(len(times)), colors):
    plt.vlines(times[k], 0, 5000, colors=color, linestyles='solid', label=f'{ions_list[k]}')
plt.text(0.8, 0.3, 'preliminary data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

events = read_data('./Ion_feedback_data/F3--alex gas--ion-300k--00000.txt')
intial_data_events = read_data('./Ion_feedback_data/F3--alex gas--00000.txt')
plt.hist((intial_data_events-peak_positions_t0_intial)*1e9, bins=300, alpha=0.5, color='blue', label='Initial Data')
print(len(events))
hist_vals, bin_edges, _ =plt.hist((events-peak_positions_t0)*1e9, bins=300, alpha=0.5, color='red', label='Data')
plt.yscale('log')

# Find peaks in the second histogram
peaks, _ = find_peaks(hist_vals, height=400)  # You can adjust the height parameter as needed

# Extract peak positions and heights
peak_positions = bin_edges[peaks]
peak_heights = hist_vals[peaks]

# Plot peak positions and heights
plt.scatter(peak_positions, peak_heights, color='black', label='Peaks')

#plt.xlim(-10, 20)
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
plt.plot(plot_peak_positions, 'o-', color='blue' , label='data')
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

plt.figure(figsize=(10, 6))
plt.plot(ions_mass_number,times_in_mcp, 'o-', color='blue')
plt.xlabel('Ion Charge (Q_m)')
plt.ylabel('Peak Height')
plt.title('Peak Height vs. Ion Charge')
plt.savefig('peak_height_v_mass_number.pdf', dpi=300)

plt.figure(figsize=(10, 6))
x,y = read_data_hist('./Ion_feedback_data/F1--alex gas--ion-300k--fr--00000.txt')
x = x*1e9
plt.plot(x, y, 'o-', color='blue')
plt.yscale('log')
plt.xlabel('Time [ns]')
plt.ylabel('Events')
plt.title('Events vs. Time')
plt.savefig('ion_feedback_F1.pdf', dpi=300)

events_intital_data = read_data('./Ion_feedback_data/F4--alex gas--ion-10k-4.1kv--newdropper--00000.txt')
peak_positions_t0_intial, peak_heights_t0_intial = find_offset(events_intital_data)

plt.figure(figsize=(10, 6))
inital_data_events = read_data('./Ion_feedback_data/F4--alex gas--ion-10k-4.1kv--newdropper--00000.txt')
inital_data_events = read_data('./Ion_feedback_data/F4--alex gas--ion-10k-4kv--newdropper--00000.txt')
plt.hist((inital_data_events-peak_positions_t0_intial)*1e9, bins=300, alpha=0.5, color='blue', label='Initial peak')
#min_events = read_data('./Ion_feedback_data/F5--alex gas--ion-100k--with-min--00000.txt')
events = read_data('./Ion_feedback_data/F3--alex gas--ion-10k-4kv--newdropper--00000.txt')
plt.hist((events-peak_positions_t0_intial)*1e9, bins=650, alpha=0.5, color='red', label='Data_4kv')
events = read_data('./Ion_feedback_data/F3--alex gas--ion-10k-4.1kv--newdropper--00000.txt')
plt.hist((events-peak_positions_t0_intial)*1e9, bins=650, alpha=0.5, color='black', label='Data_4.1kv')
F1x ,F1y = read_data_hist('./Ion_feedback_data/F1--alex gas--ion-10k-4kv--newdropper--00000.txt')
plt.plot((F1x-peak_heights_t0_intial)*1e9, F1y, 'o-', color='green', label='F1_4kv')
#laser_off = read_data('./Ion_feedback_data/F5--alex gas--ion-10k--with-min--no--laser--00000.txt')
#print(np.mean(laser_off))


#plt.yscale('log')
plt.xlim(0, 75)
plt.xlabel('Time [ns]')
plt.ylabel('Events')
plt.legend()

plt.figure(figsize=(10, 6))
cath = [50,100,200]
for i in range(len(cath)):
    events_0 = read_data(f'./Ion_feedback_data/F4--alex gas-cath--{cath[i]}v--00000.txt')
    peak_positions_t0_intial, peak_heights_t0_intial = find_offset(events_0)
    #plt.hist((events_0-peak_positions_t0_intial)*1e9, bins=600, alpha=0.5, label=f'0...{cath[i]}v')
    after_plot = read_data(f'./Ion_feedback_data/F3--alex gas-cath--{cath[i]}v--00000.txt')
    #plt.hist((after_plot-peak_positions_t0_intial)*1e9, bins=600, alpha=0.5, label=f'afterpluse..{cath[i]}v')

laser_noise = read_data('./Ion_feedback_data/F3--alex gas-just--laser-00000--00000.txt')
plt.hist(laser_noise*1e9, bins=600, alpha=0.5, label='laser noise')
plt.ylim(0, 100)
plt.xlim(0, 75)
plt.xlabel('Time [ns]')
plt.ylabel('Events')
plt.title('Events vs. Time')
plt.legend()


plt.show()