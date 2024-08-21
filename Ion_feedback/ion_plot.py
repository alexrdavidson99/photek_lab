import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import pandas as pd

def solve_for_intercept_time(x0, v0, acc, target_distance):
    """
    Solve for the intercept time when the particle reaches the target distance in the y direction.
    """
    # Define the polynomial coefficients for the equation of motion in the y direction
    coeffs = [
        0.5 * acc[1],  # t^2 term
        v0[1],  # t term
        x0[1] - target_distance  # constant term
    ]

    # Solve the quadratic equation
    roots = np.roots(coeffs)

    # Filter out complex roots and negative times
    real_roots = roots[np.isreal(roots) & (roots >= 0)]

    if len(real_roots) == 0:
        raise ValueError("No valid intercept time found.")

    # Return the smallest positive real root
    return np.min(real_roots)


def step_position(x0, v0, acc, time):
    """
    Position after time step
    """
    x0 = np.array(x0)
    v0 = np.array(v0)
    acc = np.array(acc)
    return x0 + v0 * time + 0.5 * acc * time ** 2


def rotate_coordinates(x, y, theta):
    """
    Rotate coordinates
    """
    angle_degrees = - theta
    # Convert the angle to radians
    angle_radians = np.deg2rad(angle_degrees)

    # Create the rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    coordinates = np.stack((x, y))
    # Apply the rotation matrix to the coordinates
    rotated_coordinates = np.dot(rotation_matrix, coordinates)
    x_rotate, y_rotate = rotated_coordinates

    return x_rotate, y_rotate


def electron_trajectory(target_distance, q_t, mass, initial_energy, theta_deg, voltage):
    """
    Simulates the trajectory of an electron (or ion) in an electric field until it reaches the target distance in
    the z-axis.
    
    Parameters:
    - initial_energy: Initial kinetic energy in Joules
    - target_distance: Target distance in meters for the z-axis
    - mass: Mass of the particle in kg
    - theta_deg: Electric field angle in degrees (default 8)
    - dt: Time step in seconds (default 1e-12)
    
    Returns:
    - t: Time array
    - x, y, z: Position arrays
    """
    c = scipy.constants.speed_of_light
    print(f"mass {mass} eV")
    v_i = np.sqrt((2*initial_energy)/mass)*c
    print(f"initial velocity {v_i/c} m/s")  # velocity in mm/ns

    theta = np.deg2rad(theta_deg)
    cathode_field = voltage / target_distance
    # Calculate acceleration components
    a_y = cathode_field * q_t
    print(f"acceleration in y direction {a_y} m/s^2")

    # Initialize lists for time, position, and velocity

    x = []
    y = []

    vx = [v_i*np.sin(theta)]
    vy = [v_i * np.cos(theta)]

    # Solve for intercept time
    intercept_time = solve_for_intercept_time([0, 0], [vx[0], vy[0]], [0, a_y], target_distance)
    t = [intercept_time]
    print(f"Time to intercept target {intercept_time*1e9} ns")
    x_end = step_position([0, 0], [vx[0], vy[0]], [0, a_y], intercept_time)
    print(f"end position {x_end}")
    dt = 1000
    # Simulate the trajectory
    for i in range(dt):
        x_end = step_position([0, 0], [vx[0], vy[0]], [0, a_y], intercept_time*(i/dt))
        x.append(x_end[0])
        y.append(x_end[1])
    return x, y, t


# Parameters
target_distance = 431e-6  # Target distance in meters
Q_t = 1.76e11  # Mass of electron in kg
initial_energies = [0, 700,1400]  # in eV
Q_m = 9.58e7
ions = [1.76e11, Q_m/2, Q_m / 5, Q_m / 8, Q_m / 10, Q_m / 15]
ion_mass_in_ev = [511e3, 938.272e6*2, 938.272e6*5, 938.272e6*8, 938.272e6*10, 938.272e6*15]
norm_ion_mass = [mass / 938.272e6 for mass in ion_mass_in_ev]
voltages = 200
times = {energy: [] for energy in initial_energies}


plt.figure(figsize=(16, 8))
for mass, charge_mass_ratio in zip(ion_mass_in_ev, ions):
    for initial_energy in initial_energies:
        t = 0
        xi, yi, ti = electron_trajectory(target_distance, charge_mass_ratio, mass, initial_energy, 8, voltages)
        if initial_energy == 700:
            x, y, t = electron_trajectory(460e-6, charge_mass_ratio, mass, 0,
                                          90, 700)
            x_rotated, y_rotated = rotate_coordinates(x, y, 8)
            x_rotated = np.array(x_rotated-x_rotated[-1]) * 1e3
            y_rotated = np.array(y_rotated-460e-6) * 1e3

            plt.plot(x_rotated, y_rotated, label=f'{mass / 1e6:.1f} MeV/c^2, t_o_f: {t[-1] / 1e-9:.4f},'
                                                 f' int_en: {initial_energy} ev')
            t = t[-1]

        xi = np.array(xi) * 1e3
        yi = np.array(yi) * 1e3
        total_t = ti[-1] + t
        print(f"total time {total_t*1e9} ns")
        plt.plot(xi, yi, label=f'{mass / 1e6:.1f} MeV/c^2, t_o_f: {ti[-1] / 1e-9:.4f}'
                               f', int_en: {initial_energy} ev')
        times[initial_energy].append(total_t * 1e9)


plt.xlabel('x position (mm)')
plt.ylabel('y position (mm)')


# Title and Legend
plt.title('Electron Trajectories for Different Ion Masses,\n coming out at a 8 degree angle with 700ev')
plt.legend()
plt.figure(figsize=(16, 8))
colors = ['b', 'g', 'r']  # colors for the different energy levels
for energy, color in zip(initial_energies, colors):
    print(times[energy])
    if energy == 700:
        plt.plot(norm_ion_mass, times[energy], 'o-', color=color, label=f'{energy}  eV,'
                                                                        f' starting from bottom of the pore')
    else:
        plt.plot(norm_ion_mass, times[energy], 'o-', color=color, label=f'{energy}  eV')
plt.xlabel('Mass number')
plt.ylabel('Time [ns]')
plt.title('Time of Flight vs. Mass Number')
plt.legend()

plt.show()

paper_data = pd.read_csv('Ion_feedback_data/paper/mass_time_table.txt', sep=',', names=['Source', 'Ion',
                                        'Mass [kg]', 'Calculated time [μs]', 'Corresponding after-pulse group',
                                        'Average measured time [μs]'], skiprows=1)

calculated_times = np.array(paper_data['Calculated time [μs]'])

# Perform the division of the DataFrame by the last value in the 'Calculated time [μs]' column
paper_data_normalized = paper_data.copy()  # Create a copy to avoid modifying the original data
paper_data_normalized['Calculated time [μs]'] = paper_data['Calculated time [μs]'] / calculated_times[1]

print(paper_data_normalized['Calculated time [μs]'])