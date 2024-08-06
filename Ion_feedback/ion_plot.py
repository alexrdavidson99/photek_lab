import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.constants


def solve_for_intercept_time(x0, v0, acc, target_distance):
    '''
    Solve for the intercept time when the particle reaches the target distance in the y direction.
    '''
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
    '''
    Position after time step
    '''
    x0 = np.array(x0)
    v0 = np.array(v0)
    acc = np.array(acc)
    return x0 + v0 * time + 0.5 * acc * time ** 2

def electron_trajectory(target_distance, q_t, mass, initial_energy, theta_deg=8, dt=1e-10):
    """
    Simulates the trajectory of an electron (or ion) in an electric field until it reaches the target distance in the z-axis.
    
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
    v_i = np.sqrt(initial_energy/mass)*c  # velocity in mm/ns

    # Convert angle to radians
    theta = np.deg2rad(theta_deg)
    cathode_field = 200/ target_distance
    # Calculate acceleration components
    a_y = cathode_field * q_t

    # Initialize lists for time, position, and velocity
    t = []
    x = []
    y = []
    z = []
    vx = [v_i * np.sin(theta)]
    vy = [v_i * np.cos(theta)]
    vz = [0]
    v_abs = np.sqrt(vx[0]**2 + vy[0]**2 + vz[0]**2)
    print(f"magitude of velocity {v_abs} m/s with energy in is {v_i} ev")

    # Solve for intercept time
    intercept_time = solve_for_intercept_time([0, 0], [vx[0], vy[0]], [a_x, a_y], target_distance)
    print(f"Time to intercept target {intercept_time*1e9} ns")

    for i in range(100):
       if i == 0:
           x.append(0)
           y.append(0)
           xi = step_position([0, 0], [vx[0], vy[0]], [a_x, a_y], intercept_time)
       xj = step_position(xi, [vx[0], vy[0]], [a_x, a_y], intercept_time / 1000)
       print(f"xj {xj} ")
       xi = xj
       t.append(i * intercept_time / 100)
       x.append(xj[0])
       y.append(xj[1])

    t = np.array(t)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # Plot the trajectory of the electron in 3D
    
    return x, y, z, t

# Example usage:
  # Initial kinetic energy in Joules
target_distance = 0.5e-3  # Target distance in meters (2 mm)
Q_t = 1.76e11  # Mass of electron in kg


initial_energy =  1000 # in eV
Q_m = 9.58e7
ions = [Q_m,Q_m, Q_m / 5, Q_m / 8, Q_m / 10, Q_m / 12, Q_m / 18,  Q_m /2000]
ion_mass_in_ev = [511e3,938.272e6, 938.272e6*5, 7485.3e6, 9370.4e6, 11244.5e6, 16866.8e6,  938.272e6*2000]


for mass, charge_mass_ratio in zip(ion_mass_in_ev, ions):
    x, y, z, t = electron_trajectory(target_distance,charge_mass_ratio, mass, initial_energy)
    print(mass)
    print(t)
    plt.plot(x, y, label=f'Mass {mass/1e6:.1f} MeV/c^2, time: {t[-1]/1e-9:.4f} ')

# Labeling the Axes
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')

# Title and Legend
plt.title('Electron Trajectories for Different Ion Masses')
plt.legend()

# Show plot
plt.show()
#t = electron_trajectory(target_distance, Q_t,m, initial_energy)
#print(t)