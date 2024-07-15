import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import argparse

def mA_per_watt_to_QE(current_mA, wavelength_nm):
    # Constants
    h = const.h  # Planck's constant in J·s
    c = const.c  # Speed of light in m/s
    e = const.e  # Elementary charge in coulombs

    # Calculate the constant part of the formula
    constant = ((h * c) / e) * 1e9 * 1e-3

    # Calculate QE
    qe = (constant / wavelength_nm) * current_mA * 100
    return qe

# Import the data
parser = argparse.ArgumentParser(description='Generate a heatmap from a CSV file.')
parser.add_argument('filename', type=str, nargs='+', help='Path to the CSV file')
args = parser.parse_args()  # Parse the command-line arguments
  # Read the CSV file

#plt.figure(figsize=(, 10))
for filename in args.filename:
    data = pd.read_csv(filename)
    #file_path = 'C:/Users/lexda/Desktop/QE/QE_22240313.csv'
    #data = pd.read_csv(file_path)

    # Extracting data
    MeasuredDate = pd.to_datetime(data['MeasuredDate'])
    Sensitivity = data['Sensitivity']

    # Extract wavelengths and intensities
    wavelengths = data.columns[2:].astype(int)
    intensities = data.iloc[:, 2:].astype(float)

    # Convert intensities to QE
    QE = np.zeros_like(intensities)
    print(QE)

    for i, wavelength in enumerate(wavelengths):
        QE[:, i] = mA_per_watt_to_QE(intensities.iloc[:, i], wavelength)

    # Create a DataFrame for QE with MeasuredDate as the index
    qe_df = pd.DataFrame(QE, columns=wavelengths, index=MeasuredDate)

    print(QE)

    # Plot original intensities
    #plt.figure(figsize=(10, 6))
    #for date in data.index:
    #    plt.plot(wavelengths, intensities.loc[date], label=MeasuredDate[date].strftime('%d-%b-%y'))

    #plt.xlabel('Wavelength (nm)')
    #plt.ylabel('Intensity (mA/W)')
    #plt.title('Intensity vs. Wavelength')
    #plt.legend()
    #plt.yscale('log')
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()

    # Plot QE vs. Wavelength for each date

    for date in qe_df.index:
        print(date)
        print(qe_df.loc[date,:])
        plt.semilogy(wavelengths, qe_df.loc[date, :], label=date.strftime('%d-%b-%y'))

    # Add text box with the maximum QE for each date
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    textstr = ''
    for date in qe_df.index:
        max_qe = qe_df.loc[date].max()
        textstr += f'{date.strftime("%d-%b-%y")}: {max_qe:.2f} %\n'


    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Quantum Efficiency (%)')
    plt.title('Quantum Efficiency vs. Wavelength')
    #plt.text(0.05, 0.2, textstr, transform=plt.gca().transAxes, fontsize=14,
    #        verticalalignment='center', bbox=props)
    #plt.text(0.80, 0.9, "Preliminary", transform=plt.gca().transAxes, fontsize=16,
    #        verticalalignment='center', bbox=props)
    plt.legend()
    #plt.yscale('log')


    #plt.minorticks_on()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()

    if 400 in wavelengths:
        qe_at_400 = qe_df[400]
        print(f"QE values at 400 nm:")
        for date, qe_value in qe_at_400.iteritems():
            print(f"{pd.to_datetime(date).strftime('%d-%b-%y')}: {qe_value:.2f} %")


plt.show()
