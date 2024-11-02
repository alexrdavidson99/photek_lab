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

parser = argparse.ArgumentParser(description='Plotting QE from data bace, example C:/Users/lexda/Desktop/QE/QE_22240313.csv')
parser.add_argument('filename', type=str, nargs='+', help='Path to the CSV file')
args = parser.parse_args()  # Parse the command-line arguments

def read_data(filename):
    data = pd.read_csv(filename, sep='\t', encoding='utf-8')
    # Extracting data
    MeasuredDate = pd.to_datetime(data['MeasuredDate'])
    Sensitivity = data['Sensitivity']

    # Extract wavelengths and intensities
    wavelengths = data.columns[2:].astype(int)
    intensities = data.iloc[:, 2:].astype(float)

    # Convert intensities to QE
    QE = np.zeros_like(intensities)
    for i, wavelength in enumerate(wavelengths):
        QE[:, i] = mA_per_watt_to_QE(intensities.iloc[:, i], wavelength)

    qe_df = pd.DataFrame(QE, columns=wavelengths, index=MeasuredDate)
    return qe_df, wavelengths

for filename in args.filename:
    old_filename = "C:/Users/lexda/Desktop/QE/QE_A1240515.txt"
    qe_df, wavelengths = read_data(old_filename)



    for date in qe_df.index:

        #Plot QE vs. Wavelength for each date
        plt.plot(wavelengths, qe_df.loc[date, :], label=date.strftime('%d-%b-%y'))
        #plt.semilogy(wavelengths, qe_df.loc[date, :], label=date.strftime('%d-%b-%y'))

    qe_df, wavelengths = read_data(filename)
    # Add text box with the maximum QE for each date
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    textstr = ''
    #for date in qe_df.index:
    #    max_qe = qe_df.loc[date].max()
    #    textstr += f'{date.strftime("%d-%b-%y")}: {max_qe:.2f} %\n'

    qe_df['Day'] = qe_df.index.to_period('D')
    grouped_qe = qe_df.groupby('Day')
    max_qe = grouped_qe.max()
    min_qe = grouped_qe.min()

    for day in max_qe.index:

        plt.fill_between(wavelengths, min_qe.loc[day, wavelengths], max_qe.loc[day, wavelengths], label=f'{day.strftime("%d-%b-%y")}', alpha=0.35)

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Quantum Efficiency (%)')
    plt.title('Quantum Efficiency vs. Wavelength')
   # plt.text(0.05, 0.2, textstr, transform=plt.gca().transAxes, fontsize=14,
    #        verticalalignment='center', bbox=props)
    plt.text(0.80, 0.9, "Preliminary", transform=plt.gca().transAxes, fontsize=16,
            verticalalignment='center', bbox=props)
    plt.legend()
    #plt.yscale('log')


    plt.minorticks_on()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()

    if 400 in wavelengths:
        qe_at_400 = qe_df[400]
        #print(f"QE values at 400 nm:")
        #for date, qe_value in qe_at_400.iteritems():
            # print(f"{pd.to_datetime(date).strftime('%d-%b-%y')}: {qe_value:.2f} %")


plt.show()
