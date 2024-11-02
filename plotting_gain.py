#this sq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Generate a heatmap from a CSV file.')
parser.add_argument('filename', type=str, nargs='+', help='Path to the CSV file')
args = parser.parse_args()  # Parse the command-line arguments

for filename in args.filename:
    data = pd.read_csv(filename, comment='#', names=["voltage", "gain"], skiprows=1)
    # Read the CSV file
    voltages = data['voltage']/2
    gains = data['gain']

    plt.semilogy(voltages, gains, '.r-', color='orange')
    plt.xlabel('Voltage [V]')
    plt.ylabel('Current Gain')
    plt.title('Gain vs. Voltage')

    plt.text(0.2, 0.9, 'preliminary data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.savefig('gain_vs_voltage.png')




