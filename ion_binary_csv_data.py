import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the CSV file into a DataFrame
def read_csv_file(file_path):
    file_path_full = fr'c:\Users\lexda\PycharmProjects\Photek_lab\{file_path}.csv'
    df = pd.read_csv(file_path)
    return df


file_paths = ["ion_feedback_image_d.csv","peak_times_by_file.csv"]

df_imaged = read_csv_file(file_paths[0])
df_string = read_csv_file(file_paths[1])


first_time = []
first_time_imaged = []
time_differences = []

# Loop through both dataframes by row index
for i, row in df_imaged.iterrows():
    peak_times = row["peak_times"]
    if isinstance(peak_times, float):
        first_time_imaged.append(peak_times*1e9)

for i, row in df_string.iterrows():
    peak_times = row["peak_times"]
    if isinstance(peak_times, str):
        times = [float(t.strip()) * 1e9 for t in peak_times.split(',')]
        first_time.append(times[0])  # Append the first time from the list
            # to ns
        if len(times) == 0:
            time_differences.append(0)
        elif len(times) >= 4 and i < len(first_time_imaged):
            reference_time = first_time_imaged[i]  # from float CSV
            # Calculate differences from that external reference time
            for t in times:
                delta = t - reference_time
                time_differences.append(delta)
     

print(f"len(first_time): {len(first_time)}")
print(f"len(first_time_imaged): {len(first_time_imaged)}")
print(f"len(time_differences): {len(time_differences)}")

plt.figure()
plt.hist(time_differences, bins=100, alpha=0.5, label='Time Differences')
plt.hist(first_time_imaged, bins=100, alpha=0.5, label='First Time Imaged')
plt.hist(first_time, bins=100, alpha=0.5, label='First Time from String')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Histogram of First Times')
plt.legend()
plt.show()