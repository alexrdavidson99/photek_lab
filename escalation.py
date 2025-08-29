import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mplhep
mplhep.style.use(mplhep.style.LHCb2)


escalation_data = pd.read_csv("C:/Users/lexda/local_pmt_info/characterisation/escalation_data/cathode_on_tests_LK0.csv")
escalation_data['Counts'] = escalation_data['Counts'].astype(float)
escalation_data['Seconds'] = escalation_data['Seconds'].astype(float)

plt.figure(figsize=(10, 6))
plt.scatter(escalation_data['Seconds'], escalation_data['Counts'], s=5, label='Counts vs Seconds')
plt.plot(escalation_data['Seconds'], escalation_data['Counts'], linewidth=0.5, label='Counts Trend')
plt.xlabel('Seconds')
plt.ylabel('Counts')
plt.title('Counts vs Seconds Data')
plt.show()