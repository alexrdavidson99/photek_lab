import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm , Normalize
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.ticker import FuncFormatter
import mplhep
mplhep.style.use(mplhep.style.LHCb2)



# Custom tick formatting to use "p", "n", "µ"
def format_func(value, tick_number):
    exponent = np.log10(value)
    if exponent < -9:  # For values below 1n
        return f"{value*1e12:.0f}p"
    elif exponent < -6:  # For values below 1µ but >= 100n
        return f"{value*1e9:.0f}n"
    elif exponent < -3:  # For values below 1m but >= 100µ
        return f"{value*1e6:.0f}µ"
    elif exponent < 0:  # For values below 1 but >= 100m
        return f"{value*1e3:.0f}m"
    else:
        return f"{value:.0f}"
rate_cap_data = pd.read_csv("C:/Users/lexda/local_pmt_info/characterisation/rate-cap/highest_current_values.csv", delimiter=',',
                             names=["rate" , "backgrond","signal", "number of photons" ,"rate_linear", "y" ,"x_stip_current_ratio"], skiprows=1)
rate_cap_y = rate_cap_data['y']
rate_cap_x = rate_cap_data['number of photons']
plt.scatter(rate_cap_x, rate_cap_y, )
plt.plot(rate_cap_x, rate_cap_y, label='Rate vs anode current')
plt.xlabel('number of photons')
plt.ylabel('normalised gain')
plt.title('Rate vs number of photons')
plt.xscale('log')
#plt.xlim(1e-8, 0.5e-5)
plt.ylim(0, rate_cap_y.max()+0.1)
ax = plt.gca()
ax.xaxis.set_major_formatter(FuncFormatter(format_func))

# Show plot
plt.show()
