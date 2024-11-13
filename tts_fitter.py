import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.optimize import curve_fit
from lhcb_style import apply_lhcb_style

def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)


tts_path = Path("C:/Users/lexda/local_pmt_info/characterisation/tts/tts_J_LK1_1500back_1400MCP_skew_300_V_cathode")
Fs = [1,2,3,5,6,7,9,10,11]
FILES_PREFIX = [f"F{i}" for i in Fs]
sig_files = list(tts_path.rglob("*-sig_*.csv"))


fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

for i, tts_data in enumerate(sig_files):
    path = tts_data.stem.split("_")
    if path[1] == "-0.56000": # -0.84000
        FILES_PREFIX = [int(j.lstrip('F')) for j in path[0].split("-") if j.lstrip('F').isdigit()]
        if FILES_PREFIX[0] <= 4:
            # hist_123 = pd.read_csv(tts_data, delimiter=',', names=['bins', 'count'], skiprows=5)
            # ax1.plot(hist_123["bins"], hist_123["count"], label=f"{FILES_PREFIX[0]}")
            # x = hist_123["bins"]
            # y = hist_123["count"]
            # max_y_index = y.idxmax()
            # max_x_value = x[max_y_index]
            
            # #expected = [-5e-9, 0.15e-9, 200, max_x_value, 0.15e-9, 300]
            # expected = [-5.0e-9, 0.15e-9, 100, max_x_value, 0.15e-9, 300]
            # params, cov = curve_fit(bimodal, x, y,expected)
            # sigma=np.sqrt(np.diag(cov))
            # x_fit = np.linspace(x.min(), x.max(), 500)
            # #plot combined...
            # ax1.plot(x_fit, bimodal(x_fit, *params), color='red', lw=3, label='model')
            # ax1.plot(x_fit, gauss(x_fit, *params[:3]), color='red', lw=1, ls="--", label='distribution 1')
            # ax1.plot(x_fit, gauss(x_fit, *params[3:]), color='red', lw=1, ls=":", label='distribution 2')

            # sigma1 = params[1]
            # sigma2 = params[4]

            # # Calculate FWHM for each Gaussian
            # fwhm1 = 2.355 * sigma1
            # fwhm2 = 2.355 * sigma2

            # print(f"FWHM of distribution 1: {fwhm1:.3e}")
            # print(f"FWHM of distribution 2: {fwhm2:.3e}")

            # print(FILES_PREFIX[0])
            
            # print(pd.DataFrame(data={'params': params, 'sigma': sigma}, index=bimodal.__code__.co_varnames[1:]))
            # A1 = params[2]
            # sigma1 = params[1]
            # A2 = params[5]
            # sigma2 = params[4]

            # # Calculate the overall RMS
            # overall_rms = np.sqrt((A1 * sigma1**2 + A2 * sigma2**2) / (A1 + A2))

            # print(f"Overall RMS width: {overall_rms:.3e}")

            None
        
        if 5 <= FILES_PREFIX[0] <= 7:
            area_567 = pd.read_csv(tts_data, delimiter=',', names=['entry', 'area_values'], skiprows=5)
            ax2.hist(area_567["area_values"], bins=200, alpha=0.5, label=f"{FILES_PREFIX[0]}")

        if 9 == FILES_PREFIX[0]:
            sq_91011 = pd.read_csv(tts_data, delimiter=',', names=['entry', 'sq_values'], skiprows=5)
            filtered_sq_values = sq_91011[sq_91011["sq_values"].between(-3.3e-9, -2.2e-9, inclusive="both")]
            print(filtered_sq_values["sq_values"].describe())
            y,x,_ = ax3.hist(-filtered_sq_values["sq_values"], bins=250, alpha=0.5, label=f"{FILES_PREFIX[0]}")
            x=(x[1:]+x[:-1])/2 
            max_y_index =  np.argmax(y)
            max_x_value = x[max_y_index]
            max_y_value = y[max_y_index]
    
            expected = [2.8e-9, 0.15e-9, 100, max_x_value, 0.15e-9, max_y_value]
    
            params, cov = curve_fit(bimodal, x, y,expected)
            sigma=np.sqrt(np.diag(cov))
            x_fit = np.linspace(x.min(), x.max(), 500)
            #plot combined...

            A1, sigma1, A2, sigma2 = params[2], params[1], params[5], params[4]
    
            # Calculate the overall RMS width
            overall_rms = np.sqrt((A1 * sigma1**2 + A2 * sigma2**2) / (A1 + A2))
            print(f"sigma1: {sigma1:.3e}")
            print(f"sigma2: {sigma2:.3e}")
            print(f"Overall RMS width: {overall_rms:.3e}")

            ax3.plot(x_fit, bimodal(x_fit, *params), color='red', lw=3, label='model')
            ax3.plot(x_fit, gauss(x_fit, *params[:3]), color='red', lw=1, ls="--", label='distribution 1')
            ax3.plot(x_fit, gauss(x_fit, *params[3:]), color='red', lw=1, ls=":", label='distribution 2')
            apply_lhcb_style()
        
        
    None 
ax1.set_title('Histogram for F1 to F4')
ax1.set_xlabel('Bins')

ax1.set_ylabel('Count')
ax1.legend()


ax2.set_title('Histogram for F5 to F7')
ax2.set_xlabel('Entry')
ax2.set_ylabel('Area Values')
ax2.set_yscale('log')
ax2.legend()



ax3.set_title('Histogram for F8 and above')
ax3.set_xlabel('time (ps)')
ax3.set_ylabel('counts')
ax3.legend()
apply_lhcb_style()

plt.show()
