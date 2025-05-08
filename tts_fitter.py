import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.optimize import curve_fit
from lhcb_style import apply_lhcb_style
import mplhep
mplhep.style.use(mplhep.style.LHCb2)
plt.rcParams.update({'font.size': 10})



# Define the triple Gaussian function
def triple_gaussian(t, n, f1, f2, t1, t2, t3, sigma1, sigma2, sigma3):
    term1 = (1 - f1 - f2) / sigma1 * np.exp(-0.5 * ((t - t1) / sigma1)**2)
    term2 = f1 / sigma2 * np.exp(-0.5 * ((t - t2) / sigma2)**2)
    term3 = f2 / sigma3 * np.exp(-0.5 * ((t - t3) / sigma3)**2)
    return n / np.sqrt(2 * np.pi) * (term1 + term2 + term3)

def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def single_gaussian(t, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((t - mean) / sigma)**2)

def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def trimodal(x, mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3):
    return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2) + gauss(x, mu3, sigma3, A3)

tts_path = Path("C:/Users/lexda/local_pmt_info/characterisation/tts/tts_J_LK1_1500back_1400MCP_skew_200")
Fs = [1,2,3,5,6,7,9,10,11]
FILES_PREFIX = [f"F{i}" for i in Fs]
sig_files = list(tts_path.rglob("*-sig_*.csv"))
print(sig_files)


fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

for i, tts_data in enumerate(sig_files):
    path = tts_data.stem.split("_")
    if path[1] == "-0.70000": # -0.84000
        print("yes")
        FILES_PREFIX = [int(j.lstrip('F')) for j in path[0].split("-") if j.lstrip('F').isdigit()]
        if FILES_PREFIX[0] <= 4:
            hist_123 = pd.read_csv(tts_data, delimiter=',', names=['bins', 'count'], skiprows=5)
            ax1.plot(hist_123["bins"], hist_123["count"], label=f"{FILES_PREFIX[0]}")
            x = hist_123["bins"]
            y = hist_123["count"]
            max_y_index = y.idxmax()
            max_x_value = x[max_y_index]
            
            #expected = [-5e-9, 0.15e-9, 200, max_x_value, 0.15e-9, 300]
            expected = [-5.0e-9, 0.15e-9, 100, max_x_value, 0.15e-9, 300]
            params, cov = curve_fit(bimodal, x, y,expected)
            sigma=np.sqrt(np.diag(cov))
            x_fit = np.linspace(x.min(), x.max(), 500)
            #plot combined...
            ax1.plot(x_fit, bimodal(x_fit, *params), color='red', lw=3, label='model')
            ax1.plot(x_fit, gauss(x_fit, *params[:3]), color='red', lw=1, ls="--", label='distribution 1')
            ax1.plot(x_fit, gauss(x_fit, *params[3:]), color='red', lw=1, ls=":", label='distribution 2')

            sigma1 = params[1]
            sigma2 = params[4]

            # Calculate FWHM for each Gaussian
            fwhm1 = 2.355 * sigma1
            fwhm2 = 2.355 * sigma2

            print(f"FWHM of distribution 1: {fwhm1:.3e}")
            print(f"FWHM of distribution 2: {fwhm2:.3e}")

            print(FILES_PREFIX[0])
            
            print(pd.DataFrame(data={'params': params, 'sigma': sigma}, index=bimodal.__code__.co_varnames[1:]))
            A1 = params[2]
            sigma1 = params[1]
            A2 = params[5]
            sigma2 = params[4]

            # Calculate the overall RMS
            overall_rms = np.sqrt((A1 * sigma1**2 + A2 * sigma2**2) / (A1 + A2))

            print(f"Overall RMS width: {overall_rms:.3e}")

            None
        
        if 5 <= FILES_PREFIX[0] <= 7:
            area_567 = pd.read_csv(tts_data, delimiter=',', names=['entry', 'area_values'], skiprows=5)
            ax2.hist(area_567["area_values"], bins=200, alpha=0.5, label=f"{FILES_PREFIX[0]}")

        if 9 == FILES_PREFIX[0]:
            sq_91011 = pd.read_csv(tts_data, delimiter=',', names=['entry', 'sq_values'], skiprows=5)
            filtered_sq_values = sq_91011[sq_91011["sq_values"].between(-5.4e-9, -4.5e-9, inclusive="both")]
            print(filtered_sq_values["sq_values"].describe())
            y,x,_ = ax3.hist(-filtered_sq_values["sq_values"], bins=250, alpha=0.5, label=f"binned data")
            x=(x[1:]+x[:-1])/2 
            max_y_index =  np.argmax(y)
            max_x_value = x[max_y_index]
            max_y_value = y[max_y_index]

         
    
            expected = [2.6e-9, 0.05e-9, 500, 2.7e-9, 0.05e-9, max_y_value, 2.8e-9, 0.05e-9, 100]
            expected = [600, 0.3, 0.2, 2.6e-9, 2.8e-9, 3.0e-9, 0.05e-9, 0.1e-9, 0.15e-9]
            expected = [600, 0.3, 0.2, 4.8e-9, 4.7e-9, 5.0e-9, 0.05e-9, 0.1e-9, 0.15e-9]
   

            #expected = [500, 2.6e-9, 0.05e-9, 200, 2.7e-9, 0.05e-9,  100, 2.8e-9, 0.05e-9] 
            #params, cov = curve_fit(bimodal, x, y,expected)
            params, cov = curve_fit(triple_gaussian, x, y, p0=expected, )
            sigma=np.sqrt(np.diag(cov))
            x_fit = np.linspace(x.min(), x.max(), 500)
            #plot combined...

            A1, sigma1, A2, sigma2 = params[2], params[1], params[5], params[4]
    
            # Calculate the overall RMS width
            overall_rms = np.sqrt((A1 * sigma1**2 + A2 * sigma2**2) / (A1 + A2))
            


            n, f1, f2, t1, t2, t3, sigma1, sigma2, sigma3 = params
            amplitude1 = n * (1 - f1 - f2) / (sigma1 * np.sqrt(2 * np.pi))
            amplitude2 = n * f1 / (sigma2 * np.sqrt(2 * np.pi))
            amplitude3 = n * f2 / (sigma3 * np.sqrt(2 * np.pi))
            print(f"sigma1: {sigma1:.3e}")
            print(f"sigma2: {sigma2:.3e}")
            print(f"sigma3: {sigma3:.3e}")
            print(f"mean1: {t1:.3e}")
            print(f"mean2: {t2:.3e}")
            print(f"mean3: {t3:.3e}")
            print(f"Overall RMS width: {overall_rms:.3e}")

            # Plot each Gaussian component
            ax3.plot(x_fit, triple_gaussian(x_fit, *params), color='red', lw=3, label='Triple Gaussian Fit')
            ax3.plot(x_fit, single_gaussian(x_fit, amplitude1, t1, sigma1), color='red', lw=1, ls="--", label=fr'first $\sigma$ {sigma1:.3e}')
            ax3.plot(x_fit, single_gaussian(x_fit, amplitude2, t2, sigma2), color='blue', lw=1, ls=":", label=fr'second $\sigma$ {abs(sigma2):.3e}')
            ax3.plot(x_fit, single_gaussian(x_fit, amplitude3, t3, sigma3), color='green', lw=1, ls="-.", label=fr'third $\sigma$  {abs(sigma3):.3e}')

            
            #ax3.plot(x_fit, gauss(x_fit, *params[:3]), color='red', lw=1, ls="--", label='distribution 1')
            #ax3.plot(x_fit, gauss(x_fit, *params[3:6]), color='red', lw=1, ls=":", label='distribution 2', )
            #ax3.plot(x_fit, gauss(x_fit, *params[6:]), color='green', lw=1, ls=":", label='distribution 3', )
            
        
        
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



ax3.set_title('Binned TTS data 200V on cathode')
ax3.set_xlabel('time (ps)', fontsize=25, loc='left')
ax3.set_ylabel('counts', fontsize=25)
ax3.legend()


plt.figure(figsize=(8.27, 5.5))
plt.rcParams.update({'font.size': 11})
voltages = [150]
for i in voltages: 
    #pd_data = pd.read_csv(f"C:/Users/lexda/local_pmt_info/characterisation/tts/dark-box-2-tts-data/500v_back/first_sweep_cathode_voltage/F9--{i}v-cathode-1500--mcp-high-counts-with-photo-d-hb-10k-00000--00000.csv" 
    #                    ,names= ["time","amp"], skiprows=5 ) 
    #pd_data["time"] = -pd_data["time"]*1e9
    pd_data = pd.read_csv(f"C:/Users/lexda/local_pmt_info/characterisation/tts/dark-box-2-tts-data/500v_back/first_sweep_cathode_voltage/F9--{i}v-cathode-1450-mcp-with-photo-d-hb-10k-00000--00000.csv" 
                        ,names= ["time","amp"], skiprows=5 ) 
    pd_data["time"] = -pd_data["time"]*1e9
   
    plt.plot(pd_data["time"], pd_data["amp"])
    #error, mean, amplitude
    expected = [0.1, -0.2, 275]
  
    expected = [0.05, -0.2, 275, 0.05, -0.1, 275]
    
    params_b, cov_b = curve_fit(bimodal, pd_data['time'], pd_data['amp'] )
    A1, sigma1, A2, sigma2 = params_b[2], params_b[1], params_b[5], params_b[4]
    rms = np.sqrt((A1 * sigma1**2 + A2 * sigma2**2) / (A1 + A2))
    mean_bimodal = (A1 * params_b[0] + A2 * params_b[3]) / (A1 + A2)
    mean_data = np.average(pd_data["time"], weights=pd_data["amp"])
    rms_full = np.sqrt(
        np.sum(pd_data['amp'] * (pd_data['time'] - mean_data)**2)
        / pd_data['amp'].sum()
    )
    print(f"rms_full {rms_full*1e3:.3f} ps")
    print(f"rms {rms*1e3:.3f} ps")
    print(f"printed prams {A1, sigma1, A2, sigma2}")
    print(f"FWHM {sigma1*2.355*1e3:.3f} ps")
    sigma=np.sqrt(np.diag(cov))
    print(f"Fwhm: {2.3548*params[1]:.3e}")
    print(f"printed prams {params}")
    x_fit = np.linspace(pd_data['time'].min(), pd_data['time'].max(), 500)
    plt.plot(x_fit, bimodal(x_fit, *params_b), color='red', lw=3, label=fr'Double Gaussian Fit' )
    #plt.plot(x_fit, gauss(x_fit, *params), color='red', lw=1, ls="--", label='distribution 1')
    #plt.yscale('log')
    #plt.ylim(1e-1,1e3)
    plt.xlim(-0.7,0.8)
    plt.plot(x_fit, gauss(x_fit, *params_b[:3]), color='red', lw=1, ls="--")  #label=fr'Primary  $\sigma_{{\text{{TTS}}}}$ {sigma1*1e3:.0f} ps')
    plt.plot(x_fit, gauss(x_fit, *params_b[3:]), color='green', lw=1, ls=":") 
    plt.plot([], [], ' ', label=Fr"$\sigma_{{\text{{RMS}}}}$: {rms_full*1e3:.0f} ps")
    plt.plot([], [], ' ', label=Fr"$\sigma_{{\text{{TTS}}}}$ {sigma1*1e3:.0f} ps")

    
plt.title(f'{i}V Across Cathode-MCP Gap', fontsize=22)
plt.xlabel("Time [ns]",fontsize=30)
plt.ylabel("counts",fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=18) 
plt.legend(fontsize=25)
#plt.tight_layout()
plt.savefig(f"C:/Users/lexda/local_pmt_info/characterisation/tts/tts_data_cathode{i}v.pdf")
plt.show()



