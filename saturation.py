import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
#from scipy.integrate import simps,trapezoid
import mplhep
mplhep.style.use(mplhep.style.LHCb2)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib.ticker import FuncFormatter

# Define the model: I0 = I * exp[ C * (I/A)^B / (1 - I/A)^(1/4) ]
def model_I0(I, A, B, C):
    return I * np.exp(C * ((I / A)**B) / ((1 - I / A)**(1/4)))



#import matplotlib.pyplot as plt
# Variables
folder_path = Path("C:/Users/lexda/local_pmt_info/rate-cap/Gain_results_A118_&_A124")
plt.figure(figsize=(10, 6))
for file in folder_path.glob("*CSV*.csv"):
    split_name = file.name.split("-")
    print(split_name)
    
    gain_data = pd.read_csv(file, sep=',', comment="#" ,skiprows=10)

    print(gain_data.head())
    print(gain_data.columns)
    gain_column_name = 'GAIN.1'
    voltage_column_name = 'Voltage (V)'  # Get the name of the 11th column
    gain_values = gain_data[gain_column_name][:14].astype(float).to_numpy()  # Convert to float and numpy array
    voltage_values = gain_data[voltage_column_name][:14].astype(float).to_numpy()

    if "214" in split_name[1]:
        # use pandas to save to csv
        print("Saving gain data for A124...")
        df = pd.DataFrame({'Voltage (V)': voltage_values, 'Gain': gain_values})
        df.to_csv("C:/Users/lexda/local_pmt_info/rate-cap/Gain_results_A118_&_A124/exp_sn_124_gain_data.csv", index=False)

    plt.scatter(voltage_values[:14], gain_values[:14], s=5, label=f'{split_name[1]}')  
    plt.plot(voltage_values[:14], gain_values[:14], linewidth=0.5 )  
    print(gain_values[:14])
    print(voltage_values[:14])
    plt.xlabel('Voltage (V)')
    plt.ylabel('Gain')
    plt.title('Gain vs Voltage with mask 0.326 cm2 ')
    plt.xlim(1000, 2000)
    plt.yscale('log')
    plt.legend()
    plt.grid()


#Define the folder path
folder_paths = [
    #Path("C:/Users/lexda/local_pmt_info/rate-cap/Saturation_A1181001/"),
    Path("C:/Users/lexda/local_pmt_info/rate-cap/Saturation_A1240515/")
]
plt.figure(figsize=(10, 6))
# Loop through all CSV files in the folder
for folder_path in folder_paths:
    #color = 'blue' if folder_path.name == "Saturation_A1181001" else 'red'
    for file in folder_path.glob("*MCP_1*.csv"):

        rate_cap_data = pd.read_csv(file, delimiter=',',
                                    names=["s_rate" , "rate","MCP_voltage","gain" ,"background" ,"signal"], skiprows=0)

        mask_area = 0.326  # cm^2
        electron_charge = 1.602e-19  # C
        power_supply_current = 175e-6  # A
        
        gain = rate_cap_data['gain']
        I = (rate_cap_data['signal'] - rate_cap_data['background']) *1e-9 /0.328  # Convert to A
        
        print(len(I))

        photon_rate = (I/ (gain * electron_charge))/mask_area  # Convert to photon rate
        photon_rate_linear = photon_rate[0]*rate_cap_data['rate']/rate_cap_data['rate'][0] 



        current_linear = (rate_cap_data['rate']*I[0])/rate_cap_data['rate'][0]
        #current_linear[0] = sig_sub_back_A[0]   # Convert to linear current
         
        photon_rate_diff = photon_rate / photon_rate_linear
        


        mcp_voltage = rate_cap_data['MCP_voltage'][0]
        #plt.scatter(photon_rate_linear, photon_rate_diff, s=5, label=fr'MCP voltage = ${int(mcp_voltage)}\,\mathrm{{V}}$')
        #plt.plot(photon_rate_linear, photon_rate_diff, linewidth=0.5 )

        plt.scatter(I, photon_rate_diff, s=5, label=fr'MCP voltage = ${int(mcp_voltage)}\,\mathrm{{V}}$')
        plt.plot(I, photon_rate_diff, linewidth=0.5 )
        #plt.scatter(current_linear, I, s=2,  label=f'mask area = {mask_area} cm2') 
          # Initial guess for A, B, C
        # Fit the model to data
        params, covariance = curve_fit(model_I0,current_linear, I, p0=[1e-5, 1.5, 2], bounds=(0, [1e-4, 2, 2]))  

        # Extract fitted parameters
        A_fit, B_fit, C_fit = params
        print("Fitted parameters:")
        print(f"A = {A_fit:.4f}, B = {B_fit:.4f}, C = {C_fit:.4f}")

        # Generate smooth curve for plotting
        I_fit = np.linspace(min(I), max(I)+5e-6, 5000)
        I0_fit = model_I0(I_fit, A_fit, B_fit, C_fit)

        #plt.plot(I0_fit, I_fit, linewidth=0.5)
        
            
        


folder_path = Path("C:/Users/lexda/local_pmt_info/rate-cap/James_data/")
#plt.figure(figsize=(9, 6))
# Loop through all CSV files in the folder
for file in folder_path.glob("*A118*.csv"):
    rate_cap_data = pd.read_csv(file, delimiter=',',
                                names=["rate" , "Tube Signal Current","Tube Dark Current","Anode Current","Anode / Strip Current", "Anode Current_2", "photon_rate", "photon_rate_linear" ,"photon_rate_diff"], skiprows=1)
    photon_rate = rate_cap_data['photon_rate']
    photon_rate_linear = rate_cap_data['photon_rate_linear']
    photon_rate_diff = rate_cap_data['photon_rate_diff']
    current = (rate_cap_data['Anode Current_2'])  #/175e-6)*28  # Convert to A
    current_linear = (rate_cap_data['Anode Current_2'][0]*rate_cap_data['rate']/rate_cap_data['rate'][0])
    
    #plt.scatter(photon_rate_linear, photon_rate_diff, s=5, label=r'previous Photek data at $1800\,\mathrm{V}$ fully illuminated')
    #plt.plot(photon_rate_linear, photon_rate_diff, linewidth=0.5 )
    #plt.scatter(current, photon_rate_diff, s=5,  label=r'previous Photek data at $1800\,\mathrm{V}$ fully illuminated')
    #plt.plot(current, photon_rate_diff, linewidth=0.5 )



    #plt.xlabel(r'Photon rate (Events/cm$^2$)')
    plt.xlabel(r'Current (A/cm$^2$)')

    plt.ylabel('relative gain')
    #plt.title(r'Current vs relative gain for A1181001 at $1800\,\mathrm{V}$',fontsize=16)
    plt.title(r'Current vs relative gain for A1240515 ',fontsize=16)

    #plt.xlabel('Current expected (I) (A/cm^2)', loc='left', labelpad=10, fontsize=16)
    #plt.ylabel('Current seen (I0) (A/cm^2)', loc='top', labelpad=10, fontsize=16)
    #plt.title('Current (I) vs current (I0) for A1181001 at 1800V',fontsize=16)
    plt.xscale('log')
    #plt.yscale('log')
    # #plt.xlim(1e-8, 0.5e-5)
    plt.ylim(0, 1.1)
    #plt.xlim(0, 1e9)

    params, covariance = curve_fit(model_I0,current_linear, current, p0=[1e-5, 0.7, 5], bounds=(0, [1e-4, 11, 900]))  

    # Extract fitted parameters
    A_fit, B_fit, C_fit = params
    print("Fitted parameters:")
    print(f"A = {A_fit:.4f}, B = {B_fit:.4f}, C = {C_fit:.4f}")

    # Generate smooth curve for plotting
    I_fit = np.linspace(min(current), max(current)+5e-6, 5000)
    I0_fit = model_I0(I_fit, A_fit, B_fit, C_fit)

    #plt.plot(I0_fit, I_fit, linewidth=0.5)
    #plt.plot(current_linear, current_linear, linewidth=0.5, color='black', linestyle='--', label='linear extrapolation')
    higher_rate = [3E5, 1E6, 2E6, 3E6, 4E6, 5E6, 6E6, 7E6, 8E6, 9E6, 1E7]
    higher_rate_current = current_linear[0]*np.array(higher_rate)/rate_cap_data['rate'][0]
    #plt.plot(higher_rate_current, higher_rate_current, linewidth=0.5, linestyle='--', color='black')


# Show plot
plt.legend(fontsize=22)
plt.rcParams["text.usetex"] = True
plt.show()




