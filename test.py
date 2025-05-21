import numpy as np
from sympy import Symbol, sympify, symbols, diff
import sympy as sp
from sympy.utilities.lambdify import lambdify
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import os
from copy import deepcopy

def calculate_MSE(y, y_pred):
    """Returns the mean squared error between a y value (real) and a predicted y value (prediction)."""
    mse = sum((y_pred - y)**2)
    return mse

def calculate_MSE_complex(y, y_pred):
    """Returns the mean squared error between a complex y value and a predicted complex y value."""
    mse = sum(np.abs(y_pred - y)**2)
    return mse

Hz = 36
data_volt = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
data_amp = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_2_cv_current", sep="\t", names = ["time","current"])
data_combined = data_volt.join(data_amp["current"])

#slicing data to remove start and end noise. Putting data into tuples
slice_start = 550
slice_end = -100
time_data = np.array(data_combined["time"].iloc[slice_start:slice_end])
voltage = np.array(data_combined["voltage"].iloc[slice_start:slice_end])
current = np.array(data_combined["current"].iloc[slice_start:slice_end])

x, dx, x0 = symbols("x dx x0")
voltage_eqn = sympify("0.299894563526963*sin(228.4492878688*x0 - 1.55129181348585) - 0.0496859862564122")
dv_dt_func = diff(voltage_eqn, x0)
dv_dt_lambda = lambdify(x0,dv_dt_func)
dv_dt = dv_dt_lambda(time_data)

current_pred_eqn = 1.83479321254227e-6*dx*x**3 - 7.77746142237813e-7*dx*x**2 - 7.77746142237813e-7*dx*x + 7.5642693e-6*dx + 0.00060018763*x**2 + 0.0010604324*x + 2.4237355e-5
#current_pred_eqn = -1.7188134e-9*dx**2*x - 1.3682722e-5*dx*x**4 - 7.6412488721775e-7*dx*x + 7.5565454e-6*dx + 0.00059690495*x**2 + 0.0010617722*x + 2.4272074e-5
current_pred_lambda = lambdify([x, dx], current_pred_eqn)
current_pred = current_pred_lambda(voltage, dv_dt)






def fourier_transform(current, time_data):
   
    ft=np.fft.fft(current)
    freqs=np.fft.fftfreq(len(time_data), time_data[1]-time_data[0])

    return freqs, ft
    
def filter_fft(freqs, ft, band_size, Hz, desired_harmonic):
    freq_loc=desired_harmonic*int(Hz)
    filtered_ft=np.zeros(len(ft), dtype="complex")
    for sign in [-1, 1]:
        ft_loc=np.where((freqs>sign*freq_loc-(band_size*int(Hz))) & (freqs<sign*freq_loc+(band_size*int(Hz))))
        filtered_ft[ft_loc]=ft[ft_loc]
    
    return filtered_ft

band_size=0.1
desired_harmonic=9
freqs, ft = fourier_transform(current, time_data)
freqs_pred, ft_pred = fourier_transform(current_pred, time_data)

ft = np.abs(ft)
ft_pred = np.abs(ft_pred)

ft_filtered = filter_fft(freqs, ft, band_size, Hz, desired_harmonic)
ft_filtered_pred = filter_fft(freqs_pred, ft_pred, band_size, Hz, desired_harmonic)

inverseft=np.fft.ifft(ft_filtered)
inverseft_pred=np.fft.ifft(ft_filtered_pred)

log_ft = np.log10(ft)
log_ft_pred = np.log10(ft_pred)

log_ft[~np.isfinite(log_ft)] = 0
log_ft_pred[~np.isfinite(log_ft_pred)] = 0

MSE_norm = calculate_MSE(ft, ft_pred)
MSE_log = calculate_MSE(log_ft, log_ft_pred)

fig, axs = plt.subplots()
axs.plot(freqs, np.log10(ft), color = "blue", label="real")
axs.plot(freqs_pred, np.log10(ft_pred), color = "red", label="pred")
axs.set_xlim(0,400)
axs.set_ylim(-4, 3)
axs.set_xlabel("Frequency")
axs.set_ylabel("log10 ft")
axs.set_title(f"File {Hz}Hz, all harmonics, \n MSE {MSE_norm}, \n MSE log {MSE_log}")
axs.legend()

fig.tight_layout()
plt.show()


#saving figures 1 by 1
""" output_filepath = rf"results/20250520-test/"
Path(os.path.join(output_filepath)).mkdir(parents=True, exist_ok=True)
with open(os.path.join(output_filepath, f"summary.csv"), "a", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Harmonic",
                    "MSE_freq",
                    "MSE_inverseft",
                    ]) """




#Plotting 3 by 3 figures

fig1, axs1 = plt.subplots(3,3)
axs1 = axs1.flatten()

fig2, axs2 = plt.subplots(3,3)
axs2 = axs2.flatten()

fig3, axs3 = plt.subplots(3,3)
axs3 = axs3.flatten()

MSE_dict = {}

for harmonic in range(1,10):
    ax1 = axs1[harmonic - 1]

    ft_filtered = filter_fft(freqs, ft, band_size, Hz, harmonic)
    ft_filtered_pred = filter_fft(freqs_pred, ft_pred, band_size, Hz, harmonic)
    
    ft_filtered_abs = np.abs(ft_filtered)
    ft_filtered_pred_abs = np.abs(ft_filtered_pred)

    log_ft_filtered = np.log10(ft_filtered_abs)
    log_ft_filtered_pred = np.log10(ft_filtered_pred_abs)

    data = np.array([freqs,log_ft_filtered,log_ft_filtered_pred]).T
    data_new = np.array([row for row in data if np.isfinite(row[1]) and np.isfinite(row[2])])

    """     ft_dict = {}
    ft_dict_pred = {}
    for i, item in enumerate(freqs):
        ft_dict[item] = log_ft_filtered[i]
        ft_dict_pred[item] = log_ft_filtered_pred[i]

    ft_dict_new = deepcopy(ft_dict)
    ft_dict_new_pred = deepcopy(ft_dict_pred)
    for key, item in ft_dict.items():
        if np.isfinite(item) == False:
            ft_dict_new.pop(key)

    for key, item in ft_dict_pred.items():
        if np.isfinite(item) == False:
            ft_dict_new_pred.pop(key)

    new_ft = np.array([float(item) for item in ft_dict_new.values()])
    new_freq = np.array([float(item) for item in ft_dict_new.keys()])

    new_ft_pred = np.array([float(item) for item in ft_dict_new_pred.values()])
    new_freq_pred = np.array([float(item) for item in ft_dict_new_pred.keys()]) """

    MSE_dict[harmonic] = calculate_MSE(data_new[:,1],data_new[:,2])
    ax1.scatter(data_new[:,0], data_new[:,1], color = "blue", label="real")
    ax1.scatter(data_new[:,0], data_new[:,2], color = "red", label="pred")
    #ax1.set_xlim(harmonic*int(Hz)-harmonic*int(Hz)*0.03, harmonic*int(Hz)+harmonic*int(Hz)*0.03)
    ax1.set_title(f"Harmonic {harmonic}")

    handles_1, labels_1 = ax1.get_legend_handles_labels()

    ax2 = axs2[harmonic - 1]

    inverseft=np.fft.ifft(ft_filtered)
    inverseft_pred=np.fft.ifft(ft_filtered_pred)

    ax2.plot(time_data, inverseft, color = "blue", label="real")
    ax2.plot(time_data, inverseft_pred, color = "red", label="pred")
    ax2.set_title(f"Harmonic {harmonic}")

    handles_2, labels_2 = ax2.get_legend_handles_labels()

    ax3 = axs3[harmonic - 1]
    ax3.plot(voltage, inverseft, color = "blue", label="real")
    ax3.plot(voltage, inverseft_pred, color = "red", label="pred")
    ax3.set_title(f"Harmonic {harmonic}")

    handles_3, labels_3 = ax3.get_legend_handles_labels()

    log_ft_filtered[~np.isfinite(log_ft_filtered)] = 0
    log_ft_filtered_pred[~np.isfinite(log_ft_filtered_pred)] = 0
    """     with open(os.path.join(output_filepath, f"summary.csv"), "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([harmonic,
                        calculate_MSE_complex(log_ft_filtered,log_ft_filtered_pred),
                        calculate_MSE_complex(inverseft,inverseft_pred),
                        ]) """
print(MSE_dict)

fig1.legend(handles_1, labels_1, loc='lower right')
fig1.supxlabel("Frequency (Hz)")
fig1.supylabel("Log10 ft^2")
fig1.suptitle(f"File {Hz}Hz harmonics in freq domain")
fig1.tight_layout()

fig2.legend(handles_2, labels_2, loc='lower right')
fig2.supxlabel("Time")
fig2.supylabel("Current")
fig2.suptitle(f"File {Hz}Hz harmonics in time domain")
fig2.tight_layout()

fig3.legend(handles_3, labels_3, loc='lower right')
fig3.supxlabel("Voltage")
fig3.supylabel("Current")
fig3.suptitle(f"File {Hz}Hz harmonics in voltage domain")
fig3.tight_layout()
plt.show()


