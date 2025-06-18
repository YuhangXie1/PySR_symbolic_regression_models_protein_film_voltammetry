import numpy as np
from sympy import Symbol, sympify, symbols, diff, Wild, expand
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

def load_data(Hz):
    """
    Returns numpy arrays of time_data, voltage, current and frequency for a set of data files
    """
    #loading data
    #data_volt = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
    #data_amp = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_2_cv_current", sep="\t", names = ["time","current"])
    
    data_amp = pd.read_csv(r"Data_for_eq_learning\FTacV_before_PSV_cv_current", sep="\t", names = ["time","current"])
    data_volt = pd.read_csv(r"Data_for_eq_learning\FTacV_before_PSV_cv_voltage", sep="\t", names = ["time","voltage"])

    #slicing data to remove start and end noise. Putting data into tuples
    slice_start = 0
    slice_end = -50
    time_data = np.array(data_amp["time"].iloc[slice_start:slice_end])
    voltage = np.array(data_volt["voltage"].iloc[slice_start:slice_end])
    current = np.array(data_amp["current"].iloc[slice_start:slice_end])

    return time_data, voltage, current, [slice_start, slice_end]


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

####

time_data, voltage, current, slice = load_data(0)

fig, axs = plt.subplots()
axs.plot(time_data, voltage, label = "voltage")
axs.set_xlabel("time")
axs.set_ylabel("voltage")
axs.set_title("FTacV - voltage-time")
fig.tight_layout()
plt.show()

fig, axs = plt.subplots()
axs.plot(time_data, current, label = "voltage")
axs.set_xlabel("time")
axs.set_ylabel("current")
axs.set_title("FTacV - current-time")
fig.tight_layout()
plt.show()




output_filepath = rf"results/20250618-FTacV-explore/fft/"
Path(output_filepath).mkdir(parents=True, exist_ok=True)


""" #calculating predicted current
x, dx, f = symbols("x dx f")
current_pred_lambda = lambdify([x, dx, f], eqn)
current_pred = current_pred_lambda(voltage, dv_dt, freq) """

band_size=0.1
freqs, ft = fourier_transform(current, time_data)
#freqs_pred, ft_pred = fourier_transform(current_pred, time_data)

#ploting all harmonics
fig, axs = plt.subplots()
axs.plot(freqs, np.log10(ft**2), color = "blue", label="real")
#axs.plot(freqs_pred, np.log10(ft_pred**2), color = "red", label="pred")
axs.set_xlim(0,400)
axs.set_ylim(-4, 3)
axs.set_xlabel("Frequency")
axs.set_ylabel("Log10 ft^2")
axs.set_title(f"FTacV all harmonics")
axs.legend()
fig.tight_layout()
plt.savefig(os.path.join(output_filepath, f"FTacV-all-harmonic-freq.png"))

#plotting figures 1 by 1
Hz = 36
for harmonic in range(1,10):
    fig, axs = plt.subplots()

    ft_filtered = filter_fft(freqs, ft, band_size, Hz, harmonic)
    #ft_filtered_pred = filter_fft(freqs_pred, ft_pred, band_size, Hz, harmonic)
    
    log_ft_filtered = np.log10(ft_filtered**2)
    #log_ft_filtered_pred = np.log10(ft_filtered_pred**2)
    
    axs.plot(freqs, log_ft_filtered, color = "blue", label="real")
    #axs.plot(freqs_pred, log_ft_filtered_pred, color = "red", label="pred")
    axs.set_xlim(harmonic*int(Hz)-harmonic*int(Hz)*0.03, harmonic*int(Hz)+harmonic*int(Hz)*0.03)
    axs.legend()
    axs.set_xlabel("Frequency (Hz)")
    axs.set_ylabel("Log10 ft^2")
    axs.set_title(f"FTacV harmonic {harmonic} in freq domain")
    fig.tight_layout()
    Path(os.path.join(output_filepath,"freq_domain")).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(output_filepath, "freq_domain", f"FTacV-harmonic-{harmonic}-freq.png"))
    plt.close(fig.figure)


    inverseft=np.fft.ifft(ft_filtered)
    #inverseft_pred=np.fft.ifft(ft_filtered_pred)

    fig, axs = plt.subplots()
    axs.plot(time_data, inverseft, color = "blue", label="real")
    #axs.plot(time_data, inverseft_pred, color = "red", label="pred")
    axs.legend()
    axs.set_xlabel("Time")
    axs.set_ylabel("Current")
    axs.set_title(f"File {Hz}Hz harmonic {harmonic} in time domain")
    fig.tight_layout()
    Path(os.path.join(output_filepath,"time_domain")).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(output_filepath,"time_domain", f"FTacV-harmonic-{harmonic}-time.png"))
    plt.close(fig.figure)


    fig, axs = plt.subplots()
    axs.plot(voltage, inverseft, color = "blue", label="real")
    #axs.plot(voltage, inverseft_pred, color = "red", label="pred")
    axs.legend()
    axs.set_xlabel("Voltage")
    axs.set_ylabel("Current")
    axs.set_title(f"File {Hz}Hz harmonic {harmonic} in voltage domain")
    fig.tight_layout()
    Path(os.path.join(output_filepath,"voltage_domain")).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(output_filepath,"voltage_domain", f"FTacV-harmonic-{harmonic}-freq.png"))
    plt.close(fig.figure)

#Plotting 3 by 3 figures

fig1, axs1 = plt.subplots(3,3)
axs1 = axs1.flatten()

fig2, axs2 = plt.subplots(3,3)
axs2 = axs2.flatten()

fig3, axs3 = plt.subplots(3,3)
axs3 = axs3.flatten()

for harmonic in range(1,10):
    ax1 = axs1[harmonic - 1]

    ft_filtered = filter_fft(freqs, ft, band_size, Hz, harmonic)
    #ft_filtered_pred = filter_fft(freqs_pred, ft_pred, band_size, Hz, harmonic)
    
    log_ft_filtered = np.log10(np.abs(ft_filtered))
    #log_ft_filtered_pred = np.log10(np.abs(ft_filtered_pred))
    
    ax1.plot(freqs, log_ft_filtered, color = "blue", label="real")
    #ax1.plot(freqs_pred, log_ft_filtered_pred, color = "red", label="pred")
    ax1.set_xlim(harmonic*int(Hz)-harmonic*int(Hz)*0.03, harmonic*int(Hz)+harmonic*int(Hz)*0.03)
    ax1.set_title(f"Harmonic {harmonic}")

    handles_1, labels_1 = ax1.get_legend_handles_labels()

    ax2 = axs2[harmonic - 1]

    inverseft=np.fft.ifft(ft_filtered)
    #inverseft_pred=np.fft.ifft(ft_filtered_pred)

    ax2.plot(time_data, inverseft, color = "blue", label="real")
    #ax2.plot(time_data, inverseft_pred, color = "red", label="pred")
    ax2.set_title(f"Harmonic {harmonic}")

    handles_2, labels_2 = ax2.get_legend_handles_labels()

    ax3 = axs3[harmonic - 1]
    ax3.plot(voltage, inverseft, color = "blue", label="real")
    #ax3.plot(voltage, inverseft_pred, color = "red", label="pred")
    ax3.set_title(f"Harmonic {harmonic}")

    handles_3, labels_3 = ax3.get_legend_handles_labels()


    log_ft_filtered[~np.isfinite(log_ft_filtered)] = 0
    #log_ft_filtered_pred[~np.isfinite(log_ft_filtered_pred)] = 0
    """     with open(os.path.join(output_filepath, f"summary.csv"), "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([harmonic,
                        calculate_MSE_complex(log_ft_filtered,log_ft_filtered_pred),
                        calculate_MSE_complex(inverseft,inverseft_pred),
                        ]) """

fig1.legend(handles_1, labels_1, loc='lower right')
fig1.supxlabel("Frequency (Hz)")
fig1.supylabel("Log10 abs(ft)")
fig1.suptitle(f"File {Hz}Hz harmonics in freq domain")
fig1.tight_layout()
fig1.savefig(os.path.join(output_filepath, f"FTacV-harmonics-freq-3x3.png"))
plt.close(fig1.figure)

fig2.legend(handles_2, labels_2, loc='lower right')
fig2.supxlabel("Time")
fig2.supylabel("Current")
fig2.suptitle(f"File {Hz}Hz harmonics in time domain")
fig2.tight_layout()
fig2.savefig(os.path.join(output_filepath, f"FTacV-harmonics-time-3x3.png"))
plt.close(fig2.figure)

fig3.legend(handles_3, labels_3, loc='lower right')
fig3.supxlabel("Voltage")
fig3.supylabel("Current")
fig3.suptitle(f"File {Hz}Hz harmonics in voltage domain")
fig3.tight_layout()
fig3.savefig(os.path.join(output_filepath, f"FTacV-harmonics-voltage-3x3.png"))
plt.close(fig3.figure)