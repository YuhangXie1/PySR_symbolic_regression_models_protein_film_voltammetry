import numpy as np
from sympy import Symbol, sympify, symbols, diff
import sympy as sp
from sympy.utilities.lambdify import lambdify
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

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
current_pred_lambda = lambdify([x, dx], current_pred_eqn)
current_pred = current_pred_lambda(voltage, dv_dt)






def fourier_transform(current, time_data, Hz, band_size, desired_harmonic):
   
    ft=np.fft.fft(current)
    freqs=np.fft.fftfreq(len(time_data), time_data[1]-time_data[0])

    freq_loc=desired_harmonic*int(Hz)
    filtered_ft=np.zeros(len(ft), dtype="complex")
    for sign in [-1, 1]:
        ft_loc=np.where((freqs>sign*freq_loc-(band_size*int(Hz))) & (freqs<sign*freq_loc+(band_size*int(Hz))))
        filtered_ft[ft_loc]=ft[ft_loc]
    
    inverseft=np.fft.ifft(filtered_ft)

    return freqs, ft, filtered_ft, inverseft
    
band_size=0.1
desired_harmonic=4
freqs, ft, filtered_ft, inverseft = fourier_transform(current, time_data, Hz, band_size, desired_harmonic)
freqs_pred, ft_pred, filtered_ft_pred, inverseft_pred = fourier_transform(current_pred, time_data, Hz, band_size, desired_harmonic)

fig, axs = plt.subplots()
axs.plot(freqs, np.log10(ft**2), color = "blue", label="real")
axs.plot(freqs, np.log10(ft_pred**2), color = "red", label="pred")
axs.set_xlim(0,400)
axs.set_ylim(-4, 3)
axs.set_xlabel("frequency")
axs.set_ylabel("log10 ft^2")
axs.set_title(f"File {Hz}, all harmonics, fft")

fig.tight_layout()
plt.show()


fig, axs = plt.subplots()
axs.plot(freqs, np.log10(filtered_ft**2), color = "blue", label="real")
axs.plot(freqs, np.log10(filtered_ft_pred**2), color = "red", label="pred")
axs.set_xlim(0)
#axs.set_ylim(-4, 3)
axs.set_xlabel("frequency")
axs.set_ylabel("log10 ft^2")
axs.set_title(f"File {Hz}, harmonic {desired_harmonic}, fft")

fig.tight_layout()
plt.show()

""" axs[1].plot(freqs, np.log10(filtered_ft**2), color = "blue", label="real select")
axs[1].plot(freqs, np.log10(filtered_ft_pred**2), color = "red", label="pred select")
axs[1].set_xlim(0,400)
axs[1].set_ylim(-4, 3)
axs[1].set_xlabel("frequency")
axs[1].set_ylabel("log10 ft^2")
axs[1].set_title(f"File {Hz}, harmonic {desired_harmonic}, fft")

axs[2].plot(voltage, inverseft, label="real")
axs[2].plot(voltage, inverseft_pred, label="pred")
axs[2].set_xlabel("voltage")
axs[2].set_ylabel("current")
axs[2].set_title(f"File {Hz}, harmonic {desired_harmonic}, ifft") """

#axs.plot()

""" freqs=np.fft.fftfreq(len(time), time[1]-time[0])
freq_loc=desired_harmonic*int(freq[0])
new_ft=np.zeros(len(ft), dtype="complex")
for sign in [-1, 1]:
    ft_loc=np.where((freqs>sign*freq_loc-(band_size*int(freq[0]))) & (freqs<sign*freq_loc+(band_size*int(freq[0]))))
    new_ft[ft_loc]=ft[ft_loc]

plt.plot(freqs, ft, label=freq)
plt.plot(freqs, new_ft)
inversefft=np.fft.ifft(new_ft)
plt.plot(time, inversefft) """

