from pysr import PySRRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import datetime
from pathlib import Path
import csv
from sympy import Symbol, sympify, expand, symbols, diff
from sympy.utilities.lambdify import lambdify
from scipy.optimize import curve_fit
import sympy as sp


def calculate_MSE(y, y_pred):
    """Returns the mean squared error between a y value (real) and a predicted y value (prediction)."""
    mse = sum((y_pred - y)**2)/len(y)
    return mse

def load_data(Hz):
    """
    Returns numpy arrays of time_data, voltage, current and frequency for a set of data files
    """
    #loading data
    data_volt = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_1_cv_voltage", sep="\t", names = ["time","voltage"])
    data_amp = pd.read_csv(f"Data_for_eq_learning/{Hz}_Hz_1_cv_current", sep="\t", names = ["time","current"])
    data_combined = data_volt.join(data_amp["current"])

    #slicing data to remove start and end noise. Putting data into tuples
    slice_start = 550
    slice_end = -100
    time_data = np.array(data_combined["time"].iloc[slice_start:slice_end])
    voltage = np.array(data_combined["voltage"].iloc[slice_start:slice_end])
    current = np.array(data_combined["current"].iloc[slice_start:slice_end])

    return time_data, voltage, current, [slice_start, slice_end]

def load_data_2(filepath):
    data = pd.read_csv(filepath, header=0)

    slice_start = 550
    slice_end = -100
    time_data = np.array(data["x"].iloc[slice_start:slice_end])
    voltage = np.array(data["z"].iloc[slice_start:slice_end])
    current = np.array(data["y"].iloc[slice_start:slice_end])

    return time_data, voltage, current, [slice_start, slice_end]

def fit_sin(tt, yy):
    """
    Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"
    Obtained from https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy, user unsym, Feb 19, 2017

    """
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    output_dict = {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

    #lambdifying
    x0 = Symbol("x0")
    voltage_equation = A*sp.sin(w*x0 + p) + c

    return voltage_equation, [A,w,p,c]


def model(time_data, voltage, dv_dt, current, A1, A2, A3, w1, w2, w3, ID):
    """
    Runs a PySR model to fit ([voltage, dv_dt], current) and returns the best model.
    """

    #X = np.array([voltage, voltage**2, voltage**3, dv_dt, dv_dt**2, dv_dt**3, freq, freq**2, freq**3]).T
    X = np.array([voltage, dv_dt, A1, A2, A3, w1, w2, w3]).T
    Y = np.array(current).reshape(-1,1)

    #pysr model definition
    model = PySRRegressor(
        maxsize=40,
        niterations=200,
        batching= True,
        binary_operators=["+","*"],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    )

    model.fit(X,Y)
    predicted_current = model.predict(X)

    #writing model parameters to metadata
    Path(os.path.join(output_filepath, str(ID))).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_filepath, str(ID), "metadata.txt"), "a") as metadata:
        metadata.write(f"model run ID: {model.run_id_}")
        metadata.write('''
    X = np.array([voltage, dv_dt, A1, A2, A3, w1, w2, w3]).T
    Y = np.array(current).reshape(-1,1)

    model = PySRRegressor(
        maxsize=40,
        niterations=200,
        batching= True,
        binary_operators=["+","*"],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    )

                       \n''')
        
    #printing and writing to file
    model.equations_.to_csv(os.path.join(output_filepath, str(ID), "equations.csv"))
    

    with open(os.path.join(output_filepath, "summary_current_model.csv"), "a", newline='') as file:
        best = model.get_best()
        expanded_form = expand(sympify(str(model.sympy())))
        x0, x1, x2, x3, x4, x5, x6, x7, x8 = symbols("x0 x1 x2 x3 x4 x5 x6 x7 x8")
        x, dx, A1, A2, A3, w1, w2, w3 = symbols("x dx A1 A2 A3 w1 w2 w3") 
        #substituted_form = expand(expanded_form.subs([(x0,x),(x1,x**2),(x2,x**3),(x3,dx),(x4,dx**2),(x5,dx**3),(x6,f),(x7,f**2),(x8,f**3)]))
        substituted_form = expand(expanded_form.subs([(x0,x),(x1,dx),(x2,A1),(x3,A2),(x4,A3),(x5,w1),(x5,w2),(x5,w3)]))

        writer = csv.writer(file)
        writer.writerow([
                        ID,
                        best.loss,
                        best.complexity,
                        calculate_MSE(current, model.predict(X)),
                        expanded_form,
                        substituted_form,
                        ])
        
    #current time graph
    fig, axs = plt.subplots(3)
    axs[0].plot(time_data, current, label = "data", color = "orange")
    axs[0].plot(time_data, predicted_current, label = "prediction", color = "magenta")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("current")
    axs[0].set_title(f"Repeat {ID}. Current vs time. Whole. ")
    axs[0].legend()

    slice_start = 10550
    slice_end = 20000
    axs[1].plot(time_data[slice_start:slice_end], current[slice_start:slice_end], label = "data", color = "orange")
    axs[1].plot(time_data[slice_start:slice_end], predicted_current[slice_start:slice_end], label = "pred", color = "magenta", linestyle = "dotted")
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("current")
    axs[1].set_title(f"Repeat {ID}. Current vs time. Slice {slice_start}-{slice_end}.")
    axs[1].legend()

    slice_start = 110550
    slice_end = 120000
    axs[2].plot(time_data[slice_start:slice_end], current[slice_start:slice_end], label = "data", color = "orange")
    axs[2].plot(time_data[slice_start:slice_end], predicted_current[slice_start:slice_end], label = "pred", color = "magenta", linestyle = "dotted")
    axs[2].set_xlabel("time")
    axs[2].set_ylabel("current")
    axs[2].set_title(f"Repeat {ID}. Current vs time. Slice {slice_start}-{slice_end}.")
    axs[2].legend()

    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, str(ID), f"fig{ID}-current-time.png"))

    #current dV/dt graph
    fig, axs = plt.subplots()
    axs.plot(dv_dt, current, label = "data", color = "orange")
    axs.plot(dv_dt, predicted_current, label = "prediction", color = "magenta")
    axs.set_xlabel("dV/dt")
    axs.set_ylabel("current")
    axs.set_title(f"Repeat {ID}. Current vs dV/dt.")
    axs.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, str(ID), f"fig{ID}-current-dv_dt.png"))

    return model

def fit_voltage_eqn(Hz):
    time_data, voltage, current, slice_array = load_data(Hz)
    voltage_eqn, coeff_array = fit_sin(time_data, voltage)
    
    x0 = Symbol("x0")
    voltage_equation_func = lambdify(x0, voltage_eqn)
    voltage_pred = voltage_equation_func(time_data)
    voltage_MSE = calculate_MSE(voltage, voltage_pred)
    dv_dt_eqn = diff(voltage_eqn, x0)

    return voltage_eqn, dv_dt_eqn, coeff_array

### main ###
output_filepath = rf"report_plots"

files_freq = [9, 45, 54, 63, 72, 81, 90, 99]
MSE_average = []
#Hz = 45
for Hz in files_freq:
    time_data, voltage, current, slice_array = load_data(Hz) #loading file 1s
    voltage_eqn, dv_dt_eqn, coeff_array = fit_voltage_eqn(Hz)
    A = coeff_array[0]
    w = coeff_array[1]
    p = coeff_array[2]
    c = coeff_array[3]

    #current phase and amplitude data
    phase_data = pd.read_csv(rf"results\20250617-current-phase-original-set\summary.csv")
    p_val = np.array(phase_data.loc[phase_data["Hz"] == Hz]["phase"])[0]
    A_val = np.array(phase_data.loc[phase_data["Hz"] == Hz]["amplitude"])[0]


    x0 = Symbol("x0")
    #voltage_lambda = lambdify(x0,voltage_eqn)
    #voltage = voltage_lambda(time_data)
    dv_dt_lambda = lambdify(x0,dv_dt_eqn)
    dv_dt = dv_dt_lambda(time_data)

    #loading file 2 current equation
    load_filepath = r"results\20250513-fit-workflow"
    x, dx = symbols("x dx")
    
    current_summary_file = pd.read_csv(os.path.join(load_filepath, str(Hz), "summary_current_model.csv"), header=0)
    data_equations = sympify(np.array(current_summary_file["substituted_form"]))
    current_eqn_sympy = data_equations[9]
    #current_eqn_sympy = 1.66405082916131e-6*dx*x**3 - 4.21535885060198e-7*dx*x**2 - 1.1204689e-6*dx*x + 6.65111994127854e-6*dx + 0.000520053229898644*x**2 + 0.0013823342*x + 4.408361e-5
    #current_eqn_lambda = lambdify([x, dx, A, w, p, c, A_val, p_val], eqn_rm)
    #current_pred = current_eqn_lambda(voltage, dv_dt, A_dat, w_dat, p_dat, c_dat, A_val_dat, p_val_dat)
    current_eqn_lambda = lambdify([x, dx], current_eqn_sympy)
    current_pred = current_eqn_lambda(voltage, dv_dt)

    MSE_average.append(float(calculate_MSE(current, current_pred)))

print(MSE_average)
""" fig, axs = plt.subplots()
axs.plot(dv_dt, current, label = "data", color = "blue")
axs.plot(dv_dt, current_pred, label = "pred", color = "red")
axs.set_xlabel("dV/dt (V/s)")
axs.set_ylabel("current (A)")
axs.legend()
fig.tight_layout()
plt.show()

fig, axs = plt.subplots()
axs.plot(time_data, current, label = "data", color = "blue")
axs.plot(time_data, current_pred, label = "pred", color = "red", linestyle="dotted")
axs.set_xlabel("time (s)")
axs.set_ylabel("current (A)")
axs.set_xlim([0,0.15])
axs.legend()
fig.tight_layout()
plt.show()

print(calculate_MSE(current, current_pred)) """