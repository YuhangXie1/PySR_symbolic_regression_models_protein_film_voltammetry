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
    mse = sum((y_pred - y)**2)
    return mse

def fit_sin(tt, yy, ID):
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

    #lambdifying
    x0 = Symbol("x0")
    equation = A*sp.sin(w*x0 + p) + c
    equation_func = lambdify(x0, equation)
    y_pred = equation_func(tt)

    Path(os.path.join(output_filepath, str(ID))).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_filepath, "summary.csv"), "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
                        ID,
                        calculate_MSE(yy, y_pred),
                        equation,
                        A,
                        w,
                        p,
                        c,
                        ])
        
    #writing model parameters to metadata
    with open(os.path.join(output_filepath, str(ID), "metadata.txt"), "a") as metadata:
        metadata.write("params = curve_fit(function, x, y)\n")
        metadata.write("function = A * np.sin(B * x + C) + D\n")

    generate_plots(tt, yy, y_pred, ID)

    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}



def loading_data():
    """
    Returns numpy arrays of time_data, voltage, current and frequency for a set of data files
    """
    #loading data
    loc=r".\Data_for_eq_learning"
    files=os.listdir(loc)

    #loading data
    data_volt = pd.read_csv("Data_for_eq_learning/36_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
    data_amp = pd.read_csv("Data_for_eq_learning/36_Hz_2_cv_current", sep="\t", names = ["time","current"])
    data_combined_36 = data_volt.join(data_amp["current"])
    data_combined_36.insert(0, "Freq", 36)

    data_volt = pd.read_csv("Data_for_eq_learning/9_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
    data_amp = pd.read_csv("Data_for_eq_learning/9_Hz_2_cv_current", sep="\t", names = ["time","current"])
    data_combined_9 = data_volt.join(data_amp["current"])
    data_combined_9.insert(0, "Freq", 9)

    #data_combined = pd.concat([data_combined_9, data_combined_36])
    data_combined = data_combined_9

    #slicing data to remove start and end noise. Putting data into tuples
    slice_start = 550
    slice_end = -100
    time_data = np.array(data_combined["time"].iloc[slice_start:slice_end])
    voltage = np.array(data_combined["voltage"].iloc[slice_start:slice_end])
    current = np.array(data_combined["current"].iloc[slice_start:slice_end])
    freq = np.array(data_combined["Freq"].iloc[slice_start:slice_end])
    
    return time_data, voltage, current, freq


def model(X, Y, ID):
    """
    Runs a PySR model to fit (X, Y) and returns the best model.

    Keyword arguments:
    X -- numpy array of the input values
    Y -- numpy array of the output values
    """

    #pysr model definition
    model = PySRRegressor(
        maxsize=30,
        niterations=100,
        batching= True,
        binary_operators=["+","*"],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    )

    model.fit(X,Y)

    #writing model parameters to metadata
    with open(os.path.join(output_filepath, str(ID), "metadata.txt"), "a") as metadata:
        metadata.write(f"model run ID: {model.run_id_}")
        metadata.write('''
        model = PySRRegressor(
        maxsize=30,
        niterations=100,
        batching= True,
        binary_operators=["+","*"],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    )

                       \n''')

    return model

def generate_plots(time_data, voltage, current, freq, dv_dt, predicted_voltage, predicted_current, ID):
    
    #voltage time graph
    fig, axs = plt.subplots(3)
    axs[0].plot(time_data, voltage, label = "data", color = "cyan")
    axs[0].plot(time_data, predicted_voltage, label = "prediction", color = "red")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("voltage")
    axs[0].set_title(f"Repeat {ID}. Voltage vs time. Whole. ")
    axs[0].legend()

    slice_start = 10550
    slice_end = 20000
    axs[1].plot(time_data[slice_start:slice_end], voltage[slice_start:slice_end], label = "data", color = "cyan")
    axs[1].plot(time_data[slice_start:slice_end], predicted_voltage[slice_start:slice_end], label = "pred", color = "red", linestyle = "dotted")
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("voltage")
    axs[1].set_title(f"Repeat {ID}. Voltage vs time. Slice {slice_start}-{slice_end}.")
    axs[1].legend()

    slice_start = 30550
    slice_end = 40000
    axs[2].plot(time_data[slice_start:slice_end], voltage[slice_start:slice_end], label = "data", color = "cyan")
    axs[2].plot(time_data[slice_start:slice_end], predicted_voltage[slice_start:slice_end], label = "pred", color = "red", linestyle = "dotted")
    axs[2].set_xlabel("time")
    axs[2].set_ylabel("voltage")
    axs[2].set_title(f"Repeat {ID}. Voltage vs time. Slice {slice_start}-{slice_end}.")
    axs[2].legend()

    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, str(ID), f"fig{ID}-voltage-time.png"))

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

    slice_start = 30550
    slice_end = 40000
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


### main ###

#input variables
number_of_repeats = 10

output_filepath = rf"results/20250509-real-test-fit-dv-and-v-36/test-1"
Path(output_filepath).mkdir(parents=True, exist_ok=True)
with open(os.path.join(output_filepath, "summary.csv"), "a", newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
                "ID",
                "loss",
                "complexity",
                "MSE",
                "picked_equation",
                "substituted_form",
                ])

time_data, voltage, current, freq = loading_data()

""" x0 = Symbol("x0")
v = sympify("-0.30279955*cos(228.45365811675*x0 + 0.013544817) - 0.048228793")
dv_dt_func = diff(v, x0)
print(dv_dt_func)
dv_dt_lambda = lambdify(x0,dv_dt_func)
dv_dt = dv_dt_lambda(time_data) """

for i in range(0,number_of_repeats):

    start_time = time.time()

    fit_sin()

    X = np.array([voltage, voltage**2, voltage**3, dv_dt, dv_dt**2, dv_dt**3]).T
    Y = np.array(current).reshape(-1,1)

    select_model = model(X, Y)

    print (expand(select_model.sympy()))
    end_time = time.time()



    time_elapsed = str(datetime.timedelta(seconds = end_time - start_time))
    print(f"Time elapsed: {time_elapsed}")

    with open(os.path.join(output_filepath, "metadata.txt"), "a") as metadata:
        metadata.write(f"model {i} time taken = {time_elapsed} hr:min:sec \n")




    #dv_dt = 17.0298286814874*np.sin(56.644706*time_data)
"""     x0 = Symbol("x0")
    v = sympify("-0.30279955*cos(228.45365811675*x0 + 0.013544817) - 0.048228793")
    dv_dt_func = diff(v, x0)
    print(dv_dt_func)
    dv_dt_lambda = lambdify(x0,dv_dt_func)
    dv_dt = dv_dt_lambda(time_data) """

"""     #make file structure if does not exist
    Path(os.path.join(output_filepath, str(ID))).mkdir(parents = True, exist_ok = True)

    #adding details to metadata
    with open(os.path.join(output_filepath, str(ID), "metadata.txt"), "a") as metadata:
        metadata.write(f"time data: slice start: {slice_start}, slice end: {slice_end} \n")
        metadata.write("voltage formula: v = -0.30279955*cos(228.45365811675*x0 + 0.013544817) - 0.048228793 \n")
        metadata.write(f"dv/dt formula: dv_dt = {str(dv_dt_func)} \n") """

"""
    #printing and writing to file
    print(model)
    print(model.sympy())
    Path(os.path.join(output_filepath, str(ID))).mkdir(parents=True, exist_ok=True)
    model.equations_.to_csv(os.path.join(output_filepath, str(ID), "equations.csv"))

    with open(os.path.join(output_filepath, "summary.csv"), "a", newline='') as file:
        best = model.get_best()
        expanded_form = expand(sympify(str(model.sympy())))
        x0, x1, x2, x3, x4, x5 = symbols("x0 x1 x2 x3 x4 x5")
        x, dx = symbols("x dx") 
        substituted_form = expand(expanded_form.subs([(x0,x),(x1,x**2),(x2,x**3),(x3,dx),(x4,dx**2),(x5,dx**3)]))

        writer = csv.writer(file)
        writer.writerow([
                        ID,
                        best.loss,
                        best.complexity,
                        calculate_MSE(current, model.predict(X)),
                        expanded_form,
                        substituted_form,
                        ])
"""

"""



    #writing model parameters to metadata
    with open(os.path.join(output_filepath, str(ID), "metadata.txt"), "a") as metadata:
        metadata.write(f"model run ID: {model.run_id_}")
        metadata.write('''
        model = PySRRegressor(
        maxsize=30,
        niterations=100,
        batching= True,
        binary_operators=["+","*"],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    )

                       \n''')
    
    predicted_current = model.predict(X)
    generate_plots(dv_dt, current, predicted_current, time_data, ID)
"""