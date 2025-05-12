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

    mse = sum((y_pred - y)**2)
    print(f"MSE:{mse}")

    return mse

def loading_data(ID):
    #loading data
    loc=r".\Data_for_eq_learning"
    files=os.listdir(loc)

    data_volt = pd.read_csv("Data_for_eq_learning/36_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
    data_amp = pd.read_csv("Data_for_eq_learning/36_Hz_2_cv_current", sep="\t", names = ["time","current"])
    data_combined_9 = data_volt.join(data_amp["current"])
    data_combined_9.insert(0, "Freq", 36)

    #slicing data to remove start and end noise. Putting data into tuples
    slice_start = 550
    slice_end = -100
    time_data = np.array(data_combined_9["time"].iloc[slice_start:slice_end])
    voltage = tuple(data_combined_9["voltage"].iloc[slice_start:slice_end])
    current = tuple(data_combined_9["current"].iloc[slice_start:slice_end])
    freq = tuple(data_combined_9["Freq"].iloc[slice_start:slice_end])

    #dv_dt = 17.0298286814874*np.sin(56.644706*time_data)
    x0 = Symbol("x0")
    v = sympify("-0.30279955*cos(228.45365811675*x0 + 0.013544817) - 0.048228793")
    dv_dt_func = diff(v, x0)
    print(dv_dt_func)
    dv_dt_lambda = lambdify(x0,dv_dt_func)
    dv_dt = dv_dt_lambda(time_data)

    #make file structure if does not exist
    Path(os.path.join(output_filepath, str(ID))).mkdir(parents = True, exist_ok = True)

    #adding details to metadata
    with open(os.path.join(output_filepath, str(ID), "metadata.txt"), "a") as metadata:
        metadata.write(f"time data: slice start: {slice_start}, slice end: {slice_end} \n")
        metadata.write("voltage formula: v = -0.30279955*cos(228.45365811675*x0 + 0.013544817) - 0.048228793 \n")
        metadata.write(f"dv/dt formula: dv_dt = {str(dv_dt_func)} \n")

    return voltage, dv_dt, current, time_data

def sinusoidal_fit(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

def fit(x, y, function, ID):

    params, params_cover = curve_fit(function, x, y)
    print(params)
    x0 = Symbol("x0")
    equation = params[0]*sp.sin(params[1]*x0 + params[2]) + params[3]
    equation_func = lambdify(x0, equation)
    y_pred = equation_func(np.array(x))

    Path(os.path.join(output_filepath, str(ID))).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_filepath, "summary.csv"), "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
                        ID,
                        calculate_MSE(y, y_pred),
                        equation,
                        params[0],
                        params[1],
                        params[2],
                        params[3],
                        ])

    #writing model parameters to metadata
    with open(os.path.join(output_filepath, str(ID), "metadata.txt"), "a") as metadata:
        metadata.write("params = curve_fit(function, x, y)\n")
        metadata.write("function = A * np.sin(B * x + C) + D\n")
    
    generate_plots([], y, y_pred, x, ID)

def model(voltage, dv_dt, current,  time_data, ID):

    #shaping variables for pysr
    #voltage = np.array(voltage)
    #dv_dt = np.array(dv_dt)
    #X = np.array([voltage, voltage**2, voltage**3, dv_dt, dv_dt**2, dv_dt**3]).T
    X = np.array(time_data).reshape(-1,1)
    Y = np.array(voltage).reshape(-1,1)

    #pysr model define
    model = PySRRegressor(
        maxsize=30,
        niterations=100,
        batching= True,
        binary_operators=["+","*"],
        unary_operators=["sin","cos"],
        complexity_of_operators={"sin":3, "cos":3},
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    )

    model.fit(X,Y)

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

    #writing model parameters to metadata
    with open(os.path.join(output_filepath, str(ID), "metadata.txt"), "a") as metadata:
        metadata.write(f"model run ID: {model.run_id_}")
        metadata.write('''
    model = PySRRegressor(
        maxsize=30,
        niterations=100,
        batching= True,
        binary_operators=["+","*"],
        unary_operators=["sin","cos"],
        complexity_of_operators={"sin":3, "cos":3},
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    )
                       \n''')
    
    predicted_current = model.predict(X)
    generate_plots(dv_dt, current, predicted_current, time_data, ID)

    return model

def generate_plots(dv_dt, current, predicted_current, time_data, ID):
    """     fig, axs = plt.subplots()
    axs.plot(dv_dt, current, label = "data", color = "cyan")
    axs.plot(dv_dt, predicted_current, label = "prediction", color = "red")
    axs.set_xlabel("dV/dt")
    axs.set_ylabel("current")
    axs.set_title(f"Repeat {ID} dV/dt vs current")
    axs.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, str(ID), f"fig{ID}-dvdt-current.png")) """

    fig, axs = plt.subplots()
    slice_start = 0
    slice_end = -1
    axs.plot(time_data[slice_start:slice_end], current[slice_start:slice_end], label = "data", color = "cyan")
    axs.plot(time_data[slice_start:slice_end], predicted_current[slice_start:slice_end], label = "pred", color = "red", linestyle = "dotted")
    axs.set_xlabel("time")
    axs.set_ylabel("voltage")
    axs.set_title(f"Repeat {ID} voltage vs time")
    axs.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, str(ID), f"fig{ID}-current-time-whole.png"))

    fig, axs = plt.subplots()
    slice_start = 10550
    slice_end = 20000
    axs.plot(time_data[slice_start:slice_end], current[slice_start:slice_end], label = "data", color = "cyan")
    axs.plot(time_data[slice_start:slice_end], predicted_current[slice_start:slice_end], label = "pred", color = "red", linestyle = "dotted")
    axs.set_xlabel("time")
    axs.set_ylabel("voltage")
    axs.set_title(f"Repeat {ID} voltage vs time")
    axs.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, str(ID), f"fig{ID}-voltage-time-slice-1.png"))

    fig, axs = plt.subplots()
    slice_start = 30550
    slice_end = 40000
    axs.plot(time_data[slice_start:slice_end], current[slice_start:slice_end], label = "data", color = "cyan")
    axs.plot(time_data[slice_start:slice_end], predicted_current[slice_start:slice_end], label = "pred", color = "red", linestyle = "dotted")
    axs.set_xlabel("time")
    axs.set_ylabel("voltage")
    axs.set_title(f"Repeat {ID} voltage vs time")
    axs.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, str(ID), f"fig{ID}-voltage-time-slice-2.png"))

### main ###

#input variables
number_of_repeats = 10

output_filepath = rf"results/20250512-real-curve-fit-voltage-time-36/test-1"
Path(output_filepath).mkdir(parents=True, exist_ok=True)
""" with open(os.path.join(output_filepath, "summary.csv"), "a", newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
                "ID",
                "loss",
                "complexity",
                "MSE",
                "picked_equation",
                "substituted_form",
                ]) """

with open(os.path.join(output_filepath, "summary.csv"), "a", newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
                "ID",
                "MSE",
                "picked_equation",
                "Coeff A",
                "Coeff B",
                "Coeff C",
                "Coeff D",
                ])

for i in range(0,number_of_repeats):
    voltage, dv_dt, current, time_data = loading_data(i)

    start_time = time.time()
    fit(time_data, voltage, sinusoidal_fit, i)
    #select_model = model(voltage, dv_dt, current, time_data, i)
    #print (expand(sympify(str(select_model.sympy()))))
    end_time = time.time()

    time_elapsed = str(datetime.timedelta(seconds = end_time - start_time))
    print(f"Time elapsed: {time_elapsed}")

    with open(os.path.join(output_filepath, "metadata.txt"), "a") as metadata:
        metadata.write(f"model {i} time taken = {time_elapsed} hr:min:sec \n")
