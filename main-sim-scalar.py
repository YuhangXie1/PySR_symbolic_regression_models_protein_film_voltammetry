from pysr import PySRRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import datetime
from pathlib import Path
import csv
from sympy import Symbol, sympify

def simulate_data_scalar(capacitance_scalar, percentage_noise, ID):
    #simulating data
    time_start = 0
    time_end = 100
    time_step = 0.1

    time_data = np.arange(time_start,time_end,time_step)
    voltage = np.sin(time_data)
    dv_dt = np.cos(time_data)

    #simulating current
    current = capacitance_scalar*dv_dt

    #adding noise
    gaussian_noise = np.random.normal(0, 1*max(current)*percentage_noise, len(current))
    current_noise = current + gaussian_noise

    #make file structure if does not exist
    Path(os.path.join(output_filepath, str(ID))).mkdir(parents = True, exist_ok = True)

    #adding details to metadata
    with open(os.path.join(output_filepath, str(ID), "metadata.txt"), "a") as metadata:
        metadata.write(f"time data: start: {time_start}, end: {time_end}, step: {time_step} \n")
        metadata.write("voltage formula: dv_dv = np.cos(time_data) \n")
        metadata.write(f"current formula: current = {capacitance_scalar}*dv_dt \n")
        metadata.write(f"current noise: current + np.random.normal(0, 1, len(current))*max(current)*{percentage_noise} #percentage noise \n")
        metadata.write(f"median noise percentage: {np.median(np.abs((gaussian_noise/current)*100))} % \n")

    return dv_dt, current_noise, current

def model(x, y, sim_current, capacitance_scalar, ID):

    #shaping variables for pysr
    dv_dt = x
    X = np.array(x).reshape(-1,1)
    Y = np.array(y).reshape(-1,1)

    #pysr model define
    model = PySRRegressor(
        maxsize=30,
        niterations=100,
        batching= True,
        binary_operators=["+","-","*","/","^"],
        unary_operators=[
            "cos",
            "sin",
            "tan",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "asinh",
            "acosh",
            "atanh",
            "exp",
            "log",
            "inv",
            "neg",
            "abs",
            "sign",
        ],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    )

    model.fit(X,Y)

    #printing and writing to file
    print(model)
    print(model.sympy())
    print(model.equations_)
    Path(os.path.join(output_filepath, str(ID))).mkdir(parents=True, exist_ok=True)
    model.equations_.to_csv(os.path.join(output_filepath, str(ID), "equations.csv"))

    with open(os.path.join(output_filepath, "summary.csv"), "a", newline='') as file:
        best = model.get_best()
        writer = csv.writer(file)
        writer.writerow([
                        ID,
                        str(model.sympy()),
                        best.loss,
                        best.complexity,
                        calculate_predict_MSE(dv_dt, sim_current, model, ID),
                        calculate_percent_diff_to_scalar(model, capacitance_scalar),
                        ])

    #writing model parameters to metadata
    with open(os.path.join(output_filepath, str(ID), "metadata.txt"), "a") as metadata:
        metadata.write(f"model run ID: {model.run_id_}")
        metadata.write(f'''
        
                       model = PySRRegressor(
        maxsize=30,
        niterations=100,
        batching= True,
        binary_operators=["+","-","*","/","^"],
        unary_operators=[
            "cos",
            "sin",
            "tan",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "asinh",
            "acosh",
            "atanh",
            "exp",
            "log",
            "inv",
            "neg",
            "abs",
            "sign",
        ],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    )
                       \n''')


    return model
    
def calculate_predict_MSE(dv_dt, current, model, ID):
    X = np.array(dv_dt).reshape(-1,1)
    #current = np.array(current).reshape(-1,1)
    predicted_current = model.predict(X)
    mse = sum((predicted_current - current)**2)

    generate_plots(dv_dt, current, predicted_current, ID)

    print(f"MSE:{mse}")
    return mse

def calculate_percent_diff_to_scalar(model, capacitance_scalar):
    x0 = Symbol("x0")
    expression = sympify(str(model.sympy()))
    coeff = expression.coeff(x0)
    percentage_diff = ((coeff - capacitance_scalar)/capacitance_scalar)*100

    return percentage_diff

def generate_plots(dv_dt, current, predicted_current, ID):
    fig, axs = plt.subplots()
    axs.plot(dv_dt, predicted_current, label = "prediction", color = "red")
    axs.plot(dv_dt, current, label = "data", color = "cyan")
    axs.set_xlabel("dV/dt")
    axs.set_ylabel("current")
    axs.set_title(f"Repeat {ID}")
    axs.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, str(ID), f"fig{ID}.png"))

### main ###

output_filepath = r"results/20250426-scalar-denoise-10"
Path(output_filepath).mkdir(parents=True, exist_ok=True)
with open(os.path.join(output_filepath, "summary.csv"), "a", newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
                "ID",
                "picked_equation",
                "loss",
                "complexity",
                "MSE",
                "%diff_to_set",
                ])
#os.path.join(loc, potential_file)

capacitance_scalar = np.random.uniform(-100.0,100.0,10)
percentage_noise = 0.1

for i in range(0,len(capacitance_scalar)):
    dv_dt, current_noise, current = simulate_data_scalar(capacitance_scalar[i], percentage_noise, ID=i)

    start_time = time.time()
    select_model = model(dv_dt,current_noise, current, capacitance_scalar[i], ID=i)
    print (str(select_model.sympy()))
    end_time = time.time()

    time_elapsed = str(datetime.timedelta(seconds = end_time - start_time))
    print(f"Time elapsed: {time_elapsed}")

    with open(os.path.join(output_filepath, "metadata.txt"), "a") as metadata:
        metadata.write(f"model {i} time taken = {time_elapsed} hr:min:sec \n")