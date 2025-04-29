from pysr import PySRRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import datetime
from pathlib import Path
import csv
from sympy import Symbol, sympify, expand

def simulate_data(capacitance_array, add_noise, percentage_noise, ID):
    #simulating data
    time_start = 0
    time_end = 100
    time_step = 0.1

    time_data = np.arange(time_start,time_end,time_step)
    voltage = np.sin(time_data)
    dv_dt = np.cos(time_data)

    #simulating current
    current = capacitance_array[0] + capacitance_array[1]*dv_dt + capacitance_array[2] * dv_dt**2 + capacitance_array[3] * dv_dt**3

    #adding noise
    if add_noise:
        gaussian_noise = np.random.normal(0, 1*abs(max(current))*percentage_noise, len(current))
    else:
        gaussian_noise = 0

    current_noise = current + gaussian_noise

    #make file structure if does not exist
    Path(os.path.join(output_filepath, str(ID))).mkdir(parents = True, exist_ok = True)

    #adding details to metadata
    with open(os.path.join(output_filepath, str(ID), "metadata.txt"), "a") as metadata:
        metadata.write(f"time data: start: {time_start}, end: {time_end}, step: {time_step} \n")
        metadata.write("voltage formula: dv_dv = np.cos(time_data) \n")
        metadata.write(f"current formula: current = {capacitance_array[0]} + {capacitance_array[1]}*dv_dt + {capacitance_array[2]} * dv_dt**2 + {capacitance_array[2]} * dv_dt**3 \n")
        metadata.write(f"current noise: current + np.random.normal(0, 1, len(current))*max(current)*{percentage_noise} #percentage noise \n")
        metadata.write(f"median noise percentage: {np.median(np.abs((gaussian_noise/current)*100)) if add_noise else 0} % \n")

    return dv_dt, current_noise, current

def model(x, y, sim_current, capacitance_array, ID):

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
            "sign",],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    )

    """             unary_operators=[
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
        ], """

    model.fit(X,Y)

    #printing and writing to file
    print(model)
    print(model.sympy())
    print(model.equations_)
    Path(os.path.join(output_filepath, str(ID))).mkdir(parents=True, exist_ok=True)
    model.equations_.to_csv(os.path.join(output_filepath, str(ID), "equations.csv"))

    with open(os.path.join(output_filepath, "summary.csv"), "a", newline='') as file:
        best = model.get_best()
        coeff_array = calculate_percent_diff_to_set(model, capacitance_array)
        coeff_array_np = np.array(coeff_array)

        try:
            bad_equation = 1 if any(coeff_array_np >= 100) else 1 if any(coeff_array_np <= -100) else 0
        except TypeError:
            bad_equation = 1

        writer = csv.writer(file)
        writer.writerow([
                        ID,
                        f"{capacitance_array[0]} + {capacitance_array[1]}*x0 + {capacitance_array[2]}*x0**2 + {capacitance_array[3]}*x0**3",
                        expand(sympify(str(model.sympy()))),
                        best.loss,
                        best.complexity,
                        calculate_predict_MSE(dv_dt, sim_current, model, ID),
                        coeff_array[0],
                        coeff_array[1],
                        coeff_array[2],
                        coeff_array[3],
                        bad_equation,
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
            "sign",],
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

def calculate_percent_diff_to_set(model, capacitance_array):
    x0 = Symbol("x0")
    expression = expand(sympify(str(model.sympy())))

    try:
        coeff_array = [expression.coeff(x0,0), expression.coeff(x0,1), expression.coeff(x0,2), expression.coeff(x0,3)]
        percentage_diff = lambda pred, real: ((pred - real)/real)*100
        percentage_diff_array = [percentage_diff(coeff_array[i],capacitance_array[i]) for i in range(0,len(coeff_array))]
        return percentage_diff_array
    
    except:
        print(f"{expression} is not a polynomial")
        return [np.nan,np.nan,np.nan]

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

#input variables [const,x,x**2] for number of repeats
number_of_repeats = 10
add_noise = False
percentage_noise = 0.1
capacitance_array = [[np.random.uniform(-100.0,100.0),np.random.uniform(-100.0,100.0),np.random.uniform(-100.0,100.0),np.random.uniform(-100.0,100.0)] for x in range(number_of_repeats)]

#output_filepath = rf"results/20250428-poly-2-{percentage_noise*100}"
output_filepath = rf"results/20250428-poly-3-{percentage_noise*100 if add_noise else 0}"
Path(output_filepath).mkdir(parents=True, exist_ok=True)
with open(os.path.join(output_filepath, "summary.csv"), "a", newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
                "ID",
                "set_equation",
                "picked_equation",
                "loss",
                "complexity",
                "MSE",
                "%diff_to_set_coeff_0",
                "%diff_to_set_coeff_1",
                "%diff_to_set_coeff_2",
                "%diff_to_set_coeff_3",
                "bad_equation",
                ])

for i in range(0,len(capacitance_array)):
    dv_dt, current_noise, current = simulate_data(capacitance_array[i], add_noise, percentage_noise, ID=i)

    start_time = time.time()
    select_model = model(dv_dt,current_noise, current, capacitance_array[i], ID=i)
    print (str(select_model.sympy()))
    end_time = time.time()

    time_elapsed = str(datetime.timedelta(seconds = end_time - start_time))
    print(f"Time elapsed: {time_elapsed}")

    with open(os.path.join(output_filepath, "metadata.txt"), "a") as metadata:
        metadata.write(f"model {i} time taken = {time_elapsed} hr:min:sec \n")
