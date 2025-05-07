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


def loading_data(ID):
    #loading data
    loc=r".\Data_for_eq_learning"
    files=os.listdir(loc)

    data_volt = pd.read_csv("Data_for_eq_learning/9_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
    data_amp = pd.read_csv("Data_for_eq_learning/9_Hz_2_cv_current", sep="\t", names = ["time","current"])
    data_combined_9 = data_volt.join(data_amp["current"])
    data_combined_9.insert(0, "Freq", 9)

    #slicing data to remove start and end noise. Putting data into tuples
    slice_start = 550
    slice_end = -100
    time_data = np.array(data_combined_9["time"].iloc[slice_start:slice_end])
    voltage = tuple(data_combined_9["voltage"].iloc[slice_start:slice_end])
    current = tuple(data_combined_9["current"].iloc[slice_start:slice_end])
    freq = tuple(data_combined_9["Freq"].iloc[slice_start:slice_end])
    
    dv_dt = 17.0298286814874*np.sin(56.644706*time_data)
    y_bisect = 1.0335721e-5*dv_dt
    """     y_reflect = []
    dv_dt_trunc = []

    for i in range(0,len(dv_dt)):
        if current[i] >= y_bisect[i]:
            calc = current[i]
            y_reflect.append(calc)
            dv_dt_trunc.append(dv_dt[i])
        else:
            calc = current[i] + (2*(y_bisect[i]-current[i]))
            y_reflect.append(calc) """

    #make file structure if does not exist
    Path(os.path.join(output_filepath, str(ID))).mkdir(parents = True, exist_ok = True)

    #adding details to metadata
    with open(os.path.join(output_filepath, str(ID), "metadata.txt"), "a") as metadata:
        metadata.write(f"time data: slice start: {slice_start}, slice end: {slice_end} \n")
        metadata.write("voltage formula: dv_dt = 17.0298286814874*np.sin(56.644706*time_data) \n")

    return time_data, voltage

def model(dv_dt, current, ID):

    #shaping variables for pysr
    X = np.array(dv_dt).reshape(-1,1)
    Y = np.array(current).reshape(-1,1)

    #pysr model define
    model = PySRRegressor(
        maxsize=30,
        niterations=100,
        batching= True,
        binary_operators=["+","-","*","/","^"],
        unary_operators=["sin","cos"],
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
                        expand(sympify(str(model.sympy()))),
                        best.loss,
                        best.complexity,
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
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    )
                       \n''')
    
    predicted_current = model.predict(X)
    generate_plots(dv_dt, current, predicted_current, ID)

    return model

def generate_plots(dv_dt, current, predicted_current, ID):
    fig, axs = plt.subplots()
    axs.scatter(dv_dt, predicted_current, label = "prediction", color = "red")
    axs.scatter(dv_dt, current, label = "data", color = "cyan")
    axs.set_xlabel("dV/dt")
    axs.set_ylabel("current")
    axs.set_title(f"Repeat {ID}")
    axs.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, str(ID), f"fig{ID}.png"))

### main ###

#input variables
number_of_repeats = 1

output_filepath = rf"results/20250506-real-test-fit-voltage"
Path(output_filepath).mkdir(parents=True, exist_ok=True)
with open(os.path.join(output_filepath, "summary.csv"), "a", newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
                "ID",
                "picked_equation",
                "loss",
                "complexity",
                ])

for i in range(0,number_of_repeats):
    dv_dt, current = loading_data(i)

    start_time = time.time()
    select_model = model(dv_dt, current, i)
    print (expand(sympify(str(select_model.sympy()))))
    end_time = time.time()

    time_elapsed = str(datetime.timedelta(seconds = end_time - start_time))
    print(f"Time elapsed: {time_elapsed}")

    with open(os.path.join(output_filepath, "metadata.txt"), "a") as metadata:
        metadata.write(f"model {i} time taken = {time_elapsed} hr:min:sec \n")
