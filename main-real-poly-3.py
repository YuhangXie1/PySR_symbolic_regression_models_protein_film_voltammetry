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

    data_volt = pd.read_csv("Data_for_eq_learning/36_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
    data_amp = pd.read_csv("Data_for_eq_learning/36_Hz_2_cv_current", sep="\t", names = ["time","current"])
    data_combined_36 = data_volt.join(data_amp["current"])
    data_combined_36.insert(0, "Freq", 36)

    data_volt = pd.read_csv("Data_for_eq_learning/9_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
    data_amp = pd.read_csv("Data_for_eq_learning/9_Hz_2_cv_current", sep="\t", names = ["time","current"])
    data_combined_9 = data_volt.join(data_amp["current"])
    data_combined_9.insert(0, "Freq", 9)

    data_combined = pd.concat([data_combined_9, data_combined_36])

    #slicing data to remove start and end noise. Putting data into tuples
    slice_start = 550
    slice_end = -100
    time_data = np.array(data_combined["time"].iloc[slice_start:slice_end])
    voltage = np.array(data_combined["voltage"].iloc[slice_start:slice_end])
    current = np.array(data_combined["current"].iloc[slice_start:slice_end])
    freq = np.array(data_combined["Freq"].iloc[slice_start:slice_end])

    #dv_dt = 17.0298286814874*np.sin(56.644706*time_data)
    #make file structure if does not exist
    Path(os.path.join(output_filepath, str(ID))).mkdir(parents = True, exist_ok = True)

    #adding details to metadata
    with open(os.path.join(output_filepath, str(ID), "metadata.txt"), "a") as metadata:
        metadata.write(f"time data: slice start: {slice_start}, slice end: {slice_end} \n")
        #metadata.write("voltage formula: dv_dt = 17.0298286814874*np.sin(56.644706*time_data) \n")

    return time_data, voltage, freq

def model(time_data, voltage, freq, ID):

    #shaping variables for pysr
    X = np.array([time_data,freq]).T
    Y = np.array(voltage).reshape(-1,1)

    #pysr model define
    model = PySRRegressor(
        maxsize=30,
        niterations=100,
        batching= True,
        binary_operators=["+","-","*","/","^"],
        unary_operators=["sin","cos"],
        nested_constraints={"sin": {"sin": 0, "cos": 0}, "cos": {"sin": 0, "cos": 0}},
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
        metadata.write('''
    X = np.array([time_data,freq]).T
    Y = np.array(voltage).reshape(-1,1)

    #pysr model define
    model = PySRRegressor(
        maxsize=30,
        niterations=100,
        batching= True,
        binary_operators=["+","-","*","/","^"],
        unary_operators=["sin","cos"],
        nested_constraints={"sin": {"sin": 0, "cos": 0}, "cos": {"sin": 0, "cos": 0}},
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    )
                       \n''')
    
    predicted_y = model.predict(X)
    generate_plots(time_data, voltage, predicted_y, ID)

    return model

def generate_plots(x, y, predicted_y, ID):
    fig, axs = plt.subplots()
    axs.plot(x, y, label = "data", color = "cyan")
    axs.plot(x, predicted_y, label = "prediction", color = "red")
    axs.set_xlabel("time")
    axs.set_ylabel("voltage")
    axs.set_title(f"Repeat {ID} whole")
    axs.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, str(ID), f"fig{ID}_whole.png"))

    fig, axs = plt.subplots()
    slice_start = 10550
    slice_end = 20000
    axs.plot(x[slice_start:slice_end], y[slice_start:slice_end], label = "data", color = "cyan")
    axs.plot(x[slice_start:slice_end], predicted_y[slice_start:slice_end], label = "prediction", color = "red", linestyle = "dotted")
    axs.set_xlabel("time")
    axs.set_ylabel("voltage")
    axs.set_title(f"Repeat {ID} Slice {slice_start} to {slice_end}")
    axs.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, str(ID), f"fig{ID}-slice-1.png"))

    fig, axs = plt.subplots()
    slice_start = 40550
    slice_end = 50000
    axs.plot(x[slice_start:slice_end], y[slice_start:slice_end], label = "data", color = "cyan")
    axs.plot(x[slice_start:slice_end], predicted_y[slice_start:slice_end], label = "prediction", color = "red", linestyle = "dotted")
    axs.set_xlabel("time")
    axs.set_ylabel("voltage")
    axs.set_title(f"Repeat {ID} Slice {slice_start} to {slice_end}")
    axs.legend()
    axs.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, str(ID), f"fig{ID}-slice-2.png"))

### main ###

#input variables
number_of_repeats = 30

output_filepath = rf"results/20250512-real-test-fit-voltage-multiplefiles/test-2"
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
    time_data, voltage, freq = loading_data(i)

    start_time = time.time()
    select_model = model(time_data, voltage, freq, i)
    print (expand(sympify(str(select_model.sympy()))))
    end_time = time.time()

    time_elapsed = str(datetime.timedelta(seconds = end_time - start_time))
    print(f"Time elapsed: {time_elapsed}")

    with open(os.path.join(output_filepath, "metadata.txt"), "a") as metadata:
        metadata.write(f"model {i} time taken = {time_elapsed} hr:min:sec \n")
