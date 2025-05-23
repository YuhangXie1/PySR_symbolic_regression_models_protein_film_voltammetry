from pysr import PySRRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from sympy import Symbol, sympify, expand, diff, symbols
from sympy.utilities.lambdify import lambdify
from pathlib import Path
import csv

def calculate_MSE(y, y_pred):

    mse = sum((y_pred - y)**2)
    print(f"MSE:{mse}")

    return mse

start_time = time.time()

loc=r".\Data_for_eq_learning"
files=os.listdir(loc)

""" data_complete = pd.DataFrame
for file in files[:1]:
    if "FTacV" in file:
        continue
    if "Hz" in file:
        split_file = file.split("_") 

        freq = split_file[:1]
        current_file = "_".join(freq+["Hz","2", "cv","current"])
        potential_file = "_".join(freq+["Hz","2", "cv","voltage"])
        
        data_potential = pd.read_csv(os.path.join(loc, potential_file), sep="\t", names = ["time","voltage"])
        data_current = pd.read_csv(os.path.join(loc, current_file), sep="\t", names = ["time","current"])
        data_combined = data_potential.join(data_current["current"])
        data_combined.insert(0, "freq", freq) """


#loading data
file_loc = rf"results/20250513-fit-workflow/"
x0 = Symbol("x0")

Hz_1 = 36
data_volt_1 = pd.read_csv(f"Data_for_eq_learning/{Hz_1}_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
data_amp_1 = pd.read_csv(f"Data_for_eq_learning/{Hz_1}_Hz_2_cv_current", sep="\t", names = ["time","current"])
data_combined_1 = data_volt_1.join(data_amp_1["current"])
data_combined_1.insert(0, "Freq", Hz_1)

voltage_eqn_file = pd.read_csv(os.path.join(file_loc, str(Hz_1), "summary_voltage_model.csv"))
voltage_sympy = sympify(np.array(voltage_eqn_file["picked_equation"])[0])
dv_dt_func = diff(voltage_sympy, x0)
dv_dt_lambda_1 = lambdify(x0,dv_dt_func)

#files_freq = [9, 36, 45, 54, 63, 72, 81, 90, 99]
Hz_2 = 54
data_volt_2 = pd.read_csv(f"Data_for_eq_learning/{Hz_2}_Hz_2_cv_voltage", sep="\t", names = ["time","voltage"])
data_amp_2 = pd.read_csv(f"Data_for_eq_learning/{Hz_2}_Hz_2_cv_current", sep="\t", names = ["time","current"])
data_combined_2 = data_volt_2.join(data_amp_2["current"])
data_combined_2.insert(0, "Freq", Hz_2)

voltage_eqn_file = pd.read_csv(os.path.join(file_loc, str(Hz_2), "summary_voltage_model.csv"))
voltage_sympy = sympify(np.array(voltage_eqn_file["picked_equation"])[0])
dv_dt_func = diff(voltage_sympy, x0)
dv_dt_lambda_2 = lambdify(x0,dv_dt_func)

#combining data
""" data_combined = pd.concat([data_combined_9, data_combined_36]) """

#slicing data to remove start and end noise. Putting data into tuples
#slice_start = 550
#slice_end = -100
slice_start = 550
slice_end = -100

time_data_1 = np.array(data_combined_1["time"].iloc[slice_start:slice_end])
voltage_1 = np.array(data_combined_1["voltage"].iloc[slice_start:slice_end])
current_1 = np.array(data_combined_1["current"].iloc[slice_start:slice_end])
dv_dt_1 = dv_dt_lambda_1(time_data_1)

time_data_2 = np.array(data_combined_2["time"].iloc[slice_start:slice_end])
voltage_2 = np.array(data_combined_2["voltage"].iloc[slice_start:slice_end])
current_2 = np.array(data_combined_2["current"].iloc[slice_start:slice_end])
dv_dt_2 = dv_dt_lambda_2(time_data_2)


#shaping variables for pysr
X = np.array(time_data_1).reshape(-1,1)
y = np.array(voltage_1).reshape(-1,1)

""" freq_x = np.array(freq_pos).reshape(-1,1)
X = np.concatenate([X,freq_x],1) """


#pysr model define
model = PySRRegressor(
    maxsize=30,
    niterations=100,
    batching= True,
    binary_operators=["+","*","^"],
    unary_operators=["sin","cos"],
    #nested_constraints={"sin":{"sin":0}},
    elementwise_loss="loss(prediction, target) = (prediction - target)^2"
)

run = False
if run:     
    model.fit(X,y)
    print(model)
    print(model.sympy())
    
#voltage_pred = model.predict(X)
end_time = time.time()
#print(f"Time elapsed: {end_time - start_time}")


#calculating predicted current values
data = pd.read_csv(os.path.join(file_loc, str(Hz_1),"summary_current_model.csv"))
data_equations = sympify(np.array(data["substituted_form"]))
output_filepath = rf"results/20250522-{Hz_1}-predict-other-hz/{Hz_2}/"

Path(os.path.join(output_filepath)).mkdir(parents=True, exist_ok=True)
with open(os.path.join(output_filepath, "summary.csv"), "a", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Eqn",
                    "MSE_1",
                    "MSE_2",
                    ])

for eqn_number, eqn in enumerate(data_equations):
    picked_eqn = data_equations[eqn_number]

    x, dx = symbols("x dx")
    eqn_lambda = lambdify([x,dx],picked_eqn)
    current_pred_1 = eqn_lambda(voltage_1, dv_dt_1)
    current_pred_2 = eqn_lambda(voltage_2, dv_dt_2)

    current_MSE_1 = calculate_MSE(current_1, current_pred_1)
    current_MSE_2 = calculate_MSE(current_2, current_pred_2)

    with open(os.path.join(output_filepath, f"summary.csv"), "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([eqn_number,
                        current_MSE_1,
                        current_MSE_2,
                        ])
    
    fig, axs = plt.subplots(3)
    axs[0].plot(time_data_1, current_1, label = "data", color = "blue")
    axs[0].plot(time_data_1, current_pred_1, label = "prediction", color = "red")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("current")
    axs[0].set_title(f"{Hz_1}Hz. Eqn {eqn_number}. Current vs time. Whole. ")
    axs[0].legend()

    slice_start = 10550
    slice_end = 20000
    axs[1].plot(time_data_1[slice_start:slice_end], current_1[slice_start:slice_end], label = "data", color = "blue")
    axs[1].plot(time_data_1[slice_start:slice_end], current_pred_1[slice_start:slice_end], label = "pred", color = "red", linestyle = "dotted")
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("current")
    axs[1].set_title(f"{Hz_1}Hz. Eqn {eqn_number}. Current vs time. Slice {slice_start}-{slice_end}.")
    axs[1].legend()

    slice_start = 30550
    slice_end = 40000
    axs[2].plot(time_data_1[slice_start:slice_end], current_1[slice_start:slice_end], label = "data", color = "blue")
    axs[2].plot(time_data_1[slice_start:slice_end], current_pred_1[slice_start:slice_end], label = "pred", color = "red", linestyle = "dotted")
    axs[2].set_xlabel("time")
    axs[2].set_ylabel("current")
    axs[2].set_title(f"{Hz_1}Hz. Eqn {eqn_number}. Current vs time. Slice {slice_start}-{slice_end}.")
    axs[2].legend()

    fig.tight_layout()
    Path(os.path.join(output_filepath, f"eqn_{eqn_number}")).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(output_filepath, f"eqn_{eqn_number}", f"{Hz_1}Hz_eqn-{eqn_number}-current-time.png"))
    plt.close(fig.figure)

    #current dV/dt graph
    fig, axs = plt.subplots()
    axs.plot(dv_dt_1, current_1, label = "data", color = "blue")
    axs.plot(dv_dt_1, current_pred_1, label = "prediction", color = "red")
    axs.set_xlabel("dV/dt")
    axs.set_ylabel("current")
    axs.set_title(f"{Hz_1}Hz. Eqn {eqn_number}. Current vs dV/dt. \n MSE = {current_MSE_1}.")
    axs.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, f"eqn_{eqn_number}", f"{Hz_1}Hz_eqn-{eqn_number}-current-dvdt.png"))
    plt.close(fig.figure)

    fig, axs = plt.subplots(3)
    axs[0].plot(time_data_2, current_2, label = "data", color = "blue")
    axs[0].plot(time_data_2, current_pred_2, label = "prediction", color = "red")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("current")
    axs[0].set_title(f"{Hz_2}Hz. Current vs time. Whole. Prediction from {Hz_1}Hz Eqn {eqn_number}")
    axs[0].legend()

    slice_start = 10550
    slice_end = 20000
    axs[1].plot(time_data_2[slice_start:slice_end], current_2[slice_start:slice_end], label = "data", color = "blue")
    axs[1].plot(time_data_2[slice_start:slice_end], current_pred_2[slice_start:slice_end], label = "pred", color = "red", linestyle = "dotted")
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("current")
    axs[1].set_title(f"{Hz_2}Hz. Current vs time. Slice {slice_start}-{slice_end}. Prediction from {Hz_1}Hz Eqn {eqn_number}")
    axs[1].legend()

    slice_start = 30550
    slice_end = 40000
    axs[2].plot(time_data_2[slice_start:slice_end], current_2[slice_start:slice_end], label = "data", color = "blue")
    axs[2].plot(time_data_2[slice_start:slice_end], current_pred_2[slice_start:slice_end], label = "pred", color = "red", linestyle = "dotted")
    axs[2].set_xlabel("time")
    axs[2].set_ylabel("current")
    axs[2].set_title(f"{Hz_2}Hz. Current vs time. Slice {slice_start}-{slice_end}. Prediction from {Hz_1}Hz Eqn {eqn_number}")
    axs[2].legend()

    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, f"eqn_{eqn_number}", f"{Hz_2}Hz_eqn-{eqn_number}-current-time.png"))
    plt.close(fig.figure)

    #current dV/dt graph
    fig, axs = plt.subplots()
    axs.plot(dv_dt_2, current_2, label = "data", color = "blue")
    axs.plot(dv_dt_2, current_pred_2, label = "prediction", color = "red")
    axs.set_xlabel("dV/dt")
    axs.set_ylabel("current")
    axs.set_title(f"{Hz_2}Hz. Current vs dV/dt. Prediction from {Hz_1}Hz Eqn {eqn_number}. \n MSE = {current_MSE_2}.")
    axs.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, f"eqn_{eqn_number}", f"{Hz_2}Hz_eqn-{eqn_number}-current-dvdt.png"))
    plt.close(fig.figure)