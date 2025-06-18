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

def model(X, Y, ID):
    """
    Runs a PySR model to fit ([voltage, dv_dt], current) and returns the best model.
    """

    X_shaped = np.array(X).reshape(-1,1)
    Y_shaped = np.array(Y).reshape(-1,1)

    #pysr model definition
    model = PySRRegressor(
        maxsize=30,
        niterations=100,
        batching= True,
        binary_operators=["+","*"],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    )

    model.fit(X_shaped,Y_shaped)
    predicted_y = model.predict(X_shaped)

    #writing model parameters to metadata
    Path(os.path.join(output_filepath, str(ID))).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_filepath, str(ID), "metadata.txt"), "a") as metadata:
        metadata.write(f"model run ID: {model.run_id_}")
        metadata.write('''
        X = np.array(X).reshape(-1,1)
        Y = np.array(Y).reshape(-1,1)

        #pysr model definition
        model = PySRRegressor(
            maxsize=30,
            niterations=100,
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
        x0, x1, x2, x3, x4, x5 = symbols("x0 x1 x2 x3 x4 x5")
        f = symbols("f") 
        substituted_form = expand(expanded_form.subs([(x0,f)]))

        writer = csv.writer(file)
        writer.writerow([
                        ID,
                        best.loss,
                        best.complexity,
                        calculate_MSE(Y, predicted_y),
                        expanded_form,
                        substituted_form,
                        ])


    fig, axs = plt.subplots()
    axs.plot(X, Y, label = "data", color = "blue")
    axs.plot(X, predicted_y, label = "prediction", color = "red")
    axs.set_xlabel("Frequency")
    axs.set_ylabel("Amplitude")
    axs.set_title(f"Repeat {ID}.")
    axs.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(output_filepath, str(ID), f"fig{ID}-phase-amplitude.png"))

    return model

### main ###

#input variables
number_of_repeats = 10

output_filepath = rf"results/20250617-fit-phase-amplitude/amplitude"
load_filepath = rf"results\20250617-current-phase-original-set\summary.csv"

Path(output_filepath).mkdir(parents=True, exist_ok=True)
with open(os.path.join(output_filepath, "summary_current_model.csv"), "a", newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
                "ID",
                "loss",
                "complexity",
                "MSE",
                "picked_equation",
                "substituted_form",
                ])


data = pd.read_csv(load_filepath, header=0)

freq = np.array(data["frequency"])
amplitude = np.array(data["amplitude"])
phase = np.array(data["phase"])

for i in range(0,number_of_repeats):
    select_model = model(freq, amplitude, i)
    print (expand(select_model.sympy()))
