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


""" filepath = "results/20250428-poly-2-no-noise_2"
file = pd.read_csv(os.path.join(filepath, "summary.csv"))
set_equation = file["set_equation"].to_numpy()
picked_equation = file["picked_equation"].to_numpy()

x0 = Symbol("x0")
set_equation_sympy = sympify(set_equation)
picked_equation_sympy_non_expanded = sympify(picked_equation)
picked_equation_sympy = [expand(element) for element in picked_equation_sympy_non_expanded]

output = []
for i in range(0,len(set_equation_sympy)):
    try:
        picked_coeff_array = [picked_equation_sympy[i].coeff(x0,0), picked_equation_sympy[i].coeff(x0,1), picked_equation_sympy[i].coeff(x0,2)]
        set_coeff_array = [set_equation_sympy[i].coeff(x0,0), set_equation_sympy[i].coeff(x0,1), set_equation_sympy[i].coeff(x0,2)]
        percentage_diff = lambda pred, real: ((pred - real)/real)*100
        percentage_diff_array = [percentage_diff(picked_coeff_array[j],set_coeff_array[j]) for j in range(0,len(set_coeff_array))]
        output.append(percentage_diff_array)
    except:
        print(f"{picked_equation_sympy[i]} is not a polynomial")
        output.append([np.nan,np.nan,np.nan])


output_df = pd.DataFrame(output, columns=["%diff_to_set_coeff_0","%diff_to_set_coeff_1","%diff_to_set_coeff_2"])

file = file.drop(columns=["%diff_to_set_coeff_0","%diff_to_set_coeff_1","%diff_to_set_coeff_2"], errors="ignore")
new_file = pd.concat([file,output_df], axis=1)
new_file.to_csv(os.path.join(filepath, "summary.csv")) """

x0 = Symbol("x0")
eqn = "x0 - (-6.0062575)*x0*x0 + x0*14.809779 - 1*55.2884 - 6.072818"
eqn_s = sympify(eqn)
eqn_s_e = expand(eqn_s)
print(eqn_s)
print(eqn_s_e)
print([eqn_s.coeff(x0,0),eqn_s.coeff(x0,1),eqn_s.coeff(x0,2)])
