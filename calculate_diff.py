from pysr import PySRRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import datetime
from pathlib import Path
import csv
from sympy import Symbol, sympify, expand, symbols
from sympy.utilities.lambdify import lambdify


filepath = "results/20250508-real-test-fit-dv-and-v/test-3-setpow-more-constrained-1"
file = pd.read_csv(os.path.join(filepath, "summary.csv"))
picked_equation = file["picked_equation"].to_numpy()

x0, x1, x2, x3, x4, x5 = symbols("x0 x1 x2 x3 x4 x5")
x, dx = symbols("x dx")
picked_equation_sympy_non_expanded = sympify(picked_equation)
picked_equation_sympy_non_expanded = picked_equation_sympy_non_expanded.subs([(x0,x),(x1,x**2),(x2,x**3),(x3,dx),(x4,dx**2),(x5,dx**3)])

picked_equation_sympy = [expand(element) for element in picked_equation_sympy_non_expanded]

picked_equation_one = picked_equation_sympy[0]
func = lambdify([x,dx],picked_equation_one,"numpy")

""" output = []
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
"""

output_df = pd.DataFrame(picked_equation_sympy, columns=["simplified eqn"])
new_file = pd.concat([file,output_df], axis=1)
new_file.to_csv(os.path.join(filepath, "summary_editted.csv"))
