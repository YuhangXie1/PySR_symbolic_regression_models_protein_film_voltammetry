import numpy as np
from sympy import Symbol, sympify, symbols
import sympy as sp
from sympy.utilities.lambdify import lambdify
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt


MSE_array = [0.1,0.2,1.1,1.2]



x = Symbol("x")
eqn_rm_array = [[x,x**2,x**3], [1,x,x**2],[1,x],[1]]
order_by_pow_coeff = [x**3, x**2, x, 1]

for coeff_array_index in range(0, len(eqn_rm_array)):
    coeff_array = eqn_rm_array[coeff_array_index]
    coeff_array = [term for term in order_by_pow_coeff[::-1] if term in coeff_array]
    for coeff_index in range(0,len(coeff_array)):
        coeff_array[coeff_index] = "".join([str(coeff_array[coeff_index]), "\n"])
    eqn_rm_array[coeff_array_index] = coeff_array
    eqn_rm_array[coeff_array_index].insert(0,"".join([str(coeff_array_index),"\n\n"]))
    eqn_rm_array[coeff_array_index] = "".join(eqn_rm_array[coeff_array_index])


print(eqn_rm_array)

fig, axs = plt.subplots()
axs.plot(range(0,len(MSE_array)), MSE_array, color = "red")
axs.set_xlabel("Number of terms removed\n Terms remaining")
axs.set_xticks(range(0,len(MSE_array)), eqn_rm_array)

fig.tight_layout()
plt.show()
