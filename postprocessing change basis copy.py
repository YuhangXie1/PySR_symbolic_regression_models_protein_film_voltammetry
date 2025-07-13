import os
import numpy as np
import pandas as pd
from sympy import Symbol, symbols, sympify, trigsimp, expand, simplify, factor, collect, diff
import sympy as sp
from itertools import product


file_loc = rf"results\20250616-CjX-fit-workflow-2\blank\20250611-blank-PSV-9Hz"
#files_freq = [9, 36, 45, 54, 63, 72, 81, 90, 99]

data = pd.read_csv(os.path.join(file_loc,"summary_voltage_model.csv"))

#defining symbols
x, dx  = symbols("x dx")
x0, t = symbols("x0 t", positive = True, real = True)
A, w, p, c, A_val, p_val = symbols("A w p c A_val p_val", real = True)


voltage_eqn = A*sp.sin(w*t - sp.pi/2) + c
dv_dt_eqn = A*sp.cos(w*t - sp.pi/2) #normalised

c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = symbols("c1 c2 c3 c4 c5 c6 c7 c8 c9 c10", real = True)
current_eqn = "c1*dx*p_val*x**2 + c2*dx*p_val*x + c3*dx*x**2 + c4*dx*x + c5*dx - c6*p*p_val*x**2 - c7*p*x**2 - c8*p_val*x + c9*x**2 - c10"
current_eqn = sympify(current_eqn)


 

print("# Substituted #")
#rewriting to fit the right format
substituted_form = expand(current_eqn.subs([(x,voltage_eqn),(dx,dv_dt_eqn)]))
#set_eqn = "2.7175533e-5*c*sin(2*t*w)*sin(5*t*w) + 5.4351066e-5*c*sin(2*t*w)*cos(t*w) + 2.7175533e-5*c*sin(2*t*w)*cos(2*t*w) - 0.000104288472158871*c*sin(2*t*w) + 0.0005261732*sin(t*w) - 0.00028988792*cos(t*w) + 2.7175533e-5*cos(2*t*w)"
#substituted_form = sympify(set_eqn)
print(substituted_form)

print("# fu #")
simplified_form = trigsimp(substituted_form, method = "fu")
print(simplified_form)

print("# factor #")
simplified_form = factor(simplified_form)
print(simplified_form)

print("# collect #")
num = 6
trig_basis_array = [sp.sin(n*w*x0) for n in range(1,num+1)] + [sp.cos(n*w*x0) for n in range(1,num+1)]
simplified_form = collect(simplified_form, trig_basis_array)
print(simplified_form)

#collecting coefficients
coeff_dict = {}
for item in trig_basis_array:
    coeff = simplified_form.coeff(item)
    coeff_dict[item] = coeff
constant_term = simplified_form.as_independent(*trig_basis_array)
coeff_dict[1] = constant_term[0]

print(coeff_dict) 