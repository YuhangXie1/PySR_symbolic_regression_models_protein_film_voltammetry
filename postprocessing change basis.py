import os
import numpy as np
import pandas as pd
from sympy import Symbol, symbols, sympify, trigsimp, expand, simplify, factor, collect
import sympy as sp
from itertools import product


file_loc = rf"results/20250513-fit-workflow/"
files_freq = [9, 36, 45, 54, 63, 72, 81, 90, 99]

data = pd.read_csv(os.path.join(file_loc, str(36),"summary_current_model.csv"))

#defining symbols
x, dx  = symbols("x dx")
x0 = symbols("x0", positive = True, real = True)
A, c, f, w = symbols("A, c, f w", real = True)
#data_equations = sympify(np.array(data["substituted_form"]))

#A = 0.3. c = 0.05. f = 36
#w = 2*pi*f
voltage_eqn = A*sp.sin(w*x0 - sp.pi/2) - c
dv_dt_eqn = A*w*sp.cos(w*x0 - sp.pi/2)

#current_eqn = sympify("1.83479321254227e-6*dx*x**3 - 7.77746142237813e-7*dx*x**2 - 7.77746142237813e-7*dx*x + 7.5642693e-6*dx + 0.00060018763*x**2 + 0.0010604324*x + 2.4237355e-5")
c1, c2, c3, c4, c5, c6, c7 = symbols("c1 c2 c3 c4 c5 c6 c7", real = True)
current_eqn = c1*dx*x**3 - c2*dx*x**2 - c3*dx*x + c4*dx + c5*x**2 + c6*x + c7
#current_eqn = - c2*dx*x**2 - c3*dx*x + c4*dx + c5*x**2 + c6*x + c7

#rewriting to fit the right format
substituted_form = expand(current_eqn.subs([(x,voltage_eqn),(dx,dv_dt_eqn)]))
print(substituted_form)

print("# fu #")
simplified_form = trigsimp(substituted_form, method = "fu")
print(simplified_form)

print("# factor #")
simplified_form = factor(simplified_form)
print(simplified_form)

print("# collect #")
trig_basis_array = [sp.sin(n*w*x0) for n in range(1,5)] + [sp.cos(n*w*x0) for n in range(1,5)]
simplified_form = collect(simplified_form, trig_basis_array)
print(simplified_form)

#collecting coefficients
coeff_dict = {}
trig_basis_array = [sp.sin(n*w*x0) for n in range(1,5)] + [sp.cos(n*w*x0) for n in range(1,5)]
for item in trig_basis_array:
    coeff = simplified_form.coeff(item)
    coeff_dict[item] = coeff
constant_term = simplified_form.as_independent(*trig_basis_array)
coeff_dict[1] = constant_term[0]

print(coeff_dict) 

