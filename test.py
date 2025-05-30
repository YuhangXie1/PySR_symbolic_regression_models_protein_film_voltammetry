import numpy as np
from sympy import Symbol, sympify, symbols, diff, Wild
import sympy as sp
from sympy.utilities.lambdify import lambdify
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import os
from copy import deepcopy

c1, c2, c3, c4, c5, c6, c7, A, f, x0 = symbols("c1 c2 c3 c4 c5 c6 c7 A f x0", real = True)
eqn = sympify("pi*A**4*c1*f*sin(x0) + 5", locals={"x0":x0})
print(eqn)

C = Wild("C")

const = eqn.as_independent(sp.sin(x0))
print(const[0])

""" coeff_dict = {}
trig_search_array = [sp.sin(x0), sp.cos(x0)]

for item in trig_search_array:
    coeff = eqn.match(C*item)
    coeff_dict[item] = coeff


print(coeff_dict) """

