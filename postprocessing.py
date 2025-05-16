import os
import numpy as np
import pandas as pd
from sympy import Symbol, symbols, sympify
from itertools import product


def cos_similarity(x,y):
    return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))



file_loc = rf"results/20250513-fit-workflow/"
files_freq = [9, 36, 45, 54, 63, 72, 81, 90, 99]


data = pd.read_csv(os.path.join(file_loc, str(9),"summary_current_model.csv"))

x, dx = symbols("x dx")
data_equations = sympify(np.array(data["substituted_form"]))

equation_coeffs_dict = [eqns.as_coefficients_dict() for eqns in data_equations]
pairwise_comparison = set([tuple(sorted(i)) for i in list(product(range(0,len(equation_coeffs_dict)),range(0,len(equation_coeffs_dict)))) if i[0] != i[1]])


all_keys = []
for d in equation_coeffs_dict:
    for i in d.keys():
        all_keys.append(i)
all_keys_no_duplicate = set(all_keys)

term_similarity_df = pd.DataFrame(index= range(0,len(equation_coeffs_dict)), columns= range(0, len(equation_coeffs_dict)))

for pair in pairwise_comparison:
    dict_one = set(equation_coeffs_dict[pair[0]])
    dict_two = set(equation_coeffs_dict[pair[1]])

    shared_terms = dict_one.intersection(dict_two)

    term_similarity_df.at[pair[0],pair[1]] = int(len(shared_terms)/len(all_keys_no_duplicate)*100)

print(term_similarity_df)

for key in all_keys_no_duplicate:
    for d in equation_coeffs_dict:
        if key not in d:
            d[key] = 0


coeff_similarity_df = pd.DataFrame(index= range(0,len(equation_coeffs_dict)), columns= range(0, len(equation_coeffs_dict)))

for pair in pairwise_comparison:
    dict_one = equation_coeffs_dict[pair[0]]
    dict_two = equation_coeffs_dict[pair[1]]

    x = np.array([float(i) for i in dict_one.values()])
    y = np.array([float(i) for i in dict_two.values()])

    coeff_similarity_df.at[pair[0],pair[1]] = cos_similarity(x,y)
    #pair[0] is row number. pair[1] is column number

print(coeff_similarity_df)


