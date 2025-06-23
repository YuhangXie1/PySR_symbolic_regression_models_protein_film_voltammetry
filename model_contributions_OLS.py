import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
from pathlib import Path
import seaborn as sns

#a_val_norm conversion into original MSE. 

#load MSE data
data = pd.read_csv(r"results\20250620-multi-fit-workflow-9\mse_summary.csv", header=0)

#split into non a_val_norm since difference scales if not a_val_norm_ed

data_aval_norm = data[data["a_val_norm"] == 1].copy()
data_aval_norm_not = data[data["a_val_norm"] == 0].copy()


#Features of OLS
X = data_aval_norm_not[["dv/dt_norm", "timeshift", "allow_div"]]
Y = data_aval_norm_not["average_MSE"]

#adding constant
X = sm.add_constant(X)

# Fit OLS regression
model = sm.OLS(Y, X).fit()

# Print regression summary
print(model.summary())

""" output_filepath = r"results\20250620-multi-fit-workflow-9"
filename = "not_aval_linear_reg_summary.txt"
with open(os.path.join(output_filepath, filename), "w") as file:
    file.write(model.summary().as_text()) """


import matplotlib.pyplot as plt

# Predicted values
y_pred = model.fittedvalues

# Actual values
y_true = model.model.endog

plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, color='dodgerblue')
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label="Ideal fit")
plt.xlabel("Actual MSE")
plt.ylabel("Predicted MSE")
plt.title("Actual vs Predicted MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

residuals = model.resid

plt.figure(figsize=(6, 4))
plt.scatter(y_pred, residuals, color='darkorange')
plt.axhline(0, linestyle='--', color='gray')
plt.xlabel("Predicted MSE")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

import seaborn as sns
import pandas as pd

params = model.params
conf = model.conf_int()
summary_df = pd.DataFrame({
    'coef': params.values,
    'lower': conf[0].values,
    'upper': conf[1].values
}, index=params.index)

summary_df = summary_df.drop("const", errors='ignore')  # Remove intercept for clarity

plt.figure(figsize=(6, 4))
sns.pointplot(data=summary_df.reset_index(), x='index', y='coef', join=False)
plt.errorbar(summary_df.index, summary_df['coef'], 
             yerr=[summary_df['coef'] - summary_df['lower'], summary_df['upper'] - summary_df['coef']],
             fmt='none', c='black', capsize=5)
plt.title("Model Coefficients with 95% CI")
plt.xlabel("Variable")
plt.ylabel("Effect on MSE")
plt.grid(True)
plt.tight_layout()
plt.show()
