import numpy as np

coeff_array = [00.0, 0, 0]
coeff_array_np = np.array(coeff_array)

bad_equation = 1 if any(coeff_array_np >= 100) else 1 if any(coeff_array_np <= -100) else 0

print(bad_equation)