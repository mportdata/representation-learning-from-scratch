import numpy as np
import matplotlib.pyplot as plt

from course_utils.plots import plot_seq_with_forecast

from observed_data import ideal_seq, noisy_seq

#seq = ideal_seq  # [1, 2, 3, 4, 5]
seq = noisy_seq  # [0, 3, 2, 5, 1]

# 1. Create input (x) and target (y) pairs where each x is mapped to the next value in the sequence
x = seq[:-1]
y = seq[1:]

# 2. Build the design matrix
X = np.column_stack([x, np.ones_like(x)])

# 3. Solve the normal equation
theta = np.linalg.inv(X.T @ X) @ X.T @ y

# 4. Extract the parameters
m, b = theta

# 5. Forecast 1 step ahead using the AR(1) model
prediction_step_1 = x[-1] * m + b
prediction_step_2 = prediction_step_1 * m + b

# 7. Print the results
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")
print(f"Prediction Step 1: {prediction_step_1}")
print(f"Prediction Step 2: {prediction_step_2}")

# 8. Plot actual sequence and forecasts
plot_seq_with_forecast(seq, [prediction_step_1, prediction_step_2])
