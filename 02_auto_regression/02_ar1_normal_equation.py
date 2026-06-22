import numpy as np
import matplotlib.pyplot as plt

from course_utils.loss_function import mse
from course_utils.plots import loss_landscape, mark_solution_point, plot_linear_regression

from observed_data import ideal_seq, noisy_seq


# 1. Select the sequence to use for regression
#seq = ideal_seq  # [1, 2, 3, 4, 5]
seq = noisy_seq  # [0, 3, 2, 5, 1]

# 2. Create input (x) and target (y) pairs where each x is mapped to the next value in the sequence
x = seq[:-1]
y = seq[1:]

# 3. Possible values for slope (m) and intercept (b)
m_values = np.linspace(-5, 5, 100)
b_values = np.linspace(-5, 5, 100)

# 4. Visualize the loss landscape
fig, ax = loss_landscape(x, y, m_values, b_values, mse)

# 5. Build the design matrix
X = np.column_stack([x, np.ones_like(x)])

# 6. Solve the normal equation
theta = np.linalg.inv(X.T @ X) @ X.T @ y

# 7. Extract the parameters
m, b = theta

# 8. Mark the solution point on the loss landscape
mark_solution_point(ax, m, b, mse(y, m * x + b))
plt.show()

# 9. Make predictions
predictions = X @ theta

# 10. Print the results
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")
print(f"Predictions: {predictions}")

# 11. Plot the results
plot_linear_regression(x, y, predictions, m, b)