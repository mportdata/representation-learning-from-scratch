import numpy as np
import matplotlib.pyplot as plt

from course_utils.loss_function import mse
from course_utils.plots import loss_landscape, plot_linear_regression


# 1. Initialize observed data
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)

# 2. Possible values for slope (m) and intercept (b)
m_values = np.linspace(-5, 5, 100)
b_values = np.linspace(-5, 5, 100)

# 3. Visualize the loss landscape
fig, ax = loss_landscape(x, y, m_values, b_values, mse)

# 4. Build the design matrix
X = np.column_stack([x, np.ones_like(x)])

# 5. Solve the normal equation
theta = np.linalg.inv(X.T @ X) @ X.T @ y

# 6. Extract the parameters
m, b = theta

# 7. Mark the solution point on the loss landscape
ax.scatter(
    m, 
    b, 
    mse(y, m * x + b), 
    s=200,
    marker="x",
    c="red",
    label="Optimal solution"
)
ax.legend()

plt.show()

# 8. Make predictions
predictions = X @ theta

# 9. Print the results
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")
print(f"Predictions: {predictions}")

# 10. Plot the results
plot_linear_regression(x, y, predictions, m, b)