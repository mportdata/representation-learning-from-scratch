import numpy as np
import matplotlib.pyplot as plt

from loss_function import mse
from observations import x, y
from plots import loss_landscape, plot_linear_regression


# 1. Possible values for slope (m) and intercept (b)
m_values = np.linspace(-5, 5, 100)
b_values = np.linspace(-5, 5, 100)

# 2. Visualize the loss landscape
fig, ax = loss_landscape(x, y, m_values, b_values, mse)

# 3. Build the design matrix
X = np.column_stack([x, np.ones_like(x)])

# 4. Solve the normal equation
theta = np.linalg.inv(X.T @ X) @ X.T @ y

# 5. Extract the parameters
m, b = theta

# 6. Mark the solution point on the loss landscape
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

# 7. Make predictions
predictions = X @ theta

# 8. Print the results
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")
print(f"Predictions: {predictions}")

# 9. Plot the results
plot_linear_regression(x, y, predictions, m, b)