import numpy as np
import matplotlib.pyplot as plt

# 1. Training data
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)

# 2. Build the design matrix
X = np.column_stack([x, np.ones_like(x)])

# 3. Solve the normal equation
theta = np.linalg.inv(X.T @ X) @ X.T @ y

# 4. Extract the parameters
m, b = theta

# 5. Make predictions
predictions = X @ theta

# 6. Print the results
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")
print(f"Predictions: {predictions}")

# 7. Plot the results
plt.scatter(x, y, label="Observed data")
plt.plot(x, predictions, color="red", label=f"Fitted line: y = {m:.2f}x + {b:.2f}")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression using Normal Equation")
plt.legend()
plt.show()