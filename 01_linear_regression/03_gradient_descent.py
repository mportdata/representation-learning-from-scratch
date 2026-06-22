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

# 3.Visualize the loss landscape
fig, ax = loss_landscape(x, y, m_values, b_values, mse)

# 4. Set parameters for gradient descent
#learning_rate = 0.1
learning_rate = 0.01
#epochs = 100
#epochs = 1000
epochs = 10000

# 5. Initial values for slope (m) and intercept (b)
m = -4.0
b = -4.0

# 6. Store the history of m, b, and loss for visualization
m_history = []
b_history = []
loss_history = []

# 7. Perform gradient descent
n = len(x)

for epoch in range(epochs):
    # Compute prediction and loss
    y_pred = m * x + b
    loss = mse(y, y_pred)

    # Add the current values to history
    m_history.append(m)
    b_history.append(b)
    loss_history.append(loss)

    # Compute the gradients of the loss (MSE) with respect to m and b
    m_gradient = (-2 / n) * np.sum(x * (y - y_pred))
    b_gradient = (-2 / n) * np.sum(y - y_pred)

    # Update m and b using the gradients
    m -= learning_rate * m_gradient
    b -= learning_rate * b_gradient

# 8. Use the final values of m and b to compute the predictions and final loss
y_pred_final = m * x + b
final_loss = mse(y, y_pred_final)

# 9. Print the final values of m, b, and loss
print(f"Final slope (m): {m}")
print(f"Final intercept (b): {b}")
print(f"Final loss: {final_loss}")
print(f"Predictions: {y_pred_final}")

# 10. Mark the gradient descent path on the loss landscape
ax.plot(
    m_history,
    b_history,
    loss_history,
    marker="o",
    color="black",
    label="Gradient descent path"
)

ax.scatter(
    m_history[0],
    b_history[0],
    loss_history[0],
    s=100,
    color="black",
    label="Start"
)

ax.scatter(
    m_history[-1],
    b_history[-1],
    loss_history[-1],
    s=100,
    marker="x",
    color="red",
    label="End"
)

ax.legend()

plt.show()

# 11. Plot the results
plot_linear_regression(x, y, y_pred_final, m, b)