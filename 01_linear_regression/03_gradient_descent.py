import numpy as np
import matplotlib.pyplot as plt

from loss_function import mse
from observations import x, y
from plots import loss_landscape, plot_linear_regression


# 1. Possible values for slope (m) and intercept (b)
m_values = np.linspace(-5, 5, 100)
b_values = np.linspace(-5, 5, 100)

# 2.Visualize the loss landscape
fig, ax = loss_landscape(x, y, m_values, b_values, mse)

# 3. Set parameters for gradient descent
#learning_rate = 0.1
learning_rate = 0.01
#epochs = 100
#epochs = 1000
epochs = 10000

# 4. Initial values for slope (m) and intercept (b)
m = -4.0
b = -4.0

# 5. Store the history of m, b, and loss for visualization
m_history = []
b_history = []
loss_history = []

# 10. Perform gradient descent
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

# 6. Use the final values of m and b to compute the predictions and final loss
y_pred_final = m * x + b
final_loss = mse(y, y_pred_final)

# 7. Print the final values of m, b, and loss
print(f"Final slope (m): {m}")
print(f"Final intercept (b): {b}")
print(f"Final loss: {final_loss}")
print(f"Predictions: {y_pred_final}")

# 8. Mark the gradient descent path on the loss landscape
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

# 9. Plot the results
plot_linear_regression(x, y, y_pred_final, m, b)