import numpy as np
import matplotlib.pyplot as plt

from course_utils.loss_function import mse
from course_utils.plots import loss_landscape, mark_gradient_descent_path, plot_linear_regression

from observed_data import ideal_seq, noisy_seq


# 1. Select the sequence to use for regression
#seq = ideal_seq  # [1, 2, 3, 4, 5]
seq = noisy_seq  # [0, 3, 2, 5, 1]

# 2. Create input (x) and target (y) pairs where each x is mapped to the next value in the sequence
X_lagged = np.column_stack([
    seq[:-2],  # t-2
    seq[1:-1], # t-1
])

y = seq[2:]    # t

# 3. Set parameters for gradient descent
#learning_rate = 0.1
learning_rate = 0.01
#epochs = 100
#epochs = 1000
epochs = 10000

# 4. Initial values for coefficients (a, b) and intercept (c)
a = 0  # Coefficient for t-2
b = 0  # Coefficient for t-1
c = 0  # Intercept

# 8. Perform gradient descent
n = len(X_lagged)

for epoch in range(epochs):
    # Compute prediction and loss
    y_pred = a * X_lagged[:, 0] + b * X_lagged[:, 1] + c
    loss = mse(y, y_pred)

    # Compute the gradients of the loss (MSE) with respect to a, b, and c
    a_gradient = (-2 / n) * np.sum(X_lagged[:, 0] * (y - y_pred))
    b_gradient = (-2 / n) * np.sum(X_lagged[:, 1] * (y - y_pred))
    c_gradient = (-2 / n) * np.sum(y - y_pred)

    # Update a, b, and c using the gradients
    a -= learning_rate * a_gradient
    b -= learning_rate * b_gradient
    c -= learning_rate * c_gradient

# 9. Use the final values of a, b, and c to compute the predictions and final loss
y_pred_final = a * X_lagged[:, 0] + b * X_lagged[:, 1] + c
final_loss = mse(y, y_pred_final)

# 10. Print the final values of a, b, and c
print(f"Final coefficient (a): {a}")
print(f"Final coefficient (b): {b}")
print(f"Final intercept (c): {c}")
print(f"Final loss: {final_loss}")
print(f"Predictions: {y_pred_final}")