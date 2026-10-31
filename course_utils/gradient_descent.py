import numpy as np
from course_utils.loss_function import mse

def perform_gradient_descent(
    X_observed: np.ndarray,
    y_observed: np.ndarray,
    initial_parameters: np.ndarray,
    learning_rate: float,
    epochs: int,
) -> tuple[np.ndarray, float, list[np.ndarray], list[float]]:
    parameter_history = []
    loss_history = []

    X_observed = np.asarray(X_observed, dtype=float)
    if X_observed.ndim == 1:
        X_observed = X_observed.reshape(-1, 1)

    y_observed = np.asarray(y_observed, dtype=float).reshape(-1)

    # Create design matrix by adding intercept column
    X = np.column_stack([
        X_observed,
        np.ones(len(X_observed)),
    ])

    current_parameters = np.asarray(initial_parameters, dtype=float).copy()
    n = len(X)

    for epoch in range(epochs):
        # Predict y values using current parameters
        y_pred = X @ current_parameters

        # Compute loss using the notebook's MSE function
        loss = mse(y_observed, y_pred)

        parameter_history.append(current_parameters.copy())
        loss_history.append(loss)

        # Compute the gradients of MSE with respect to each parameter
        gradients = (-2 / n) * X.T @ (y_observed - y_pred)

        # Update the parameters using the gradients (we subtract so that the direction is the opposite to that of the gradient)
        current_parameters -= learning_rate * gradients

    # Set the final parameters and final loss value
    final_parameters = current_parameters
    final_loss = mse(y_observed, X @ final_parameters)

    return (final_parameters, final_loss, parameter_history, loss_history)