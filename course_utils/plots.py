import matplotlib.pyplot as plt
import numpy as np


def loss_landscape(
        x_observed, 
        y_observed, 
        possible_m_values, 
        possible_b_values,
        loss_function
    ):

    # Initialize a 2D array to store the loss values
    losses = np.zeros((100, 100))

    # Compute the loss for each combination of m and b
    for i, m in enumerate(possible_m_values):
        for j, b in enumerate(possible_b_values):

            y_pred = m * x_observed + b

            loss = loss_function(y_observed, y_pred)

            losses[i, j] = loss

    # Create a visualization of the loss landscape
    M, B = np.meshgrid(possible_m_values, possible_b_values)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_wireframe(
        M,
        B,
        losses.T,
        rstride=5,
        cstride=5
    )

    ax.set_xlabel("Gradient (m)")
    ax.set_ylabel("Intercept (b)")
    ax.set_zlabel("Loss")

    return fig, ax

def plot_linear_regression(x, y, predictions, m, b):
    plt.scatter(x, y, label="Observed data")
    plt.plot(x, predictions, color="red", label=f"Fitted line: y = {m:.2f}x + {b:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression using Normal Equation")
    plt.legend()
    plt.show()