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

def mark_solution_point(ax, m, b, loss):
    ax.scatter(
        m, 
        b, 
        loss, 
        color="red", 
        s=100, 
        label=f"Solution: m={m:.2f}, b={b:.2f}, loss={loss:.2f}"
    )
    ax.legend()

def mark_gradient_descent_path(ax, m_history, b_history, loss_history):
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

def plot_linear_regression(x, y, predictions, m, b):
    plt.scatter(x, y, label="Observed data")
    plt.plot(x, predictions, color="red", label=f"Fitted line: y = {m:.2f}x + {b:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression using Normal Equation")
    plt.legend()
    plt.show()

def plot_seq_with_forecast(seq, forecast_sequence):
    forecast_sequence = [seq[-1]] + forecast_sequence  # Include the last observed value for continuity
    plt.figure(figsize=(8, 5))

    # Plot the observed sequence
    plt.plot(
        range(len(seq)),
        seq,
        marker="o",
        label="Observed Sequence"
    )

    # Plot the forecasted sequence
    plt.plot(
        range(len(seq) - 1, len(seq) + 2),
        forecast_sequence,
        marker="o",
        linestyle="--",
        label="Forecast"
    )

    # Add labels, title, and legend
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("Forecast")
    plt.legend()
    plt.grid(True)

    plt.show()