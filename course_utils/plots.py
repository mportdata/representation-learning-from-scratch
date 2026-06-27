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
    plt.show()

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

def _split_input_variables(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError("y must be a 1D array")

    if x.ndim == 1:
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        return x, None, y

    if x.ndim == 2 and x.shape == (2, len(y)):
        return x[0], x[1], y

    if x.ndim == 2 and x.shape == (len(y), 2):
        return x[:, 0], x[:, 1], y

    raise ValueError("x must be a 1D array, or two input variables shaped (2, n) or (n, 2)")


def _split_x_labels(x_label):
    if isinstance(x_label, (list, tuple)):
        if len(x_label) != 2:
            raise ValueError("x_label must contain two labels when x has two input variables")
        return x_label[0], x_label[1]

    return x_label, "Input 2"


def _parse_surface_values(line_values):
    if len(line_values) == 3:
        surface_x1, surface_x2, surface_y = line_values
    elif len(line_values) == 2:
        surface_x, surface_y = line_values
        surface_x = np.asarray(surface_x)

        if surface_x.ndim >= 2 and surface_x.shape[0] == 2:
            surface_x1, surface_x2 = surface_x[0], surface_x[1]
        elif surface_x.ndim >= 2 and surface_x.shape[-1] == 2:
            surface_x1, surface_x2 = surface_x[..., 0], surface_x[..., 1]
        else:
            raise ValueError("3D line_values must be (x1_grid, x2_grid, y_grid)")
    else:
        raise ValueError("3D line_values must be (x1_grid, x2_grid, y_grid)")

    surface_x1 = np.asarray(surface_x1)
    surface_x2 = np.asarray(surface_x2)
    surface_y = np.asarray(surface_y)

    if surface_x1.shape != surface_x2.shape or surface_x1.shape != surface_y.shape:
        raise ValueError("3D line_values arrays must all have the same shape")

    return surface_x1, surface_x2, surface_y


def _predict_from_surface(prediction_x1, prediction_x2, surface_x1, surface_x2, surface_y):
    features = np.column_stack([
        surface_x1.ravel(),
        surface_x2.ravel(),
        np.ones(surface_y.size)
    ])
    weights, _, _, _ = np.linalg.lstsq(features, surface_y.ravel(), rcond=None)
    return weights[0] * prediction_x1 + weights[1] * prediction_x2 + weights[2]


def plot_scatter_plot(
        x: np.ndarray,
        y: np.ndarray,
        x_label: str = "x",
        y_label: str = "y",
        line_values: np.ndarray = None,
        single_prediction: float = None,
        residuals: np.ndarray = None
    ):
    x_1, x_2, y = _split_input_variables(x, y)

    if x_2 is None:
        plt.scatter(x_1, y, label="Observed Data")
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if line_values is not None:
            line_x = np.asarray(line_values[0])
            line_y = np.asarray(line_values[1])
            if line_x.shape != line_y.shape:
                raise ValueError("line_values must contain x and y arrays with the same shape")

            plt.plot(line_x, line_y, color="red", label="Linear Regression")

        if residuals is not None:
            residuals = np.asarray(residuals)
            if residuals.shape != y.shape:
                raise ValueError("residuals must have the same shape as y")

            predicted_y = y - residuals
            plt.vlines(
                x_1,
                predicted_y,
                y,
                colors="gray",
                linestyles="--",
                label="Residuals"
            )

        if single_prediction is not None:
            if line_values is None:
                raise ValueError("single_prediction requires line_values")

            sort_order = np.argsort(line_x)
            prediction_x = single_prediction
            prediction_y = np.interp(prediction_x, line_x[sort_order], line_y[sort_order])
            x_min, _ = plt.xlim()
            y_min, _ = plt.ylim()

            plt.vlines(
                prediction_x,
                y_min,
                prediction_y,
                colors="blue",
                linestyles=":",
                label="Prediction Guide"
            )
            plt.hlines(
                prediction_y,
                x_min,
                prediction_x,
                colors="blue",
                linestyles=":"
            )
            plt.annotate(
                f"{prediction_y:.2f}",
                xy=((x_min + prediction_x) / 2, prediction_y),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                color="blue"
            )
            plt.scatter(
                prediction_x,
                prediction_y,
                color="blue",
                zorder=3,
                label="Prediction"
            )

        plt.legend()
        return

    x_1_label, x_2_label = _split_x_labels(x_label)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x_1, x_2, y, label="Observed Data")
    ax.set_xlabel(x_1_label, labelpad=8)
    ax.set_ylabel(x_2_label, labelpad=8)
    ax.set_zlabel("")
    ax.text2D(
        0.96,
        0.52,
        y_label,
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center"
    )

    if line_values is not None:
        surface_x1, surface_x2, surface_y = _parse_surface_values(line_values)
        if surface_x1.ndim == 2:
            ax.plot_surface(surface_x1, surface_x2, surface_y, color="red", alpha=0.25)
            ax.plot_wireframe(surface_x1, surface_x2, surface_y, color="red", alpha=0.45)
        else:
            ax.plot_trisurf(
                surface_x1.ravel(),
                surface_x2.ravel(),
                surface_y.ravel(),
                color="red",
                alpha=0.25
            )
        ax.plot([], [], [], color="red", label="Linear Regression")

    if residuals is not None:
        residuals = np.asarray(residuals)
        if residuals.shape != y.shape:
            raise ValueError("residuals must have the same shape as y")

        predicted_y = y - residuals
        for index, (x1_value, x2_value, observed_y, prediction_y) in enumerate(
                zip(x_1, x_2, y, predicted_y)
            ):
            ax.plot(
                [x1_value, x1_value],
                [x2_value, x2_value],
                [prediction_y, observed_y],
                color="gray",
                linestyle="--",
                label="Residuals" if index == 0 else None
            )

    if single_prediction is not None:
        if line_values is None:
            raise ValueError("single_prediction requires line_values")

        prediction_input = np.asarray(single_prediction)
        if prediction_input.shape != (2,):
            raise ValueError("single_prediction must contain two values when x has two input variables")

        prediction_x1, prediction_x2 = prediction_input
        prediction_y = _predict_from_surface(
            prediction_x1,
            prediction_x2,
            surface_x1,
            surface_x2,
            surface_y
        )
        x_min, _ = ax.get_xlim()
        z_min, _ = ax.get_zlim()

        ax.plot(
            [prediction_x1, prediction_x1],
            [prediction_x2, prediction_x2],
            [z_min, prediction_y],
            color="blue",
            linestyle=":",
            label="Prediction Guide"
        )
        ax.plot(
            [x_min, prediction_x1],
            [prediction_x2, prediction_x2],
            [prediction_y, prediction_y],
            color="blue",
            linestyle=":"
        )
        ax.text(
            (x_min + prediction_x1) / 2,
            prediction_x2,
            prediction_y,
            f"{prediction_y:.2f}",
            color="blue"
        )
        ax.scatter(
            [prediction_x1],
            [prediction_x2],
            [prediction_y],
            color="blue",
            label="Prediction"
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
