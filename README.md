# Representation Learning From Scratch

This repository is a set of worked lessons for building up machine learning ideas from basic numerical examples. The current notebooks focus on linear regression: plotting observations, making guessed models, measuring residuals and loss, fitting parameters with the normal equation, and using gradient descent.

The emphasis is on seeing each idea directly in NumPy and Matplotlib before introducing higher-level modelling tools.

## Notebook Contents

The material currently converted into notebooks lives in `01_linear_regression`.

| Notebook | Topic | Contents |
| --- | --- | --- |
| [`01_linear_regression.ipynb`](01_linear_regression/01_linear_regression.ipynb) | Linear regression basics | Single-input linear regression, 2D scatter plots, guessed model lines, single predictions, multiple-input linear regression, and 3D plots for height/age/weight examples. |
| [`02_residuals_and_loss.ipynb`](01_linear_regression/02_residuals_and_loss.ipynb) | Residuals and loss | Residuals as observed minus predicted values, visual residual lines, why summing residuals is misleading, and mean squared error as a more useful loss function. |
| [`03_minimising_loss.ipynb`](01_linear_regression/03_minimising_loss.ipynb) | Minimising loss | Loss landscapes, matrix form for linear regression, the normal equation, marking the fitted solution on the loss landscape, and fitting both single-input and multiple-input regression models. |
| [`04_gradient_descent.ipynb`](01_linear_regression/04_gradient_descent.ipynb) | Gradient descent | Iterative optimisation for linear regression, learning rate and epoch hyperparameters, deriving the MSE gradient, recording parameter history, and marking the gradient descent path on the loss landscape. |

## Supporting Code

Shared plotting utilities are in [`course_utils/plots.py`](course_utils/plots.py). These include:

- 2D and 3D scatter plotting for regression examples.
- Regression line and plane plotting.
- Residual visualisation.
- Single-prediction guide lines.
- Loss landscape plotting, solution markers, and gradient descent path markers.

The `02_auto_regression` directory currently contains script-based experiments for AR(1) and AR(2) sequence modelling. Those examples cover turning sequences into regression problems, fitting with the normal equation, using gradient descent, and simple forecasting.

## Running the Notebooks

This project uses Python `>=3.13` and keeps dependencies in `pyproject.toml`.

With `uv`:

```bash
uv sync
uv run jupyter notebook
```

Then open the notebooks under `01_linear_regression`.

## Repository Layout

```text
course_utils/              Shared plotting and utility code
01_linear_regression/      Linear regression notebooks and related scripts
02_auto_regression/        Autoregression scripts
pyproject.toml             Project metadata and dependencies
```

## Current Status

The first linear regression lessons have been moved into notebooks. The rest of the repository is still mostly script-based, with the same from-scratch style intended for future notebook conversion.
