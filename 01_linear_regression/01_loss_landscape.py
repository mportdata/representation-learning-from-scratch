import numpy as np
import matplotlib.pyplot as plt

from loss_function import mse
from observations import x, y
from plots import loss_landscape


# 1. Possible values for slope (m) and intercept (b)
m_values = np.linspace(-5, 5, 100)
b_values = np.linspace(-5, 5, 100)

# 2. Visualize the loss landscape
fig, ax = loss_landscape(x, y, m_values, b_values, mse)

plt.show()