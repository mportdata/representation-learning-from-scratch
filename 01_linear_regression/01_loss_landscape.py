import numpy as np
import matplotlib.pyplot as plt

from course_utils.loss_function import mse
from course_utils.plots import loss_landscape

from observed_data import x, y


# 1. Possible values for slope (m) and intercept (b)
m_values = np.linspace(-5, 5, 100)
b_values = np.linspace(-5, 5, 100)

# 2. Visualize the loss landscape
fig, ax = loss_landscape(x, y, m_values, b_values, mse)

plt.show()