import numpy as np
import matplotlib.pyplot as plt

from course_utils.loss_function import mse
from course_utils.plots import loss_landscape


# 1. Initialize the sequence
seq = np.array([1, 2, 3, 4, 5], dtype=np.float32)

# 2. Create input (x) and target (y) pairs where each x is mapped to the next value in the sequence
x = seq[:-1]  # [1, 2, 3, 4]
y = seq[1:]   # [2, 3, 4, 5]

# 3. Possible values for slope (m) and intercept (b)
m_values = np.linspace(-5, 5, 100)
b_values = np.linspace(-5, 5, 100)

# 4. Visualize the loss landscape
fig, ax = loss_landscape(x, y, m_values, b_values, mse)

plt.show()