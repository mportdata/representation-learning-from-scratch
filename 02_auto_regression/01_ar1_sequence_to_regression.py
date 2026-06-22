import numpy as np
import matplotlib.pyplot as plt

from course_utils.loss_function import mse
from course_utils.plots import loss_landscape

from observed_data import ideal_seq, noisy_seq


# 1. Create input (x) and target (y) pairs where each x is mapped to the next value in the sequence
x = ideal_seq[:-1]  # [1, 2, 3, 4]
y = ideal_seq[1:]   # [2, 3, 4, 5]

#x = noisy_seq[:-1]  # [0, 3, 2, 5]
#y = noisy_seq[1:]   # [3, 2, 5, 1]

# 2. Possible values for slope (m) and intercept (b)
m_values = np.linspace(-5, 5, 100)
b_values = np.linspace(-5, 5, 100)

# 3. Visualize the loss landscape
fig, ax = loss_landscape(x, y, m_values, b_values, mse)

plt.show()