import numpy as np

# demonstrates perfect fit of a linear model to a sequence of numbers
ideal_seq = np.array([1, 2, 3, 4, 5], dtype=np.float32)

# demonstrates residual error in a linear model fit to a sequence of numbers
noisy_seq = np.array([0, 3, 2, 5, 1], dtype=np.float32)