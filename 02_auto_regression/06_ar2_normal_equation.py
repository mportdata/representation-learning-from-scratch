import numpy as np

from observed_data import ideal_seq, noisy_seq


# 1. Select the sequence to use for regression
#seq = ideal_seq  # [1, 2, 3, 4, 5]
seq = noisy_seq  # [0, 3, 2, 5, 1]

# 2. Create input (x) and target (y) pairs where each x is mapped to the next value in the sequence
X_lagged = np.column_stack([
    seq[:-2],  # t-2
    seq[1:-1], # t-1
])

y = seq[2:]    # t

# 3. Build the design matrix
X = np.column_stack([X_lagged, np.ones(len(X_lagged))])

# 4. Solve the normal equation
theta = np.linalg.inv(X.T @ X) @ X.T @ y

# 5. Extract the parameters
a, b, c = theta

# 6. Make predictions
predictions = X @ theta

# 7. Print the results
print(f"Coefficient for t-2 (a): {a}")
print(f"Coefficient for t-1 (b): {b}")
print(f"Intercept (c): {c}")