import numpy as np

from observed_data import ideal_seq, noisy_seq


# 1. Select the sequence to use for regression
#seq = ideal_seq  # [1, 2, 3, 4, 5]
seq = noisy_seq  # [0, 3, 2, 5, 1]

# 2. Create input (X_lagged) and target (y) pairs where each y is mapped to from the previous two values in the sequence

X_lagged = np.column_stack([
   seq[:-2],  # t-2
   seq[1:-1], # t-1
])
y = seq[2:]    # t

# 3. Print the input and target pairs
print("Original Sequence:")
print([int(x) for x in seq])
print("Input (t-2, t-1) and Target (t) pairs:")
for i in range(len(y)):
    print(f"{[int(x) for x in X_lagged[i]]} --> {int(y[i])}")