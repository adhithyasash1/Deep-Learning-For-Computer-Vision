import numpy as np

# Define the second moment matrix M
M = np.array([[1, 2], 
              [2, 1]])

# Calculate the eigenvalues of the matrix M
eigenvalues = np.linalg.eigvals(M)

# The measure of straightness is the smallest eigenvalue divided by the largest eigenvalue
straightness_measure = min(eigenvalues) / max(eigenvalues)

straightness_measure
