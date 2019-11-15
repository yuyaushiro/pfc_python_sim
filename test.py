import numpy as np


a = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
b = np.array([0.2, 0.3, 0.4, 0.5])
c = np.dot(1/(b**2), a)
print(c)
