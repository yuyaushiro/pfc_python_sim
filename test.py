import numpy as np
import datetime


a = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
b = np.array([0.2, 0.3, 0.4, 0.5])
c = np.dot(1/(b**2), a)
# print(c)

aaa = None
if aaa:
    # print('not None')
    pass
else:
    # print('None')
    pass

x_rand = np.random.normal(5, 0.1, 5)
y_rand = np.random.normal(10, 0.1, 5)

rand = np.vstack((x_rand, y_rand)).T
# print(rand)

now = datetime.datetime.now()
print(now.strftime('_%Y%m%d_%H%M%S'))
