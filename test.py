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
# print(now.strftime('_%Y%m%d_%H%M%S'))

# 回転行列チェック
robot = np.array([0, 0, np.deg2rad(60)])
grad_p = np.array([1, 2])
c_r, s_r = np.cos(-robot[2]), np.sin(-robot[2])
rot = np.array([[c_r, -s_r], [s_r, c_r]])
grad = np.dot(rot, grad_p)
phi = np.arctan2(grad[1], grad[0])
# print(grad)
# print(np.rad2deg(phi))

a = np.array([0, 1, 2, 3, 4, 5])
b = np.array([0, 1, 2, 3, 4, 5])
print(np.dot(a, b))
print(a * b)
