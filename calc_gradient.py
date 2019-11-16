import matplotlib
import math
import cv2
import numpy as np


def intoMap(index, cell_nums):
    for i in [0, 1]:
        if index[i] < 0: index[i] = 0
        elif index[i] >= cell_nums[i]: index[i] = cell_nums[i] - 1
    return index

if __name__ == "__main__":
    file_name = 'CorridorGimp_200x200'

    cell_nums = np.array([200, 200])
    state_num =cell_nums[0]*cell_nums[1]
    value_data = np.zeros(np.r_[cell_nums])
    values = np.loadtxt('value/' + file_name + '.value')
    for line in open('value/' + file_name + '.value', 'r'):
        d = line.split()
        value_data[int(d[0]), int(d[1])] = float(d[2])

    with open('gradient/' + file_name + '.grad', 'w') as f:
        for i, value in enumerate(values):
            index = value[0:2]
            index_up = intoMap(index + np.array([1, 1]), cell_nums)
            index_low = intoMap(index + np.array([-1, -1]), cell_nums)

            x_grad = value_data[int(index_up[0]), int(index[1])] - value_data[int(index_low[0]), int(index[1])]
            y_grad = value_data[int(index[0]), int(index_up[1])] - value_data[int(index[0]), int(index_low[1])]

            gradient = np.array([value[0], value[1], x_grad, y_grad])
            f.write("{} {} {} {}\n".format(int(gradient[0]), int(gradient[1]), gradient[2], gradient[3]))
