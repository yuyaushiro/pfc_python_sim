import math
import cv2
import numpy as np


# 格子地図クラス
class GridMap:
    def __init__(self, map_name, resolution=[0.05, 0.05], origin=[0.0, 0.0]):
        # マップ名
        self.map_name = map_name
        # マップ画像
        self.map_image = cv2.imread('map/' + map_name + '.png', cv2.IMREAD_GRAYSCALE)
        # マップデータ(座標系合わせ後)
        self.map_data = self.map_image.T[:, ::-1]
        # 解像度 (m/pixel)
        self.resolution = np.array(resolution)
        # セルの数
        self.cell_num = np.array(self.map_image.shape)
        # マップ内の最小の姿勢
        self.pose_min = np.array(origin)
        # マップ内の最大の姿勢
        self.pose_max = self.cell_num*self.resolution + self.pose_min

        # 価値関数の初期化
        self.value_data = self.init_value(self.cell_num, self.map_name)

    # 勾配を計算する
    def calc_gradient(self, pose):
        x_up = pose[0] + self.resolution[0]
        x_low = pose[0] - self.resolution[0]
        y_up = pose[1] + self.resolution[1]
        y_low = pose[1] - self.resolution[1]

        x_grad = self.value([x_up, pose[1]]) - self.value([x_low, pose[1]])
        y_grad = self.value([pose[0], y_up]) - self.value([pose[0], y_low])

        return x_grad, y_grad

    # 価値関数を呼び出す
    def value(self, pose):
        position = pose[0:2]
        index = self.to_index(position)
        value = self.value_data[index]

        return value

    # 姿勢を整数のインデックスに変換
    def to_index(self, position):
        index = np.floor((position - self.pose_min)/self.resolution).astype(int)
        for i in [0, 1]:
            if index[i] < 0: index[i] = 0
            elif index[i] >= self.cell_num[i]: index[i] = self.cell_num[i] - 1

        return tuple(index)

    # 価値関数の初期化
    def init_value(self, cell_num, map_name):
        value_data = np.zeros(np.r_[cell_num])
        for line in open('value/' + map_name + '.value', 'r'):
            d = line.split()
            value_data[int(d[0]), int(d[1])] = float(d[2])

        return value_data
