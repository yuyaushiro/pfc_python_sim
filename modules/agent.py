from modules.mcl import Particle, Mcl
from modules.grid_map import GridMap

import math
import numpy as np


class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def make_decision(self, observation=None):
        return self.nu, self.omega


class EstimationAgent:
    def __init__(self, time_interval, nu, omega, estimator):
        self.nu = nu
        self.omega = omega

        self.estimator = estimator
        self.time_interval = time_interval

        self.prev_nu = 0.0
        self.prev_omega = 0.0

    def make_decision(self, observation=None):
        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.prev_nu, self.prev_omega = self.nu, self.omega
        self.estimator.observation_update(observation)
        return self.nu, self.omega

    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)
        x, y, t = self.estimator.pose
        s = "({:.2f}, {:.2f}, {})".format(x,y,int(t*180/math.pi)%360)
        elems.append(ax.text(x, y+0.1, s, fontsize=8))


# 2次元の価値関数の勾配をもとに行動するエージェントクラス
class GradientAgent:
    def __init__(self, time_interval, max_nu, max_omega, grid_map, goal):
        self.time_interval = time_interval

        # 最大速度設定
        self.max_nu = max_nu
        self.max_omega = max_omega

        # マップ設定
        self.grid_map = grid_map

        # ゴール設定
        self.goal = goal
        self.in_goal = False

    # 行動決定する
    def make_decision(self, pose, observation):
        if self.in_goal:
            return 0.0, 0.0

        x_grad, y_grad = self.grid_map.calc_gradient(pose)
        direction = math.atan2(y_grad, x_grad)
        nu, omega = self.direction_to_vel(pose, direction)

        return nu, omega

    # 勾配から速度に変換する
    def direction_to_vel(self, pose, direction):
        rotation = pose[2]
        head_direction = self.angle_difference(direction, rotation)
        # 旋回のみを行う角度
        spin_turn_thresh = 45 * math.pi/180
        if head_direction > spin_turn_thresh:
            return 0.0, 0.4
        if head_direction < -spin_turn_thresh:
            return 0.0, -0.4
        # 速度の旋回比率
        turn_ratio = head_direction / spin_turn_thresh
        omega = 0.4 * turn_ratio
        nu = 0.2 * (spin_turn_thresh- turn_ratio)

        return nu, omega

    # angle1 - angle2 を計算
    def angle_difference(self, angle1, angle2):
        angle1 = self.normalize_angle(angle1)
        angle2 = self.normalize_angle(angle2)
        angle_diff = angle1 - angle2
        if angle_diff > math.pi: angle_diff -= 2*math.pi
        if angle_diff < -math.pi: angle_diff += 2*math.pi

        return angle_diff

    # 角度の正規化 [0 ~ 360)
    def normalize_angle(self, rotation):
        while rotation < 0: rotation += 2*math.pi
        while rotation >= 2*math.pi: rotation -= 2*math.pi

        return rotation
