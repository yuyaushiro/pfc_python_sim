from modules.mcl import Particle, Mcl
from modules.grid_map import GridMap

import math
import numpy as np


class GradientPfc:
    def __init__(self, time_interval, max_nu, max_omega, turn_only_thresh,
                 estimator, grid_map, goal, magnitude=2):
        self.time_interval = time_interval

        # 最大速度設定
        self.max_nu = max_nu
        self.max_omega = max_omega
        # ロボットが旋回のみ行う方向
        self.turn_only_thresh = turn_only_thresh
        # 推定用に 1 ステップ前の速度を保存
        self.prev_nu = 0.0
        self.prev_omega = 0.0

        # 推定器
        self.estimator = estimator

        # マップ設定
        self.grid_map = grid_map
        # ゴール設定
        self.goal = goal

        # 積極度
        self.magnitude = magnitude

        # パーティクルの勾配リスト
        self.p_gradient = np.zeros((len(self.estimator.particles), 2))
        self.p_value = np.zeros(len(self.estimator.particles))

    def make_decision(self, pose, observation):
        # 状態を推定
        self.estimate_state(self.estimator, observation)

        # パーティクルの価値を取得
        self.p_value = np.array([self.grid_map.value(p.pose)
                                 for p in self.estimator.particles])
        # パーティクルの勾配を計算
        self.p_gradient = np.array([self.grid_map.calc_gradient(p.pose)
                                    for p in self.estimator.particles])
        # 各パーティクルの向かいたい方向
        self.p_relative_gradient =\
            np.array([self.rotate_vector(self.p_gradient[i], -p.pose[2])
                      for i, p in enumerate(self.estimator.particles)])
        # Q_gradient
        gradient = np.dot(1/abs(self.p_value**self.magnitude), self.p_relative_gradient)

        direction = math.atan2(gradient[1], gradient[0])
        nu, omega = self.direction_to_vel(direction)

        self.prev_nu, self.prev_omega = nu, omega
        return nu, omega

    # ロボットの状態を推定する
    def estimate_state(self, estimator, observation):
        estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        estimator.observation_update(observation)
        # ゴールに入ったパーティクルを削除
        for p in self.estimator.particles:
            if self.goal.inside(p.pose): p.weight *= 1e-10
        self.estimator.resampling()

    # ベクトルを回転する
    def rotate_vector(self, vector, theta):
        c_r, s_r = np.cos(theta), np.sin(theta)
        rotation = np.array([[c_r, -s_r], [s_r, c_r]])
        rotated_vector = np.dot(rotation, vector)
        return rotated_vector

    # 勾配から速度に変換する
    def direction_to_vel(self, direction):
        # 旋回のみを行う角度
        if direction > self.turn_only_thresh:
            return 0.0, self.max_omega
        if direction < -self.turn_only_thresh:
            return 0.0, self.max_omega
        # 速度の旋回比率
        turn_ratio = direction / self.turn_only_thresh
        omega = self.max_omega * turn_ratio
        nu = self.max_nu * (1 - turn_ratio)

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

    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)
