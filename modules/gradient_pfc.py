from modules.mcl import Particle, Mcl
from modules.grid_map import GridMap

import math
import numpy as np


class GradientPfc:
    def __init__(self, time_interval, max_nu, max_omega, turn_only_thresh,
                 estimator, grid_map, goal, magnitude=2,
                 draw_direction=False, draw_p_gradient=False):
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

        # 描画用
        self.draw_direction = draw_direction
        self.draw_p_gradient = draw_p_gradient
        self.p_value = np.zeros(len(self.estimator.particles))
        self.p_gradient = np.zeros((len(self.estimator.particles), 2))
        self.p_relative_gradient = np.zeros((len(self.estimator.particles), 2))
        self.direction = 0
        self.true_pose = np.array([0.0, 0.0, 0.0])

    def make_decision(self, pose, observation):
        # 描画用に正しい姿勢を保持
        self.true_pose = pose
        # 状態を推定
        self.estimate_state(self.estimator, observation)

        # パーティクルの勾配を計算
        self.p_gradient = np.array([self.grid_map.gradient(p.pose)
                                    for p in self.estimator.particles])
        # 各パーティクルの向かいたい方向
        self.p_relative_gradient =\
            np.array([self.rotate_vector(self.p_gradient[i], -p.pose[2])
                      for i, p in enumerate(self.estimator.particles)])

        # パーティクルの価値を取得
        p_pos_value = np.array([self.grid_map.value(p.pose)
                                for p in self.estimator.particles])
        p_rot_value = np.array([abs(math.atan2(g[1], g[0])) / self.max_omega
                                for g in self.p_relative_gradient])
        self.p_value = p_pos_value - p_rot_value

        # Q_gradient
        gradient = np.dot(1/abs(self.p_value**self.magnitude),
                          self.p_relative_gradient)

        self.direction = math.atan2(gradient[1], gradient[0])
        nu, omega = self.direction_to_vel(self.direction)

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
            return 0.0, -self.max_omega
        # 速度の旋回比率
        turn_ratio = direction / self.turn_only_thresh
        omega = self.max_omega * turn_ratio
        nu = self.max_nu * (1 - abs(turn_ratio))

        return nu, omega

    # 描画する
    def draw(self, ax, elems):
        # 推定器の描画
        self.estimator.draw(ax, elems)

        # パーティクルの勾配を描画
        if self.draw_p_gradient:
            for i, g in enumerate(self.p_gradient):
                pos = self.estimator.particles[i].pose[0:2]
                posn = pos + g * 0.2
                elems += ax.plot([pos[0], posn[0]], [pos[1], posn[1]], color='green')

        # 向かう方向を描画
        if self.draw_direction:
            pos = self.true_pose[0:2]
            posn = pos + np.array([math.cos(self.direction+self.true_pose[2]),
                                math.sin(self.direction+self.true_pose[2])])
            elems += ax.plot([pos[0], posn[0]], [pos[1], posn[1]], color='red')

