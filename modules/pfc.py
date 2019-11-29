from modules.mcl import Particle, Mcl
from modules.grid_map import GridMap
from modules.robot import IdealRobot

import math
import numpy as np


class Pfc:
    def __init__(self, time_interval, nu, omega, estimator,
                 grid_map, goal, magnitude=2):
        self.time_interval = time_interval

        # 最大速度設定
        self.nu = nu
        self.omega = omega

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

        # 行動
        self.actions = np.array([[0.0, 0.5], [0.1, 0.0], [0.0, -0.5]])

        self.prev_Q_fpc = 0.0

    def make_decision(self, pose, observation):
        # 描画用に正しい姿勢を保持
        self.true_pose = pose
        # 状態を推定
        self.estimate_state(self.estimator, observation)

        Q_PFC_list = np.array([self.calc_Q_pfc(a[0], a[1], pose) for a in self.actions])
        if np.max(Q_PFC_list) < self.prev_Q_fpc:
            self.prev_Q_fpc = np.max(Q_PFC_list)
            action = self.actions[1]
        else:
            action = self.actions[np.argmax(Q_PFC_list)]

        # action = self.grid_map.policy(pose)

        nu, omega = action
        self.prev_nu, self.prev_omega = nu, omega
        return nu, omega

    def estimate_state(self, estimator, observation):
        estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        estimator.observation_update(observation)
        # ゴールに入ったパーティクルを削除
        for p in self.estimator.particles:
            if self.goal.inside(p.pose): p.weight *= 1e-10
        self.estimator.resampling()

    def calc_Q_pfc(self, nu, omega, pose):
        poses = np.array([p.pose for p in self.estimator.particles])
        values = np.array([self.grid_map.value(p) for p in poses])
        poses_n = np.array([IdealRobot.transition_state(nu, omega, self.time_interval, p)
                           for p in poses])
        rewards = np.array([self.time_interval + self.grid_map.in_obstacle(pn)*10
                            for pn in poses_n])
        values_n = np.array([self.grid_map.value(pn) for pn in poses_n])
        Q_pfc = np.sum((values_n - rewards)/np.power(values, self.magnitude))
        return Q_pfc
        # pose_n = IdealRobot.transition_state(nu, omega, self.time_interval, pose)
        # reward = self.time_interval + self.grid_map.in_obstacle(pose_n)*10
        # value_n = self.grid_map.value(pose_n) - reward
        # return value_n

    def draw(self, ax, elems):
        # 推定器の描画
        self.estimator.draw(ax, elems)
