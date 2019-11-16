from modules.world import World, Landmark, Map, Goal
from modules.grid_map import GridMap
from modules.robot import IdealRobot
from modules.sensor import IdealCamera, Camera
from modules.agent import Agent, EstimationAgent, GradientAgent
from modules.gradient_pfc import GradientPfc
from modules.mcl import Particle, Mcl

import math
import numpy as np


if __name__ == '__main__':   ###name_indent
    time_interval = 0.1
    # world = World(10, time_interval, debug=False, recording_file_name='aaa')
    world = World(100, time_interval, debug=False)

    m = Map()

    ### 専有格子地図を追加 ###
    grid_map = GridMap('CorridorGimp_200x200', origin=[-5.0, -5.0])
    # world.append(grid_map)

    ##ゴールの追加##
    goal = Goal(1.75,3.0)  #goalを変数に
    world.append(goal)

    ### ロボットを作る ###
    # 初期位置
    init_pose = np.array([-3, -1, 0])
    # 初期位置推定のばらつき
    init_pose_stds = np.array([0.2, 0.2, 0.01])
    # モーションアップデートのばらつき
    # motion_noise_stds = {"nn":0.19, "no":0.001, "on":0.13, "oo":0.2}
    motion_noise_stds = {"nn":0.05, "no":0.05, "on":0.05, "oo":0.05}
    # 推定器
    estimator = Mcl(m, init_pose, 100, motion_noise_stds=motion_noise_stds,
                    init_pose_stds=init_pose_stds)
    # エージェント
    agent = GradientPfc(time_interval, 0.2, 0.5, np.deg2rad(45), estimator, grid_map, goal)
    # ロボット
    robot = IdealRobot(init_pose, sensor=Camera(m), agent=agent)
    world.append(robot)

    world.draw()
