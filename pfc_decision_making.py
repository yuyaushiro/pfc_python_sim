from modules.world import World, Landmark, Map, Goal
from modules.grid_map import GridMap
from modules.robot import IdealRobot
from modules.sensor import IdealCamera, Camera
from modules.agent import Agent, EstimationAgent, GradientAgent
from modules.pfc import Pfc
from modules.mcl import Particle, Mcl

import math
import numpy as np


if __name__ == '__main__':   ###name_indent
    time_interval = 0.1
    # world = World(200, time_interval, debug=False, recording_file_name='std(0.3_0.3)_回避行動0秒-1秒', playback_speed=3)
    world = World(150, time_interval, debug=False, drawing_range=[-2.5, 2.5])

    m = Map()

    ### 専有格子地図を追加 ###
    grid_map = GridMap('CorridorGimp_100x100', origin=[-2.5, -2.5, 0])
    # world.append(grid_map)

    ##ゴールの追加##
    goal = Goal(0.8, 1.5)
    world.append(goal)

    ### ロボットを作る ###
    # 初期位置
    init_pose = np.array([-1.0, -1.0, 0])
    # 初期位置推定のばらつき
    init_pose_stds = np.array([0.2, 0.2, 0.02])
    # モーションアップデートのばらつき
    # motion_noise_stds = {"nn":0.19, "no":0.001, "on":0.13, "oo":0.2}
    motion_noise_stds = {"nn":0.03, "no":0.03, "on":0.03, "oo":0.03}
    # 推定器
    estimator = Mcl(m, init_pose, 100, motion_noise_stds=motion_noise_stds,#)
                    init_pose_stds=init_pose_stds)

    # エージェント
    agent = Pfc(time_interval, 0.1, 0.5, estimator, grid_map, goal, magnitude=2)

    # ロボット
    robot = IdealRobot(init_pose, sensor=Camera(m), agent=agent)
    world.append(robot)

    world.draw()
