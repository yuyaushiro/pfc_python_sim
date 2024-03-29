from modules.world import World, Landmark, Map, Goal
from modules.grid_map_2d import GridMap2D
from modules.robot import IdealRobot
from modules.sensor import IdealCamera, Camera
from modules.agent import Agent, EstimationAgent, GradientAgent
from modules.gradient_pfc import GradientPfc
from modules.mcl import Particle, Mcl

import math
import numpy as np


if __name__ == '__main__':   ###name_indent
    time_interval = 0.1
    world = World(200, time_interval, debug=False, recording_file_name='std(0.3_0.3)_回避行動0秒-1秒', playback_speed=3)
    # world = World(150, time_interval, debug=False)

    m = Map()

    ### 専有格子地図を追加 ###
    grid_map = GridMap2D('CorridorGimp_200x200', origin=[-5.0, -5.0])
    # world.append(grid_map)

    ##ゴールの追加##
    goal = Goal(1.75,3.0)  #goalを変数に
    world.append(goal)

    ### ロボットを作る ###
    # 初期位置
    init_pose = np.array([-4.5, 0.5, 0])
    # 初期位置推定のばらつき
    init_pose_stds = np.array([0.3, 0.3, 0.01])
    # モーションアップデートのばらつき
    # motion_noise_stds = {"nn":0.19, "no":0.001, "on":0.13, "oo":0.2}
    motion_noise_stds = {"nn":0.01, "no":0.01, "on":0.01, "oo":0.01}
    # 推定器
    estimator = Mcl(m, init_pose, 300, motion_noise_stds=motion_noise_stds,
                    init_pose_stds=init_pose_stds)
    # エージェント
    agent = GradientPfc(time_interval, 0.1, 0.5, np.deg2rad(90), estimator, grid_map, goal,
                        magnitude=2, draw_direction=False, draw_p_gradient=False)
    # ロボット
    robot = IdealRobot(init_pose, sensor=Camera(m), agent=agent)
    world.append(robot)

    world.draw()
