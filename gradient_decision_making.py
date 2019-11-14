from modules.world import World, Landmark, Map, Goal
from modules.robot import IdealRobot
from modules.sensor import IdealCamera, Camera
from modules.agent import Agent, EstimationAgent, GradientAgent
from modules.mcl import Particle, Mcl

import math
import numpy as np


if __name__ == '__main__':   ###name_indent
    time_interval = 0.1
    world = World(300, time_interval, debug=False)

    ### 地図を生成して3つランドマークを追加 ###
    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]: m.append_landmark(Landmark(*ln))
    world.append(m)

    ##ゴールの追加##
    goal = Goal(1.75,3.0)  #goalを変数に
    world.append(goal)

    ### ロボットを作る ###
    initial_pose = np.array([-3, -1, 0]).T
    # motion_noise_stds = {"nn":0.19, "no":0.001, "on":0.13, "oo":0.2}
    motion_noise_stds = {"nn":0.05, "no":0.05, "on":0.05, "oo":0.05}
    estimator = Mcl(m, initial_pose, 100, motion_noise_stds=motion_noise_stds)
    agent = GradientAgent(time_interval, 0.2, 0.5, 'CorridorGimp_200x200', [0.05, 0.05], goal)
    r = IdealRobot(initial_pose, sensor=Camera(m), agent=agent)
    world.append(r)

    world.draw()
