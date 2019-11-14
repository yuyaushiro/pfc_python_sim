from modules.world import World, Landmark, Map
from modules.robot import IdealRobot
from modules.sensor import IdealCamera, Camera
from modules.agent import Agent

import math
import numpy as np


if __name__ == '__main__':   ###name_indent
    world = World(30, 0.1)

    ### 地図を生成して3つランドマークを追加 ###
    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]: m.append_landmark(Landmark(*ln))
    world.append(m)

    ### ロボットを作る ###
    straight = Agent(0.2, 0.0)
    circling = Agent(0.2, 10.0/180*math.pi)
    robot1 = IdealRobot( np.array([ -2, -3, math.pi/6]).T,    sensor=Camera(m), agent=straight )
    robot2 = IdealRobot( np.array([-2, -1, math.pi/5*6]).T, sensor=IdealCamera(m), agent=circling, color="red")
    world.append(robot1)
    world.append(robot2)

    ### アニメーション実行 ###
    world.draw()
