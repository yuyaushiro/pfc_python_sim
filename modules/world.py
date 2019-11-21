import matplotlib
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np
import datetime


# 世界を管理するクラス
class World:
    def __init__(self, time_span, time_interval, debug=False,
                 recording_file_name=None, playback_speed=1):
        self.objects = []
        self.debug = debug
        self.time_span = time_span
        self.time_interval = time_interval
        self.recording_file_name = recording_file_name
        self.playback_speed = playback_speed

    # 世界にオブジェクトを追加する
    def append(self,obj):
        self.objects.append(obj)

    # 世界を描画する
    def draw(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        ax.set_xlabel("X",fontsize=10)
        ax.set_ylabel("Y",fontsize=10)

        elems = []

        if self.debug:
            for i in range(int(self.time_span/self.time_interval)): self.one_step(i, elems, ax)
        else:
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax),
                                     frames=int(self.time_span/self.time_interval)+1, interval=int(self.time_interval*1000), repeat=False)
            if self.recording_file_name is None:
                plt.show()
            else:
                Writer = anm.writers['ffmpeg']
                fps = self.playback_speed / self.time_interval
                writer = Writer(fps=fps)
                now = datetime.datetime.now()
                self.ani.save('movie/' + self.recording_file_name + now.strftime('-%Y%m%d-%H%M%S') + '.mp4',
                              writer=writer, dpi=250)

    # 世界を 1 ステップ進める
    def one_step(self, i, elems, ax):
        while elems: elems.pop().remove()
        time_str = "t = %.2f[s]" % (self.time_interval*i)
        print(time_str)
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"): obj.one_step(self.time_interval)


# ランドマーククラス
class Landmark:
    def __init__(self, x, y):
        self.pos = np.array([x, y]).T
        self.id = None

    def draw(self, ax, elems):
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="landmarks", color="orange")
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10))


# ランドマーク管理のクラス
class Map:
    def __init__(self):
        self.landmarks = []

    def append_landmark(self, landmark):
        landmark.id = len(self.landmarks)
        self.landmarks.append(landmark)

    def draw(self, ax, elems):
        for lm in self.landmarks: lm.draw(ax, elems)


class Goal:
    def __init__(self, x, y, radius=0.3, value=0.0):
        self.pos = np.array([x, y]).T
        self.radius = radius
        self.value = value

    def inside(self, pose):
        return self.radius > math.sqrt( (self.pos[0]-pose[0])**2 + (self.pos[1]-pose[1])**2 )

    def draw(self, ax, elems):
        x, y = self.pos
        c = ax.scatter(x + 0.16, y + 0.5, s=50, marker=">", label="landmarks", color="red")
        elems.append(c)
        elems += ax.plot([x, x], [y, y + 0.6], color="black")
