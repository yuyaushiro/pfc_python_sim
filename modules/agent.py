import numpy as np


class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def make_decision(self, observation=None):
        return self.nu, self.omega
