import gym
from gym import error, spaces, utils
from gym.utils import seeding


class GRNControlSimpleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # TODO
        ...

    def step(self, action):
        # TODO
        ...

    def reset(self):
        # TODO
        ...

    def render(self, mode='human'):
        # TODO
        ...

    def close(self):
        # TODO
        ...
