import numpy as np

from . import AgentInterface


class RandomAgent(AgentInterface):
    """
    A random agent.
    """

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def select_action(self, state):
        return np.random.randint(self.num_actions)
