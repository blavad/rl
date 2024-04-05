import random
from . import AgentInterface


class RandomAgent(AgentInterface):
    """
    A random agent.
    """

    def __init__(self, action_space: list[any]):
        self.action_space = action_space

    def select_action(self, state):
        return random.choice(self.action_space)
