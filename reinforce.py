import numpy as np
from copy import deepcopy

from agent import AgentInterface
from world.deterministic_maze import DeterministicMazeModel

import pandas as pd

class reinforce(AgentInterface):
    """ 
    Un agent capable de résoudre un labyrinthe donné grâce à l'algorithme d'itération 
    sur les valeurs (VI = Value Iteration).
    """

    def __init__(self, maze_model: DeterministicMazeModel, gamma: float,alpha: float):

        self.gamma = gamma
        self.maze_model = maze_model
        self.V = np.zeros([maze_model.ny, maze_model.nx])
        self.mazeValues = pd.DataFrame(data={'nx': maze_model.nx, 'ny': [maze_model.ny]})
        self.alpha = alpha
        self.theta = np.random.random((maze_model.nx*maze_model.ny,4))

    def solve(self, error: float):
        """
        Main loop, stops whenever a quality criterion is reached
        Mainly generates trajectories according to current policy. 
        After each trajectory, update the policy using computed gradients for each (state,action)
        """

    def policy(self,s):
        """
        encodes the current policy distribution over possible actions in s
        """
        res = np.random.random(4)
        """
        computes policy and stores it in res
        """
        return res
    def compute_gradient(self, s, a) -> float:
        """
        computes the current gradient in state s for action a
        """

        pass

    def generate_action(self,s) -> int:
        """
        generates an action according to the current policy
        """
        return 0

    def store(self,s,a,p,r):
        """
        store the node,action,prob and reward by adding it to the current trajectory
        """
        return
        
    def softmax(self,s,a) :
        pass
