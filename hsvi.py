import numpy as np
from copy import deepcopy

from agent import AgentInterface
from world.deterministic_maze import DeterministicMazeModel

import pandas as pd

class hsvi(AgentInterface):
    """ 
    Un agent capable de résoudre un labyrinthe donné grâce à l'algorithme d'itération 
    sur les valeurs (VI = Value Iteration).
    """

    def __init__(self, maze_model: DeterministicMazeModel, gamma: float):

        self.gamma = gamma
        self.maze_model = maze_model
        self.upperBoundV = np.zeros([maze_model.ny, maze_model.nx])
        self.lowerBoundV = np.zeros([maze_model.ny, maze_model.nx])

        self.initBounds();

    def solve(self, error: float):
        """
        Main function. Launch explore() while stopping criterion is not reached.
        """
        pass

    def done(self, s, error) -> bool:
        """
        Returns wether upper- and lower- bounds are close enough for a specific state.
        """
        pass

    def select_action(self, s) -> int:
        """
        Selects the best action to do in state s.
        """
        pass

    def explore(self,error: float):
        """
        Make a trajectory through the Tree
        For each node, 
        (i) selects best action to do
        (ii) update bounds
        (iii) selects best next state to explore
        (iv) update bounds
        (v) returns
        """
        pass

    def initBounds(self):
        """
        Initialize the upper- and lower- value functions.
        """
        pass

    def update(self,s,v : float, upperOrLowerBound : int):
        """
        Update the upper- or the lower- value function in state s with the value v.
        """
        pass

    def computeReward(self,s,a) -> float:
        """
        Computes the (average) reward the agent would get by doing action a in state s.
        """
        pass

    def getValue(upperOrLowerBoundValue : int, s) -> float:
        """
        return the current value of a specific state s either relatively to the upper- or the lower- bound.
        """
        pass