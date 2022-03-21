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

        #to store trajectories
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []

    def solve(self, N: int):
        """
        Main loop, stops whenever a the limit of iteration is reached
        Mainly generates trajectories according to current policy. 
        After each trajectory, update the policy using computed gradients for each (state,action)
        """
        
        self.rewards -= np.mean(self.rewards)
        self.rewards /= np.std(self.rewards)
        for t in range(len(self.states)):
            s = self.states[t]
            a = self.actions[t]
            r = self._R(t)
            grad = self._gradient(s, a)
            self.theta = self.theta + self.alpha * r * grad
        # print(self.theta)
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []

        return

    def policy(self,s):
        """
        encodes the current policy distribution over possible actions in s
        """
        res = np.random.random(4)
        """
        computes policy and stores it in res
        """
        weights = np.empty(self.maze_model.nx * self.maze_model.ny)
        for a in range(4):
            weights[a] = self._softmax(s, a)
        res = weights / np.sum(weights)
        return res
    def compute_gradient(self, s, a) -> float:
        """
        computes the current gradient in state s for action a
        """
        expected = 0
        probs = self.pi(s)
        for b in range(0, self.action_size):
            expected += probs[b] * self._phi(s, b)
        return self._phi(s, a) - expected

    def generate_action(self,s) -> int:
        """
        generates an action according to the current policy
        """
        probs = self.pi(state)
        a = random.choices(range(0, self.action_size), weights=probs)
        a = a[0]
        pi = probs[a]
        return (a, pi)

    def store(self,s,a,p,r):
        """
        store the node,action,prob and reward by adding it to the current trajectory
        """
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)
        return

    def softmax(self,s,a) :
        return np.exp(self.theta.dot(self._phi(s, a)) / 100)

    def _phi(self, s, a):
        encoded = np.zeros([4, self.maze_model.nx*self.maze_model.ny])
        encoded[a] = s
        return encoded.flatten()

    def _R(self, t):
        """Reward function."""
        total = 0
        for tau in range(t, len(self.rewards)):
            total += self.gamma**(tau - t) * self.rewards[tau]
        return total