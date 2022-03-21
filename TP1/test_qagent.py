#!/usr/bin/env python3

import unittest
import random

from agent.qagent import QAgent
from TP1.epsilon_profile import EpsilonProfile

from world.maze import Maze 

import numpy as np

class TestQAgent(unittest.TestCase):

    def test_04_updateQ(self):
        #learning constants 
        n_episodes = 100
        max_steps = 50
        gamma = 1.
        alpha = 0.001
        eps_profile = EpsilonProfile(1.0, 0.0001)

        #env
        env = Maze.from_file("tests/maze_ex2.txt") # Create a maze from a file
        _qagent = QAgent(env, eps_profile,gamma,alpha)
        #_qagent.learn(env,n_episodes,max_steps)
        _qagent.Q[(2,3)] = [-0.03883854, 0.0, -0.03896855, -0.03897572]
        _qagent.Q[(1,2)] = [-0.13596999, -0.13593551, -0.13639433, -0.13559852]
        _qagent.Q[(1,3)] = [-0.08694712, -0.08691367, -0.08704894, -0.08696433]

        r = ((1, 3), -1.0, 0)

        _qagent.updateQ((2,3),0,-1.0,(1,3))
        self.assertTrue(_qagent.Q[2,3,0]==-0.03988661513)
        _qagent.updateQ((2,3),1,1.0,(1,3))
        self.assertTrue(_qagent.Q[2,3,1]==0.0009130863300000001)

    def test_03_select_greedy_action(self):

        #learning constants 
        n_episodes = 100
        max_steps = 50
        gamma = 1.
        alpha = 0.001
        eps_profile = EpsilonProfile(1.0, 0.001)

        #env
        env = Maze.from_file("tests/maze_ex2.txt") # Create a maze from a file
        _qagent = QAgent(env, eps_profile,gamma,alpha)

        _qagent.Q[(2,3)] = [-0.03883854, 0.0, -0.03896855, -0.03897572]
        _qagent.Q[(1,2)] = [-0.13596999, -0.13593551, -0.13639433, -0.13559852]
        _qagent.Q[(1,3)] = [-0.08694712, -0.08691367, -0.08704894, -0.08696433]

        r = ((1, 3), -1.0, 0)

        self.assertTrue(_qagent.select_greedy_action((1,3))==1)
        self.assertTrue(_qagent.select_greedy_action((1,2))==3)

    def test_02_select_action(self):

        #learning constants 
        n_episodes = 100
        max_steps = 50
        gamma = 1.
        alpha = 0.001
        eps_profile = EpsilonProfile(0.001, 0.001)

        #env
        env = Maze.from_file("tests/maze_ex2.txt") # Create a maze from a file
        _qagent = QAgent(env, eps_profile,gamma,alpha)

        _qagent.Q[(2,3)] = [-0.03883854, 0.0, -0.03896855, -0.03897572]
        _qagent.Q[(1,2)] = [-0.13596999, -0.13593551, -0.13639433, -0.13559852]
        _qagent.Q[(1,3)] = [-0.08694712, -0.08691367, -0.08704894, -0.08696433]

        r = ((1, 3), -1.0, 0)

        res=[]
        for i in range(10000):
            res.append(_qagent.select_action((1,3)))
        self.assertTrue(np.sum(np.asarray(res))/len(res)>0.99 and np.sum(np.asarray(res))/len(res)< 1.01)

        _qagent.epsilon = 1.0
        res=[]
        for i in range(100000):
            res.append(_qagent.select_action((1,3)))
        print("\n val : ",np.sum(np.asarray(res))/len(res))
        self.assertTrue(np.sum(np.asarray(res))/len(res)>1.49 and np.sum(np.asarray(res))/len(res)< 1.51)

    def test_01_learn(self):

        #learning constants 
        n_episodes = 1000
        max_steps = 100
        gamma = 1.
        alpha = 0.001
        eps_profile = EpsilonProfile(1.0, 0.0001)

        #env
        env = Maze.from_file("tests/maze_ex1.txt") # Create a maze from a file
        _qagent = QAgent(env, eps_profile,gamma,alpha)
        _qagent.learn(env,n_episodes,max_steps)
        sol = [[[ 0.,          0.,          0.,          0.        ],
  [ 0.,          0.,          0.,          0.        ],
  [ 0.,          0.,          0.,          0.        ],
  [ 0.,          0.,          0.,          0.         ],
  [ 0.,          0.,          0.,          0.         ],
  [ 0.,          0.,          0.,          0.        ],
  [ 0.,          0.,          0.,          0.         ]],

  [[ 0.,          0.,          0.,          0.        ],
  [-2.91635605, -2.9158402 , -2.91657386, -2.9164166 ],
  [-2.70586108, -2.70650246, -2.70576875, -2.70640025],
  [-2.48746921, -2.4881926 , -2.48764199, -2.48810045],
  [-2.26048366, -2.26028633, -2.2609845 , -2.26104212],
  [-2.02473943, -2.02478894, -2.02429746, -2.02439835],
  [ 0.,          0.,          0.,          0.        ]],

 [[ 0.,          0.,          0.,          0.        ],
  [ 0.,          0.,          0.,          0.        ],
  [ 0.,          0.,          0.,          0.        ],
  [ 0.,          0.,          0.,          0.        ],
  [ 0.,          0.,          0.,          0.        ],
  [-1.7766618,  -1.7762219,  -1.77591096, -1.77610667],
  [ 0.,         0.,          0.,          0.        ]],

 [[ 0.,          0.,          0.,          0.        ],
  [-0.05996794, -0.06004588, -0.05998859, -0.06001859],
  [-0.0279895,  -0.02799325, -0.02843982, -0.02859763],
  [ 0.,          0.,          0.,          0.        ],
  [ 0.,          0.,          0.,          0.        ],
  [-1.51410223, -1.51396483, -1.51407916, -1.51346672],
  [ 0.,          0.,          0.,          0.        ]],

 [[ 0.,          0.,          0.,          0.        ],
  [-0.0990449,  -0.09951131, -0.0989625,  -0.09898472],
  [ 0.,          0.,          0.,          0.        ],
  [ 0.,          0.,          0.,          0.        ],
  [ 0.,          0.,          0.,          0.        ],
  [-1.23282727, -1.23228355, -1.23211743, -1.23197262],
  [ 0.,          0.,          0.,          0.        ]],

 [[ 0.,          0.,          0.,          0.        ],
  [-0.15551516, -0.15631678, -0.1559694 , -0.15593626],
  [ 0.,          0.,          0.,          0.        ],
  [-0.68458931, -0.68481541, -0.68475955, -0.68427105],
  [-0.76332095, -0.76329818, -0.76386523, -0.76404934],
  [-0.92891144, -0.92801454, -0.92862772, -0.92859971],
  [ 0.,          0.,          0.,          0.        ]],

 [[ 0.,          0.,          0.,          0.        ],
  [-0.22740779, -0.2280408 , -0.22695536, -0.22787698],
  [ 0.,          0.,          0.,          0.        ],
  [-0.63075964, -0.63117803, -0.63050747, -0.63119937],
  [-0.69033811, -0.68990731, -0.69010835, -0.68939419],
  [-0.75851084, -0.75809737, -0.75844511, -0.75755689],
  [ 0.,          0.,          0.,          0.        ]],

 [[ 0.,          0.,          0.,          0.        ],
  [-0.31756677, -0.3178784 , -0.31781235, -0.31830951],
  [-0.42677634, -0.42669539, -0.42697339, -0.42751154],
  [-0.55555325, -0.55574177, -0.55540412, -0.55617257],
  [-0.62987017, -0.63066622, -0.6298833 , -0.63087541],
  [-0.67727361, -0.67662248, -0.67712506, -0.67702963],
  [ 0.,          0.,          0.,          0.        ]],

 [[ 0.,          0.,          0.,          0.        ],
  [ 0.,          0.,          0.,          0.        ],
  [ 0.,          0.,          0.,          0.        ],
  [ 0.,          0.,          0.,          0.        ],
  [ 0.,          0.,          0.,          0.        ],
  [ 0.,          0.,          0.,          0.        ],
  [ 0.,          0.,          0.,          0.        ]]]
        print("\n sol.shape : ",np.asarray(sol[0][0]), _qagent.Q[0,0])
        for i in range(env.ny):
            for j in range(env.nx):
                print("\n _qagent.Q[i,j] : ", _qagent.Q[i,j])
                print("\n sol[i,j] : ", sol[i][j])
                print("\n val : ", abs(np.max(_qagent.Q[i,j]-sol[i][j])))
                self.assertTrue(abs(np.max(_qagent.Q[i,j]-sol[i][j]))<0.02)

        #print("\n norm val : ",_qagent.Q-np.asarray(sol))
        #self.assertTrue(np.linalg.norm(_qagent.Q-sol)<0.001)    
if __name__ == "__main__":
    unittest.main()
