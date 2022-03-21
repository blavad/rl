#!/usr/bin/env python3

import unittest
import random
import numpy as np

from world.maze import Maze 
from agent.viagent import VIAgent

class TestVIAgent(unittest.TestCase):

    def test_00_init(self):

        nx = random.randint(10,15)
        ny = nx
        _maze = Maze(nx,ny,30)

        gamma = random.random()
        _viagent = VIAgent(_maze, gamma)
        self.assertTrue(_viagent.maze_model==_maze);
        self.assertTrue(_viagent.gamma==gamma);
        self.assertTrue(len(_viagent.V)>0);
        for x in range(len(_viagent.V)):
            for y in range(len(_viagent.V[0])):
                self.assertTrue(_viagent.V[x,y]==0)

    def test_01_solve(self):

        env = Maze.from_file("tests/maze_ex1.txt") # Create a maze from a file

        sol = np.asarray(
            [[ 0., 0., 0., 0., 0., 0., 0.],
                [  0., -20.,-19., -18., -17., -16.,   0.],
                [  0.,  0.,  0.,   0.,   0., -15.,   0.],
                [  0., -2., -1.,   0.,   0., -14.,   0.],
                [  0., -3.,  0.,   0.,   0., -13.,   0.],
                [  0., -4.,  0., -10., -11., -12.,   0.],
                [  0., -5.,  0.,  -9., -10., -11.,   0.],
                [  0., -6., -7.,  -8.,  -9., -10.,   0.],
                [  0.,  0.,  0.,   0.,   0.,   0.,   0.]])
        _viagent = VIAgent(env, 1.0)
        _viagent.solve(0.01)
        self.assertTrue(np.array_equal(np.asarray(_viagent.V),sol))

    def test_02_done(self):
        nx = random.randint(10,15)
        ny = nx
        _maze = Maze(nx,ny,30)

        error = 0.001
        _viagent = VIAgent(_maze, 1.0)
        self.assertTrue(_viagent.done(_viagent.V,(_viagent.V-error/2),error))
        self.assertFalse(_viagent.done(_viagent.V,(_viagent.V-error*2),error))

    def test_03_belllman_operator(self):
        env = Maze.from_file("tests/maze_ex2.txt") # Create a maze from a file
        _viagent = VIAgent(env, 1.0)
        self.assertTrue(_viagent.bellman_operator((2,3))==-1)
        _viagent.solve(0.01)
        _viagent.V = np.asarray([[ 0., -4.,  0.,  0.,  0.],[ 0., -4., -3., -2.,  0.],[ 0., -4.,  0., -1.,  0.],
            [ 0., -4.,  0.,  0.,  0.],[ 0.,  0.,  0.,  0.,  0.]])
        with self.assertRaises(Exception) as context:
            _viagent.bellman_operator((0,0))
        self.assertTrue(len(str(context.exception))>0)
        self.assertTrue(_viagent.bellman_operator((2,1))==-5)

    def test_04_select_action(self):
        env = Maze.from_file("tests/maze_ex2.txt") # Create a maze from a file
        _viagent = VIAgent(env, 1.0)
        _viagent.V = np.asarray([[ 0., -5.,  0.,  0.,  0.,],[ 0., -4., -3., -2.,  0.],[ 0., -5.,  0., -1.,  0.],
            [ 0., -6.,  0.,  0.,  0.],[ 0.,  0.,  0.,  0.,  0.]])
        s = (2,3)
        self.assertTrue(_viagent.select_action((2,3))==1)
        with self.assertRaises(Exception) as context:
            _viagent.select_action((0,0))
        self.assertTrue(len(str(context.exception))>0)
        self.assertTrue(_viagent.select_action((3,1))==0)

if __name__ == "__main__":
    unittest.main()
