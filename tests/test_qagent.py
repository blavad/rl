#!/usr/bin/env python3

import unittest
import random

from agent.qagent import QAgent
from epsilon_profile import EpsilonProfile

from world.maze import Maze 


class TestQAgent(unittest.TestCase):
    def setUp(self):
        # self._maze = Maze::from_file("maze_q.txt")
        self._qagent = QAgent(self._maze, EpsilonProfile(1.0, 1.0), 0.5)
        self._qagent2 = QAgent(self._maze, EpsilonProfile(0.5, 0.5), 0.5)
        self._qagent1 = QAgent(self._maze, EpsilonProfile(0.0, 0.0), 0.5)

    def test_00_qupdate(self):
        s = (1,1)
        a = 0
        r=  1.
        s_ =(1,2)
        res = 5 # TO DO
        self._qagent.updateQ(s,a,r,s_)

        self.assertTrue((self._qagent.Q[s] == res))

    def test_01_is_solved(self):
        self.assertFalse(self._solver.is_solved())

    def test_02_solve_step(self):
        self._solver.solve_step()
        self.assertEqual(list(self._grid.get_row(0))[7], 6)
        self.assertEqual(list(self._grid.get_row(2))[6], 8)
        self.assertEqual(list(self._grid.get_row(6))[6:], [9, 5, 8])
        self.assertEqual(list(self._grid.get_row(7))[8], 4)

    def test_03_solve(self):
        sol = self._solver.solve()
        for i, row in enumerate(([3, 4, 9, 2, 8, 7, 5, 6, 1],
                [5, 8, 2, 6, 4, 1, 7, 9, 3],
                [6, 1, 7, 5, 3, 9, 8, 4, 2],
                [2, 3, 4, 1, 9, 5, 6, 8, 7],
                [7, 5, 1, 8, 6, 3, 4, 2, 9],
                [8, 9, 6, 7, 2, 4, 1, 3, 5],
                [1, 6, 3, 4, 7, 2, 9, 5, 8],
                [9, 7, 8, 3, 5, 6, 2, 1, 4])):
            self.assertEqual(list(sol.get_row(i)), row)


if __name__ == "__main__":
    unittest.main()
