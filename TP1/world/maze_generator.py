import numpy as np
from numpy.random import random_integers as rand

# maze generator using randomized Prim's algorithm
# from https://en.wikipedia.org/wiki/Maze_generation_algorithm
class MazeGenerator:
    @classmethod
    def make(cls, width, height, complexity=.75, density=.75):
        # Only odd shapes
        nx = (width // 2) * 2 + 1
        ny = (height // 2) * 2 + 1
        shape = ny, nx
        # Adjust complexity and density relative to maze size
        # number of components
        complexity = int(complexity * (5 * (shape[0] + shape[1])))
        # size of components
        density = int(density * ((shape[0] // 2) * (shape[1] // 2)))
        # Build actual maze
        Z = np.zeros(shape, dtype=np.int)
        # Fill borders
        Z[0, :] = Z[-1, :] = 1
        Z[:, 0] = Z[:, -1] = 1
        # Make aisles
        for i in range(density):
            x, y = rand(0, shape[1] // 2) * 2, rand(0,
                                                    shape[0] // 2) * 2  # pick a random position
            Z[y, x] = 1
            for j in range(complexity):
                neighbours = []
                if x > 1:
                    neighbours.append((y, x - 2))
                if x < shape[1] - 2:
                    neighbours.append((y, x + 2))
                if y > 1:
                    neighbours.append((y - 2, x))
                if y < shape[0] - 2:
                    neighbours.append((y + 2, x))
                if len(neighbours) > 0:
                    y_, x_ = neighbours[rand(0, len(neighbours) - 1)]
                    if Z[y_, x_] == 0:
                        Z[y_, x_] = 1
                        Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_
        return Z

    # pick a random point in a maze that is not a wall and not equal to p2
    # if not found any, returns (-1,-1)
    @classmethod
    def pick_random_point(cls, m, p2=(-1, -1)):
        ny, nx = m.shape
        exist = False
        for y, x in ((y, x) for y in range(ny) for x in range(nx)):
            if m[y, x] == 0 and (y, x) != p2:
                exist = True
                break
        if not exist:
            return (-1, -1)
        while True:
            p = (np.random.randint(ny), np.random.randint(nx))
            if m[p] == 0 and p != p2:
                break
        return p

    # returns the length of the shortest path between p1 and p2
    # returns 0 if not reacheable
    # pick random points if p1 and p2 are not specified
    # also returns p1 and p2

    @classmethod
    def check_maze(cls, m, p1=(-1, -1), p2=(-1, -1)):
        ny, nx = m.shape
        if p1 == (-1, -1):
            p1 = MazeGenerator.pick_random_point(m, p2)
            if p1 == (-1, -1):
                return 0, p1, p2
        if p2 == (-1, -1):
            p2 = MazeGenerator.pick_random_point(m, p1)
            if p2 == (-1, -1):
                return 0, p1, p2
        # run value iteration to find the shortest path from p1 to p2
        v = np.zeros(m.shape, dtype=np.int)
        vn = np.copy(v)
        for i in range(nx * ny):
            for y, x in ((y, x) for y in range(ny) for x in range(nx)):
                if (y, x) == p2:
                    continue       # v[p2] is always 0 since p2 is the goal
                if m[y, x] == 0:
                    vn[y, x] = nx * ny
                    if x > 0 and m[y, x-1] == 0:
                        vn[y, x] = min([vn[y, x], 1 + v[y, x-1]])
                    if x < nx-1 and m[y, x+1] == 0:
                        vn[y, x] = min([vn[y, x], 1 + v[y, x+1]])
                    if y > 0 and m[y-1, x] == 0:
                        vn[y, x] = min([vn[y, x], 1 + v[y-1, x]])
                    if y < ny-1 and m[y+1, x] == 0:
                        vn[y, x] = min([vn[y, x], 1 + v[y+1, x]])
                    if vn[y, x] == nx * ny:
                        vn[y, x] = 0
            v = np.copy(vn)
        return v[p1], p1, p2

    # returns m (maze), v (length of shortest path from start to end)
    #   p1 (start point), p2 (end point)
    @classmethod
    def make_with_goal(cls, width, height):
        while True:
            m = MazeGenerator.make(width, height)
            v, p1, p2 = MazeGenerator.check_maze(m)
            if v > 0:
                return m, v, p1, p2
