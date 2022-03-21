# EE807 Special Topics in EE <Deep Reinforcement Learning and AlphaGo>
# Fall 2019, School of EE, KAIST
# written by Sae-Young Chung
# revision history
# 11/02/2018: written for EE807B

import os
# import cv2
import numpy as np
import gym
from gym import spaces

import assets
from .maze_generator import MazeGenerator


class Maze(gym.Env):
    """
    Description:
        Un robot recherche un objet dans un labyrinthe.
        Ce code n'a pas besoin d'être compris. On prendra simplement
        le soin savoir manipuler l'environnement "Maze" via ses méthodes:
            - __init__ (constructeur de Maze)
            - getDynamics (accède à la dynamique)
            - getReward (accède aux récompenses)
            - reset 
            - reset_using_existing_maze
            - step
            - render
            - from_file (crée un labyrinthe depuis un fichier)
    Observation:
        Type: Box(4)
        Num	    Observation         Min     Max
        y	    Y coordinate        0       ny
        x	    X coordinate        0       nx
    Actions:
        Type: Discrete(4)
        Num	    Action
        0	    Up
        1	    Down
        2	    Left
        3	    Right
    """
    actions = ['up', 'down', 'left', 'right']
    URL_ROBOT = os.path.join(os.path.dirname(assets.__file__), "robot.png")

    def __init__(self, nx, ny, min_shortest_length=0, mode='tabular'):
        self.nx = (nx // 2) * 2 + 1
        self.ny = (ny // 2) * 2 + 1
        self.min_shortest_length = min_shortest_length
        # number of planes (walls, agent's location, goal location)
        self.nf = 3
        # number of actions
        self.na = 4       
        # Can be 'tabular' or 'nn' 
        self.mode = mode
        
        _ = self.reset()

        self.action_space = spaces.Discrete(4)

        # Window dimensions
        max_width = 400
        max_height = 400
        self.pixel_per_case = min(max_width // nx, max_height // ny)

        # Viewer
        self.viewer = None
        self.bg_color = [255, 255, 255]
        self.wall_color = [50, 50, 50]
        self.robot_color = [0, 200, 100]
        self.init_color = [0, 0, 200]
        self.terminal_color = [200, 0, 0]

    def getDynamics(self, state, action, next_state):
        """ Retourne la probabilité de transition associé au problème déterministe
        """
        action_name = self.actions[action]
        proba = 0.
        if (state == self.terminal_state):
            return (next_state == self.terminal_state)
               
        if action_name == 'up':    # up
            if (self.maze[state[0]-1, state[1]] == 0) and (next_state == (state[0]-1, state[1])):  # can go up
                proba = 1
            elif (self.maze[state[0]-1, state[1]] == 1) and (next_state == state):  # cant go up
                proba = 1
        elif action_name == 'down':  # down
            if (self.maze[state[0]+1, state[1]] == 0) and (next_state == (state[0]+1, state[1])):  # can go down
                proba = 1
            elif (self.maze[state[0]+1, state[1]] == 1) and (next_state == state):  # cant go down
                proba = 1
        elif action_name == 'left':  # left
            if (self.maze[state[0], state[1]-1] == 0) and (next_state == (state[0], state[1]-1)):  # can go left
                proba = 1
            elif (self.maze[state[0], state[1]-1] == 1) and (next_state == state):  # cant go left
                proba = 1
        else:              # right
            if (self.maze[state[0], state[1]+1] == 0) and (next_state == (state[0], state[1]+1)):  # can go right
                proba = 1
            elif (self.maze[state[0], state[1]+1] == 1) and (next_state == state):  # cant go right
                proba = 1

        return proba

    def getReward(self, state, action):
        """ Retourne la récompense.
        """
        if (state == self.terminal_state):
            return 0
        else:
            return -1

    def reset(self):     
        """ Génère un nouveau labyrinthe.
        """
        while True:
            maze, shortest_length, p1, p2 = MazeGenerator.make_with_goal(self.nx, self.ny)
            if shortest_length >= self.min_shortest_length:
                break
        self.maze = maze
        self.shortest_length = shortest_length
        self.init_state = p1          # initial state (y,x)
        self.terminal_state = p2      # terminal state (y,x)
        return self.reset_using_existing_maze()

    # reset state without generating a new maze
    def reset_using_existing_maze(self):
        """ Initialise le problème dans un état initial.
        """
        self.terminal = 0             # 1 means game ended, 0 means game in progress
        self.loc = self.init_state  # current location (y,x) of agent as state
        if self.mode == 'tabular':    # tabular mode
            return self.loc
        else:    # neural network mode
            self.s = np.zeros([self.nf, self.ny, self.nx])
            self.s[0, :, :] = self.maze
            # initial location of agent
            self.s[1, self.init_state[0], self.init_state[1]] = 1
            self.s[2, self.terminal_state[0], self.terminal_state[1]] = 1   # goal location
            return np.copy(self.s)


    @classmethod
    def from_file(cls, filename: str):
        """ Charge un labyrinthe contenu dans un fichier texte. 
        
        Description:
            - P : Path 
            - W : Wall
            - S : Starting
            - G : Goal

        Exemple : 
            WWWWWWW
            WSPPPPW
            WWWWWPW
            WPPGWPW
            WPWWWPW
            WPPPPPW
            WWWWWWW

        :param filename: name of the file
        :type filename: str
        """
        file = open(filename, 'r')
        grid = file.readlines()
        maze_ = Maze(len(grid[0])-1, len(grid))
        for y, line in enumerate(grid):
            for x, val in enumerate(line):
                if (val not in {'\n', '\t', ' ', '\r'}):
                    #print(x," - ",y, " = ", val)
                    if val in {"W"}:
                        maze_.maze[y,x] = 1
                    elif val in {"P", "S", "G"}:
                        maze_.maze[y,x] = 0
                    else:
                        raise ValueError("When parsing, bad symbole were found : {} ".format(val))
                    if val is "G":
                        maze_.terminal_state = (y, x)
                    if val is "S":
                        maze_.init_state = (y, x)
        return maze_
                
    def step(self, action):     
        """ Execute une action dans l'environnement et retourne l'état suivant, la récompense et un booléen 
        indiquant si le système se trouve dans un état terminal.
        """
        if self.terminal:
            print(
                'Warning: maze_environment.run() has been called after reaching terminal = 1')
            if self.mode == 'tabular':
                return self.s, 0., self.terminal
            else:
                return np.copy(self.s), 0., self.terminal
        loc_new = self.loc
        if action == 0:    # up
            if self.maze[self.loc[0]-1, self.loc[1]] == 0:   # can go up
                loc_new = self.loc[0]-1, self.loc[1]
        elif action == 1:  # down
            if self.maze[self.loc[0]+1, self.loc[1]] == 0:
                loc_new = self.loc[0]+1, self.loc[1]
        elif action == 2:  # left
            if self.maze[self.loc[0], self.loc[1]-1] == 0:
                loc_new = self.loc[0], self.loc[1]-1
        else:              # right
            if self.maze[self.loc[0], self.loc[1]+1] == 0:
                loc_new = self.loc[0], self.loc[1]+1
        if loc_new == self.terminal_state:
            reward = -1.
            self.terminal = 1
        else:
            reward = -1.
        if self.mode == 'tabular':
            self.loc = loc_new
            return loc_new, reward, self.terminal
        else:
            self.s[1, self.loc[0], self.loc[1]] = 0
            self.s[1, loc_new[0], loc_new[1]] = 1
            self.loc = loc_new
            return np.copy(self.s), reward, self.terminal

    @classmethod
    def make(cls, width, height, complexity=.75, density=.75):
        return MazeGenerator.make(width, height, complexity, density)

    def render(self, mode='human', filename=''):
        # if filename != '':
        #     cv2.imwrite(filename, cv2.cvtColor(
        #         self.render_game(self.bg_color), cv2.COLOR_RGB2BGR))
        if mode == 'human':
            return self.render_human(mode)
        return self.render_array()

    def render_human(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer.imshow(self.render(mode='rbg_array'))

    def render_array(self):
        # color = np.array([0,0,0])
        return self.render_game(color_background=self.bg_color)

    def render_game(self, color_background=[50, 200, 200]):
        img = np.full(
            (self.pixel_per_case*self.ny, self.pixel_per_case*self.nx, 3),
            255,
            dtype=np.uint8)

        img[:, :, :3] = color_background

        self.render_walls(img)

        img[self.init_state[0]*self.pixel_per_case:(self.init_state[0]+1)*self.pixel_per_case, self.init_state[1]
            * self.pixel_per_case: (self.init_state[1]+1)*self.pixel_per_case, : 3] = self.init_color

        img[self.terminal_state[0]*self.pixel_per_case: (self.terminal_state[0]+1)*self.pixel_per_case, self.terminal_state[1]
            * self.pixel_per_case: (self.terminal_state[1]+1)*self.pixel_per_case, : 3] = self.terminal_color

        self.render_robot(img)

        return img

    def render_walls(self, img):
        for lig in range(len(self.maze)):
            for col in range(len(self.maze[lig])):
                if (self.maze[lig, col]):
                    img[lig*self.pixel_per_case:(lig+1)*self.pixel_per_case, col*self.pixel_per_case:(
                        col+1)*self.pixel_per_case, :3] = self.wall_color

    def render_robot(self, img):
        # img_cv2 = cv2.resize(cv2.imread(Maze.URL_ROBOT, cv2.IMREAD_UNCHANGED),
        #                      (self.pixel_per_case, self.pixel_per_case))[:, :, :3]
        padding = int(self.pixel_per_case * 0.3)
        img[self.loc[0]*self.pixel_per_case + padding : (self.loc[0]+1)*self.pixel_per_case - padding, self.loc[1] * self.pixel_per_case + padding:(
            self.loc[1]+1)*self.pixel_per_case - padding, :3] = self.robot_color #cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
