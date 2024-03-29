# EE807 Special Topics in EE <Deep Reinforcement Learning and AlphaGo>
# Fall 2019, School of EE, KAIST
# written by Sae-Young Chung
# revision history
# 11/02/2018: written for EE807B

import os
import pygame

import numpy as np
import assets
from .maze_generator import MazeGenerator


class Maze:
    """
    Description:
        Un robot recherche un objet dans un labyrinthe.
        Ce code n'a pas besoin d'être compris. On prendra simplement
        le soin savoir manipuler l'environnement "Maze" via ses méthodes:
            - __init__ (constructeur de Maze)
            - getStates (renvoie l'ensemble des états)
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

    actions = ["up", "down", "left", "right"]
    URL_ROBOT = os.path.join(os.path.dirname(assets.__file__), "dav.png")
    URL_WALL = os.path.join(os.path.dirname(assets.__file__), "wall.png")
    URL_GROUND = os.path.join(os.path.dirname(assets.__file__), "ground.png")
    URL_START = os.path.join(os.path.dirname(assets.__file__), "start.png")
    URL_EXIT = os.path.join(os.path.dirname(assets.__file__), "exit.png")

    def __init__(self, nx, ny, min_shortest_length=0, mode="tabular"):
        self.nx = (nx // 2) * 2 + 1
        self.ny = (ny // 2) * 2 + 1
        self.min_shortest_length = min_shortest_length
        # number of planes (walls, agent's location, goal location)
        self.nf = 3
        # number of actions
        self.na = 4
        # Can be 'tabular' or 'nn'
        self.mode = mode

        self.states = [(y, x) for y in range(self.ny) for x in range(self.nx)]

        _ = self.reset()

        # Window dimensions
        max_width = 700
        max_height = 700
        self.pixel_per_case = min(max_width // nx, max_height // ny)

        # Init pygame
        pygame.init()
        self.window = pygame.display.set_mode((self.ny * self.pixel_per_case, self.nx * self.pixel_per_case), 0)
        pygame.display.set_caption("Introduction à l'apprentissage par renforcement")

        # Load assets
        self.load_assets()

    @property
    def action_space(self) -> list[str]:
        """Renvoie les actions possibles"""
        return Maze.actions

    @property
    def state_space(self) -> list[tuple[int, int]]:
        """Renvoie les états possibles du labyrinthe (i.e. les coordonnée (y, x))"""
        return self.states

    def getDynamics(self, state, action, next_state):
        """Retourne la probabilité de transition associé au problème déterministe"""
        action_name = self.actions[action]
        proba = 0.0
        if state == self.terminal_state:
            return next_state == self.terminal_state

        if action_name == "up":  # up
            if (self.maze[state[0] - 1, state[1]] == 0) and (next_state == (state[0] - 1, state[1])):  # can go up
                proba = 1
            elif (self.maze[state[0] - 1, state[1]] == 1) and (next_state == state):  # cant go up
                proba = 1
        elif action_name == "down":  # down
            if (self.maze[state[0] + 1, state[1]] == 0) and (next_state == (state[0] + 1, state[1])):  # can go down
                proba = 1
            elif (self.maze[state[0] + 1, state[1]] == 1) and (next_state == state):  # cant go down
                proba = 1
        elif action_name == "left":  # left
            if (self.maze[state[0], state[1] - 1] == 0) and (next_state == (state[0], state[1] - 1)):  # can go left
                proba = 1
            elif (self.maze[state[0], state[1] - 1] == 1) and (next_state == state):  # cant go left
                proba = 1
        else:  # right
            if (self.maze[state[0], state[1] + 1] == 0) and (next_state == (state[0], state[1] + 1)):  # can go right
                proba = 1
            elif (self.maze[state[0], state[1] + 1] == 1) and (next_state == state):  # cant go right
                proba = 1

        return proba

    def getReward(self, state, action):
        """Retourne la récompense."""
        return 0 if state == self.terminal_state else -1

    def reset(self):
        """Génère un nouveau labyrinthe."""
        while True:
            maze, shortest_length, p1, p2 = MazeGenerator.make_with_goal(self.nx, self.ny)
            if shortest_length >= self.min_shortest_length:
                break
        self.maze = maze
        self.shortest_length = shortest_length
        self.init_state = p1  # initial state (y,x)
        self.terminal_state = p2  # terminal state (y,x)
        return self.reset_using_existing_maze()

    # reset state without generating a new maze
    def reset_using_existing_maze(self):
        """Initialise le problème dans un état initial."""
        self.terminal = 0  # 1 means game ended, 0 means game in progress
        self.loc = self.init_state  # current location (y,x) of agent as state
        if self.mode == "tabular":  # tabular mode
            return self.loc
        else:  # neural network mode
            self.s = np.zeros([self.nf, self.ny, self.nx])
            self.s[0, :, :] = self.maze
            # initial location of agent
            self.s[1, self.init_state[0], self.init_state[1]] = 1
            self.s[2, self.terminal_state[0], self.terminal_state[1]] = 1  # goal location
            return np.copy(self.s)

    @classmethod
    def from_file(cls, filename: str):
        """Charge un labyrinthe contenu dans un fichier texte.

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
        file = open(filename, "r")
        grid = file.readlines()
        maze_ = Maze(len(grid[0]) - 1, len(grid))
        for y, line in enumerate(grid):
            for x, val in enumerate(line):
                if val not in {"\n", "\t", " ", "\r"}:
                    # print(x," - ",y, " = ", val)
                    if val in {"W"}:
                        maze_.maze[y, x] = 1
                    elif val in {"P", "S", "G"}:
                        maze_.maze[y, x] = 0
                    else:
                        raise ValueError("When parsing, bad symbole were found : {} ".format(val))
                    if val is "G":
                        maze_.terminal_state = (y, x)
                    if val is "S":
                        maze_.init_state = (y, x)
        return maze_

    def step(self, action):
        """Execute une action dans l'environnement et retourne l'état suivant, la récompense et un booléen
        indiquant si le système se trouve dans un état terminal.
        """
        if self.terminal:
            print("Warning: maze_environment.run() has been called after reaching terminal = 1")
            if self.mode == "tabular":
                return self.s, 0.0, self.terminal
            else:
                return np.copy(self.s), 0.0, self.terminal
        loc_new = self.loc
        if action == 0:  # up
            if self.maze[self.loc[0] - 1, self.loc[1]] == 0:  # can go up
                loc_new = self.loc[0] - 1, self.loc[1]
        elif action == 1:  # down
            if self.maze[self.loc[0] + 1, self.loc[1]] == 0:
                loc_new = self.loc[0] + 1, self.loc[1]
        elif action == 2:  # left
            if self.maze[self.loc[0], self.loc[1] - 1] == 0:
                loc_new = self.loc[0], self.loc[1] - 1
        else:  # right
            if self.maze[self.loc[0], self.loc[1] + 1] == 0:
                loc_new = self.loc[0], self.loc[1] + 1
        if loc_new == self.terminal_state:
            reward = -1.0
            self.terminal = 1
        else:
            reward = -1.0
        if self.mode == "tabular":
            self.loc = loc_new
            return loc_new, reward, self.terminal
        else:
            self.s[1, self.loc[0], self.loc[1]] = 0
            self.s[1, loc_new[0], loc_new[1]] = 1
            self.loc = loc_new
            return np.copy(self.s), reward, self.terminal

    @classmethod
    def make(cls, width, height, complexity=0.75, density=0.75):
        return MazeGenerator.make(width, height, complexity, density)

    def render(self):
        self.render_maze()
        self.render_robot()
        pygame.display.flip()

    def render_maze(self) -> None:
        for lig in range(len(self.maze)):
            for col in range(len(self.maze[lig])):
                self.window.blit(self.img_ground, (lig * self.pixel_per_case, col * self.pixel_per_case))

        self.window.blit(
            self.img_start, (self.init_state[0] * self.pixel_per_case, self.init_state[1] * self.pixel_per_case)
        )

        self.window.blit(
            self.img_exit,
            (self.terminal_state[0] * self.pixel_per_case, self.terminal_state[1] * self.pixel_per_case),
        )

        self.render_walls()

    def render_walls(self):
        for lig in range(len(self.maze)):
            for col in range(len(self.maze[lig])):
                if self.maze[lig, col]:
                    self.window.blit(self.img_wall, (lig * self.pixel_per_case, col * self.pixel_per_case))

    def render_robot(self) -> None:
        self.window.blit(self.img_robot, (self.loc[0] * self.pixel_per_case, self.loc[1] * self.pixel_per_case))

    def load_assets(self) -> None:
        self.img_robot = pygame.image.load(Maze.URL_ROBOT)
        self.img_robot = pygame.transform.scale(self.img_robot, (self.pixel_per_case, self.pixel_per_case))

        self.img_wall = pygame.image.load(Maze.URL_WALL)
        self.img_wall = pygame.transform.scale(self.img_wall, (self.pixel_per_case, self.pixel_per_case))

        self.img_ground = pygame.image.load(Maze.URL_GROUND)
        self.img_ground = pygame.transform.scale(self.img_ground, (self.pixel_per_case, self.pixel_per_case))

        self.img_start = pygame.image.load(Maze.URL_START)
        self.img_start = pygame.transform.scale(self.img_start, (self.pixel_per_case, self.pixel_per_case))

        self.img_exit = pygame.image.load(Maze.URL_EXIT)
        self.img_exit = pygame.transform.scale(self.img_exit, (self.pixel_per_case, self.pixel_per_case))
