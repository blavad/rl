import numpy as np
from copy import deepcopy

from agent import AgentInterface
from world.deterministic_maze import DeterministicMazeModel


class VIAgent(AgentInterface):
    """ 
    Un agent capable de résoudre un labyrinthe donné grâce à l'algorithme d'itération 
    sur les valeurs (VI = Value Iteration).
    """

    def __init__(self, maze_model: DeterministicMazeModel, gamma: float = 1.0):
        """"À COMPLÉTER!
        Ce constructeur initialise une nouvelle instance de la classe ValueIteration.
        Il doit stocker les différents paramètres nécessaires au fonctionnement de l'algorithme et initialiser la 
        fonction de valeur d'état, notée V.

        :param maze_model: Le modèle du problème
        :type maze_model: DeterministicMazeModel
        :param gamma: le discount factor, defaults to 1.0
        :type gamma: float, optional
        """
        self.gamma = gamma
        self.maze_model = maze_model
        self.V = np.zeros([maze_model.ny, maze_model.nx])

    def solve(self, error: float):
        """À COMPLÉTER!
        Cette méthode résoud le problème avec une tolérance donnée.
        """
        n_iteration = 0
        V_copy = np.zeros([self.maze_model.ny, self.maze_model.nx])

        while ((n_iteration == 0) or not self.done(self.V, V_copy, error)):
            n_iteration += 1
            self.V = deepcopy(V_copy)
            for y in range(self.maze_model.ny):
                for x in range(self.maze_model.nx):
                    if (not self.maze_model.maze[y, x]):
                        V_copy[y, x] = self.bellman_operator((y, x))

    def done(self, V, V_copy, error) -> bool:
        """À COMPLÉTER!
        Cette méthode retourne vraie si la condition d'arrêt de 
        l'algorithme est vérifiée. Sinon elle retourne faux.
        """
        return (abs(V - V_copy).max() < error)

    def bellman_operator(self, s) -> float:
        """À COMPLÉTER!
        Cette méthode calcul l'opérateur de mise à jour de bellman pour un état s.

        :param s: Un état quelconque
        :return: La valeur de mise à jour de la fonction de valeur
        """
        max_value = -np.infty
        for a in range(self.maze_model.na):
            q_s_a = 0.
            for next_y in range(self.maze_model.ny):
                for next_x in range(self.maze_model.nx):
                    q_s_a += self.maze_model.T((s[0], s[1]), a, (next_y, next_x)) * (self.maze_model.R(
                        s, a) + self.gamma * self.V[next_y, next_x])
            if (q_s_a > max_value):
                max_value = q_s_a
        return max_value

    def select_action(self, s):
        """À COMPLÉTER!
        Cette méthode retourne l'action optimale.

        :param state: L'état courant
        :return: L'action optimale
        """
        max_value = -np.infty
        amax = 0
        for a in range(self.maze_model.na):
            q_s_a = 0.
            for next_y in range(self.maze_model.ny):
                for next_x in range(self.maze_model.nx):
                    q_s_a += self.maze_model.T((s[0], s[1]), a, (next_y, next_x)) * (self.maze_model.R(s, a) + self.gamma * self.V[next_y, next_x])
            if (q_s_a > max_value):
                max_value = q_s_a
                amax = a
        return amax
