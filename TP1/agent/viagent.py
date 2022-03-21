import numpy as np
from copy import deepcopy

from agent import AgentInterface
from world.deterministic_maze import DeterministicMazeModel

import pandas as pd


class VIAgent(AgentInterface):
    """ 
    Un agent capable de résoudre un labyrinthe donné grâce à l'algorithme d'itération 
    sur les valeurs (VI = Value Iteration).
    """

    def __init__(self, maze_model: DeterministicMazeModel, gamma: float):
        """"
        Ce constructeur initialise une nouvelle instance de la classe ValueIteration.
        Il stocke les différents paramètres nécessaires au fonctionnement de l'algorithme et initialiser à 0 la 
        fonction de valeur d'état, notée V.

        :param maze_model: Le modèle du problème
        :type maze_model: DeterministicMazeModel

        :param gamma: le facteur d'atténuation
        :type gamma: float
        :requirement: 0<gamma<1

        #visualisation des données
        :attribut mazeValues: la fonction de valeur stockée qui sera écrite dans un fichier de log après la résolution complète
        :type mazeValues: data frame pandas
        :penser à bien stocker aussi la taille du labyrinthe (nx,ny)
        """
        self.gamma = gamma
        self.maze_model = maze_model
        self.V = np.zeros([maze_model.ny, maze_model.nx])
        self.mazeValues = pd.DataFrame(
            data={'nx': maze_model.nx, 'ny': [maze_model.ny]})

    def solve(self, error: float):
        """
        Cette méthode résoud le problème avec une tolérance donnée.
        Elle doit proposer l'option de stockage de la fonction de valeur dans un fichier de log (logVI.csv)
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
            if (True):
                self.mazeValues = self.mazeValues.append({'episode': n_iteration, 'value': np.reshape(
                    self.V, (1, self.maze_model.ny*self.maze_model.nx))[0]}, ignore_index=True)
        self.mazeValues.to_csv('logV.csv')

    def done(self, V, V_copy, error) -> bool:
        """À COMPLÉTER!
        Cette méthode retourne vraie si la condition d'arrêt de 
        l'algorithme est vérifiée. Sinon elle retourne faux.
        """
        raise NotImplementedError("VI NotImplementedError at function done.")

    def bellman_operator(self, s: 'Pair[int, int]') -> float:
        """À COMPLÉTER!
        Cette méthode calcul l'opérateur de mise à jour de bellman pour un état s.

        :param s: Un état quelconque
        :return: La valeur de mise à jour de la fonction de valeur

        doit retourner une exception si l'état n'est pas valide
        """
        if (self.maze_model.maze[s[0], s[1]]):
            raise Exception('this state is a wall, should not be considered')
        max_value = -np.infty
        for a in range(self.maze_model.na):
            q_s_a = 0.
            for next_y in range(self.maze_model.ny):
                for next_x in range(self.maze_model.nx):
                    # Compléter ici votre équation de Bellman
                    raise NotImplementedError("Value Iteration NotImplementedError at Function bellman_operator.")
            if (q_s_a > max_value):
                max_value = q_s_a
        return max_value

    def select_action(self, s):
        """À COMPLÉTER!
        Cette méthode retourne l'action optimale.

        :param state: L'état courant
        :return: L'action optimale

        doit retourner une exception si l'état n'est pas valide
        """
        if (self.maze_model.maze[s[0], s[1]]):
            raise Exception('this state is a wall, should not be considered')
        max_value = -np.infty
        amax = 0
        for a in range(self.maze_model.na):
            q_s_a = 0.
            for next_y in range(self.maze_model.ny):
                for next_x in range(self.maze_model.nx):
                    # Compléter ici votre équation de Bellman
                    raise NotImplementedError("Value Iteration NotImplementedError at Function select_action")
            if (q_s_a > max_value):
                max_value = q_s_a
                amax = a
        return amax
