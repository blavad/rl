import numpy as np
from copy import deepcopy

from agent import AgentInterface
from world.maze import Maze


class VIAgent(AgentInterface):
    """
    Un agent capable de résoudre un labyrinthe donné grâce à l'algorithme d'itération
    sur les valeurs (VI = Value Iteration).
    """

    def __init__(self, maze: Maze, gamma: float):
        """ "À LIRE
        Ce constructeur initialise une nouvelle instance de la classe ValueIteration.
        Il stocke les différents paramètres nécessaires au fonctionnement de l'algorithme et initialise à 0 la
        fonction de valeur d'état, notée V.

        :attribut V: La fonction de valeur d'états
        :type V: un tableau de dimension : ny x nx

        :attribut maze: Le modèle du labyrinthe. Il permet de récupérer la fonction de transition (maze.dynamics) et la récompense (maze.reward)
        :type maze: DeterministicMazeModel

        :attribut gamma: le facteur d'atténuation
        :type gamma: float
        :requirement: 0 <= gamma <= 1

        - Visualisation des données
        :attribut mazeValues: la fonction de valeur stockée qui sera écrite dans un fichier de log après la résolution complète
        :type mazeValues: data frame pandas
        :penser à bien stocker aussi la taille du labyrinthe (nx,ny)
        """
        self.gamma = gamma
        self.maze = maze
        self.V = np.zeros([maze.nx, maze.ny])

    def solve(self, error: float):
        """
        Cette méthode résoud le problème avec une tolérance donnée.
        Elle doit proposer l'option de stockage de la fonction de valeur dans un fichier de log (logV.csv)
        """
        n_iteration = 0
        V_copy = np.zeros([self.maze.nx, self.maze.ny])
        while (n_iteration == 0) or not self.done(self.V, V_copy, error):
            n_iteration += 1
            self.V = deepcopy(V_copy)
            for state in self.maze.state_space:
                if not self.maze.maze[state]:
                    V_copy[state] = self.bellman_operator(state)

    def done(self, V, V_copy, error) -> bool:
        """À COMPLÉTER!
        Cette méthode retourne vraie si la condition d'arrêt de
        l'algorithme est vérifiée. Sinon elle retourne faux.
        Pour garantie la convergence en tout état, il est préférable
        d'utiliser la norme infini comme critère d'arrêt.
        """
        raise NotImplementedError("VI NotImplementedError at function 'done'.")

    def bellman_operator(self, state: tuple[int, int]) -> float:
        """À COMPLÉTER!
        Cette méthode calcul l'opérateur de mise à jour de bellman pour un état s.

        :param state: Un état quelconque
        :return: La valeur de mise à jour de la fonction de valeur
        """
        # Retourne une exception si l'état n'est pas valide
        # Compléter ici votre équation de Bellman
        # Note: On utilisera la fonction de récompense (self.maze.get_reward) et la fonction de transition (self.maze.get_dynamics).
        max_value = -np.infty
        raise NotImplementedError("Value Iteration NotImplementedError at function 'bellman_operator'.")

    def select_action(self, state: tuple[int, int]) -> int:
        """À COMPLÉTER!
        Cette méthode retourne l'action optimale.

        :param state: L'état courant
        :return: L'action optimale

        doit retourner une exception si l'état n'est pas valide
        """
        # Compléter ici votre équation de Bellman
        # Note: On utilisera la fonction de récompense (self.maze.get_reward) et la fonction de transition (self.maze.get_dynamics).
        max_value = -np.infty
        raise NotImplementedError("Value Iteration NotImplementedError at Function select_action")
