import numpy as np
from agent import AgentInterface
from environment.maze import Maze
from epsilon_profile import EpsilonProfile


class QAgent(AgentInterface):
    """
    Cette classe d'agent représente un agent utilisant la méthode du Q-learning
    pour mettre à jour sa politique d'action.
    """

    def __init__(self, maze: Maze, eps_profile: EpsilonProfile, gamma: float, alpha: float):
        """A LIRE
        Ce constructeur initialise une nouvelle instance de la classe QAgent.
        Il doit stocker les différents paramètres nécessaires au fonctionnement de l'algorithme et initialiser la
        fonction de valeur d'action, notée Q.

        :param maze: Le labyrinthe à résoudre
        :type maze: Maze

        :param eps_profile: Le profil du paramètre d'exploration epsilon
        :type eps_profile: EpsilonProfile

        :param gamma: Le discount factor
        :type gamma: float

        :param alpha: Le learning rate
        :type alpha: float
        """
        # Initialise la fonction de valeur Q
        self.Q = np.zeros([maze.ny, maze.nx, maze.na])

        self.maze = maze
        self.na = len(maze.action_space)

        # Paramètres de l'algorithme
        self.gamma = gamma
        self.alpha = alpha

        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial

    def learn(self, env, n_episodes, max_steps):
        """Cette méthode exécute l'algorithme de q-learning.
        Il n'y a pas besoin de la modifier. Simplement la comprendre et faire le parallèle avec le cours.

        :param env: L'environnement
        :type env: gym.Envselect_action
        :param num_episodes: Le nombre d'épisode
        :type num_episodes: int
        :param max_num_steps: Le nombre maximum d'étape par épisode
        :type max_num_steps: int
        """
        n_steps = np.zeros(n_episodes) + max_steps

        # Execute N episodes
        for episode in range(n_episodes):
            # Reinitialise l'environnement
            state = env.reset_using_existing_maze()
            # Execute K steps
            for step in range(max_steps):
                # Selectionne une action
                action = self.select_action(state)
                # Echantillonne l'état suivant et la récompense
                next_state, reward, terminal = env.step(action)

                # Mets à jour la fonction de valeur Q
                self.updateQ(state, action, reward, next_state)

                if terminal:
                    n_steps[episode] = step + 1
                    break

                state = next_state
            # Mets à jour la valeur du epsilon
            self.epsilon = max(
                self.epsilon - self.eps_profile.dec_episode / (n_episodes - 1.0), self.eps_profile.final
            )

            # Sauvegarde et affiche les données d'apprentissage
            if n_episodes >= 0:
                state = env.reset_using_existing_maze()

                print(
                    f"\r#> Ep. {episode}/{n_episodes} Value {self.Q[state][self.select_greedy_action(state)]}", end=""
                )

    def updateQ(self, state: tuple[int, int], action: int, reward: float, next_state: tuple[int, int]) -> None:
        """À COMPLÉTER!
        Cette méthode utilise une transition pour mettre à jour la fonction de valeur Q de l'agent.
        Une transition est définie comme un tuple (état, action récompense, état suivant).

        :param state: L'état origine
        :param action: L'action
        :param reward: La récompense perçue
        :param next_state: L'état suivant
        """
        raise NotImplementedError("Q-learning NotImplementedError at function 'updateQ'.")

    def select_action(self, state: tuple[int, int]) -> int:
        """À COMPLÉTER!
        Cette méthode retourne une action échantilloner selon le processus d'exploration (ici epsilon-greedy).

        :param state: L'état courant
        :return: L'action
        """
        raise NotImplementedError("Q-learning NotImplementedError at function 'select_action'.")

    def select_greedy_action(self, state: tuple[int, int]) -> int:
        """
        Cette méthode retourne l'action gourmande.

        :param state: L'état courant
        :return: L'action gourmande
        """
        raise NotImplementedError("Q-learning NotImplementedError at function 'select_greedy_action'.")
