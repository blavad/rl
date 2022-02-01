import numpy as np
import gym

from agent import AgentInterface
from world.maze import Maze
from epsilon_profile import EpsilonProfile


class QAgent(AgentInterface):
    """ 
    Cette classe d'agent représente un agent utilisant la méthode du Q-learning 
    pour mettre à jour sa politique d'action.
    """

    def __init__(self, maze: Maze, eps_profile: EpsilonProfile, gamma: float, alpha: float):
        """À COMPLÉTER!
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
        self.Q = np.zeros([maze.ny, maze.nx, maze.na])

        self.maze = maze

        # Algorithm parameters
        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial

        self.gamma = gamma
        self.alpha = alpha

    def learn(self, env, n_episodes, max_steps):
        """Cette méthode exécute l'algorithme de q-learning. 
        Il n'y a pas besoin de la modifier. Simplement la comprendre et faire le parallèle avec le cours.

        :param env: L'environnement 
        :type env: gym.Env
        :param num_episodes: Le nombre d'épisode
        :type num_episodes: int
        :param max_num_steps: Le nombre maximum d'étape par épisode
        :type max_num_steps: int
        """
        n_steps = np.zeros(n_episodes) + max_steps
        sum_rewards = np.zeros(n_episodes)  # total reward for each episode

        # Compute N episodes 
        for episode in range(n_episodes):
            # Reinitialise l'environnement
            state = env.reset_using_existing_maze()
            # Compute K steps 
            for step in range(max_steps):
                # Selectionne une action 
                action = self.select_action(state)

                # Echantillonne l'état suivant et la récompense
                next_state, reward, terminal = env.step(action)

                # Mets à jour la fonction de valeur Q
                self.updateQ(state, action, reward, next_state)

                sum_rewards[episode] += reward
                
                if terminal:
                    n_steps[episode] = step+1  # number of steps taken
                    break

                state = next_state
                epsilon = max(self.epsilon - self.eps_profile.dec_step, self.eps_profile.final)

            if n_episodes > 0:
                self.epsilon = max(epsilon - self.eps_profile.dec_episode / (n_episodes - 1.), self.eps_profile.final)
                state = env.reset_using_existing_maze()
                print("\r#> Ep. {}/{} Value {}".format(episode, n_episodes, self.Q[state[0],state[1], self.select_greedy_action(state)]), end =" ")

            # test_n_steps[episode], test_sum_rewards[episode] = Q_test_maze(
            #     env, Q, max_steps)

    def updateQ(self, state, action, reward, next_state):
        """À COMPLÉTER!
        Cette méthode utilise une transition pour mettre à jour la fonction de valeur Q de l'agent. 
        Une transition est définie comme un tuple (état, action récompense, état suivant).

        :param state: L'état origine
        :param action: L'action
        :param reward: La récompense perçue
        :param next_state: L'état suivant
        """
        self.Q[state[0], state[1], action] = (1. - self.alpha) * self.Q[state[0], state[1], action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]))

    def select_action(self, state):
        """À COMPLÉTER!
        Cette méthode retourne une action échantilloner selon le processus d'exploration (ici epsilon-greedy).

        :param state: L'état courant
        :return: L'action 
        """
        if np.random.rand() < self.epsilon:
            a = np.random.randint(
                self.maze.action_space.n)      # random action
        else:
            a = self.select_greedy_action(state)
        return a

    def select_greedy_action(self, state):
        """À COMPLÉTER!
        Cette méthode retourne l'action gourmande.

        :param state: L'état courant
        :return: L'action gourmande
        """
        mx = np.max(self.Q[state])
        # greedy action with random tie break
        return np.random.choice(np.where(self.Q[state] == mx)[0])
