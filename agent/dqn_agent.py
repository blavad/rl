import copy
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

from agent.qagent import QAgent
from world.maze import Maze
from epsilon_profile import EpsilonProfile


class DQNAgent(QAgent):
    """ 
    Cette classe d'agent représente un agent utilisant la méthode du Q-learning 
    pour mettre à jour sa politique d'action.
    """

    def __init__(self, qnetwork: nn.Module, eps_profile: EpsilonProfile, gamma: float, alpha: float, replay_memory_size: int = 1000, batch_size: int = 32, target_update_freq: int = 100, tau: float = 1., final_exploration_episode : int = 500):
        """
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
        :param tau: Le taux de mise à jour du réseau cible
        :type tau: float
        """
        self.policy_net = qnetwork
        self.target_net = copy.deepcopy(qnetwork)

        # Paramètres d'apprentissage
        self.alpha = alpha
        self.gamma = gamma
        self.replay_memory_size = replay_memory_size
        self.minibatch_size = batch_size
        self.target_update_frequency = target_update_freq
        self.tau = tau

        # Profil du Epsilon
        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial
        self.init_epsilon = self.eps_profile.initial
        self.final_epsilon = self.eps_profile.final
        self.final_exploration_episode = final_exploration_episode

        # Cirtère d'optimisation
        self.criterion = nn.MSELoss()

        # Méthode de descente de gradient
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)

    def init_replay_memory(self, env: Maze):
        """Cette méthode initialise le buffer d'expérience replay.

        :param env: Environnement (le labyrinthe)
        :type env: Maze
        """
        # Replay memory pour s, a, r, terminal, and sn
        self.Ds = np.zeros([self.replay_memory_size, env.nf, env.ny, env.nx], dtype=np.float32)
        self.Da = np.zeros([self.replay_memory_size, env.na], dtype=np.float32)
        self.Dr = np.zeros([self.replay_memory_size], dtype=np.float32)
        self.Dt = np.zeros([self.replay_memory_size], dtype=np.float32)
        self.Dsn = np.zeros([self.replay_memory_size, env.nf, env.ny, env.nx], dtype=np.float32)

        self.d = 0     # counter for storing in D
        self.ds = 0    # total number of steps

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
        self.na = env.action_space.n
        self.init_replay_memory(env)

        # Initialisation des stats d'apprentissage
        sum_rewards = np.zeros(n_episodes)
        len_episode = np.zeros(n_episodes)
        n_steps = np.zeros(n_episodes) + max_steps

        start_time = time.time()

        # Execute N episodes
        for episode in range(n_episodes):
            # Reinitialise l'environnement
            state = env.reset()
            # Execute K steps
            for step in range(max_steps):
                
                # env.render()
                # Selectionne une action
                action = self.select_action(state)

                # Echantillonne l'état suivant et la récompense
                next_state, reward, terminal = env.step(action)

                # Stocke les données d'apprentissage
                sum_rewards[episode] += reward
                len_episode[episode] += 1

                # Mets à jour la fonction de valeur Q
                self.updateQ(state, action, reward, next_state, terminal)

                if terminal:
                    n_steps[episode] = step + 1  # number of steps taken
                    break

                state = next_state

            self.epsilon = max(self.final_epsilon, self.epsilon - (1. / self.final_exploration_episode))
            # self.epsilon = max(self.epsilon - self.eps_profile.dec_step, self.eps_profile.final)

            # Mets à jour le réseau cible, en copiant tous les weights et biases dans DQN
            if n_episodes % self.target_update_frequency == 0:
                if (self.tau < 1.):
                    # Mets à jour le réseau de neurones cible en lissant ses paramètres avec ceux de policy_net
                    self.soft_update(self.tau)
                else:
                    # Copie le réseau de neurones courant dans le réseau cible
                    self.hard_update()

            n_ckpt = 10
            n_test_period = 100
            if episode % n_test_period == n_test_period - 1:   # for testing
                test_score, test_extra_steps = self.run_tests(
                    env, 100, max_steps)
                # print test score and # of time steps to achieve the score
                state = env.reset()
                print('episode: %4d, frames: %6d, train score: %.1f, mean steps: %.1f, test score: %.1f, test extra steps: %.1f, test success ratio: %.2f, epsilon: %.2f, elapsed time: %.1f'
                      % (episode + 1, self.ds, np.mean(sum_rewards[episode-(n_ckpt-1):episode+1]), np.mean(len_episode[episode-(n_ckpt-1):episode+1]), test_score, np.mean(test_extra_steps), np.sum(test_extra_steps == 0) / 100, self.epsilon, time.time() - start_time))

        n_test_runs = 100
        test_score, test_extra_steps = self.run_tests(
            env, n_test_runs, max_steps)
        for k in range(n_test_runs):
            print(test_extra_steps[k])
        print('Final test score: %.1f' % test_score)
        print('Final test success ratio: %.2f' %
              (np.sum(test_extra_steps == 0) / n_test_runs))

    def updateQ(self, state, action, reward, next_state, terminal):
        """À COMPLÉTER!
        Cette méthode utilise une transition pour mettre à jour la fonction de valeur Q de l'agent. 
        Une transition est définie comme un tuple (état, action récompense, état suivant).

        :param state: L'état origine
        :param action: L'action
        :param reward: La récompense perçue
        :param next_state: L'état suivant
        """
        # Ajoute les éléments dans le buffer d'expérience
        self.Ds[self.d], self.Dr[self.d], self.Dsn[self.d], self.Dt[self.d] = state, reward, next_state, terminal

        # since Da[d,:] is a one-hot vector
        self.Da[self.d] = 0
        self.Da[self.d, action] = 1

        # since D is a circular buffer
        self.d = (self.d + 1) % self.replay_memory_size
        self.ds = self.ds + 1

        # Commencer l'apprentissage quand le buffer est plein
        if self.ds >= self.replay_memory_size:

            self.optimizer.zero_grad()

            # Sélectionne des indices aléatoires dans le buffer
            c = np.random.choice(self.replay_memory_size, self.minibatch_size)

            # Récupère les batch de données associés
            x_batch, a_batch, r_batch, y_batch, t_batch = torch.from_numpy(self.Ds[c]), torch.from_numpy(
                self.Da[c]), torch.from_numpy(self.Dr[c]),  torch.from_numpy(self.Dsn[c]), torch.from_numpy(self.Dt[c])
            # Calcul de la valeur courante
            current_value = self.policy_net(x_batch).gather(1, a_batch.max(1).indices.unsqueeze(1)).squeeze(1)
            # Calcul de la valeur cible
            target_value = self.target_net(y_batch).max(1).values * self.gamma * (1. - t_batch) + r_batch

            # La fonction 'detach' stoppe la rétropopagation du gradient à 
            # travers une partie du graphe (ici target network)
            loss = self.criterion(current_value, target_value.detach())

            loss.backward()
            self.optimizer.step()

    def select_greedy_action(self, state):
        """
        Cette méthode retourne l'action gourmande.

        :param state: L'état courant
        :return: L'action gourmande
        """
        return self.policy_net(torch.FloatTensor(state).unsqueeze(0)).argmax()

    def hard_update(self):
        """ Cette fonction copie le réseau de neurones 
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update(self, tau):
        """ Cette fonction fait mise à jour glissante du réseau de neurones cible 
        """
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau) * target_param.data)

    def run_tests(self, env, n_runs, max_steps, printenv=False):
        test_score = 0.
        extra_steps = np.zeros((n_runs, 2))
        for k in range(n_runs):
            s = env.reset()
            for t in range(max_steps):
                q = self.policy_net(torch.FloatTensor(s).unsqueeze(0))
                # greedy action with random tie break
                a = np.random.choice(np.where(q[0] == q[0].max())[0])
                sn, r, terminal = env.step(a)
                test_score += r
                if terminal:
                    break
                s = sn
            extra_steps[k] = t + 1 - env.shortest_length, env.shortest_length
        order = extra_steps[:, 0].argsort()
        extra_steps = extra_steps[order]
        return test_score / n_runs, extra_steps
