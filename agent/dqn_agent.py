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

    def __init__(self, qnetwork: nn.Module, eps_profile: EpsilonProfile, gamma: float, alpha: float, tau : float = 1.):
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
        self.QNet = qnetwork
        self.targetQNet = copy.deepcopy(qnetwork)

        # Algorithm parameters
        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial

        self.gamma = gamma
        self.alpha = alpha

        self.criterion = nn.MSELoss() # Huber criterion
        self.optimizer = optim.Adam(self.QNet.parameters(), lr=self.alpha)

        self.na = None

        self.replay_memory_size = 1000
        self.tau = tau # 
        self.target_update_frequency = 100
        self.minibatch_size = 32

        self.init_epsilon = 1.
        self.final_epsilon = 0.1
        self.final_exploration_episode = 500

    def init_replay_memory(self, env: Maze):
        # replay memory for s, a, r, terminal, and sn
        self.Ds = np.zeros([self.replay_memory_size, env.nf, env.ny, env.nx], dtype=np.float32)
        self.Da = np.zeros([self.replay_memory_size, env.na], dtype=np.float32)
        self.Dr = np.zeros([self.replay_memory_size], dtype=np.float32)
        self.Dt = np.zeros([self.replay_memory_size], dtype=np.float32)    # 1 if terminal
        self.Dsn = np.zeros([self.replay_memory_size, env.nf, env.ny, env.nx], dtype=np.float32)

        self.d = 0     # counter for storing in D
        self.ds = 0    # total number of steps
        

    # runs tests by taking greedy actions based on deep Q-network
    def run_tests(self, env, n_runs, max_steps, printenv=False):
        test_score = 0.
        extra_steps = np.zeros((n_runs, 2))
        for k in range(n_runs):
            s = env.reset()
            for t in range(max_steps):     
                q = self.QNet(torch.FloatTensor(s).unsqueeze(0))
                a = np.random.choice(np.where(q[0]==q[0].max())[0])               # greedy action with random tie break
                sn, r, terminal = env.step(a)
                test_score += r
                if terminal:
                    break
                s = sn
            extra_steps[k] = t + 1 - env.shortest_length, env.shortest_length
        order = extra_steps[:,0].argsort()
        extra_steps = extra_steps[order]
        return test_score / n_runs, extra_steps
       
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
        n_steps = np.zeros(n_episodes) + max_steps
        sum_rewards = np.zeros(n_episodes)  # total reward for each episode
        len_episode = np.zeros(n_episodes)  # total reward for each episode

        self.init_replay_memory(env)

        start_time = time.time()

        # Compute N episodes
        for episode in range(n_episodes):
            # Reinitialise l'environnement
            state = env.reset()
            # Compute K steps
            for step in range(max_steps):

                # Selectionne une action
                action = self.select_action(state)

                # Echantillonne l'état suivant et la récompense
                next_state, reward, terminal = env.step(action)

                sum_rewards[episode] += reward
                len_episode[episode] += 1

                # Mets à jour la fonction de valeur Q
                self.updateQ(state, action, reward, next_state, terminal)
        
                if terminal:
                    n_steps[episode] = step+1  # number of steps taken
                    break

                state = next_state
            
            self.epsilon = max(self.final_epsilon, self.epsilon - 1. / self.final_exploration_episode) 
            # self.epsilon = max(self.epsilon - self.eps_profile.dec_step, self.eps_profile.final)

            # Update the target network, copying all weights and biases in DQN
            if n_episodes % self.target_update_frequency == 0:
                if (self.tau < 1.):
                    self.soft_update(self.tau) # Mets à jour le réseau de neurones cible en lissant ses paramètres avec ceux du QNet
                else:
                    self.hard_update() # Copie le réseau de neurones courant dans le réseau cible

            n_ckpt = 10
            n_test_period = 100
            if episode % n_test_period == n_test_period - 1:   # for testing
                test_score, test_extra_steps = self.run_tests(env, 100, max_steps)
                # print test score and # of time steps to achieve the score
                print('episode: %4d, frames: %6d, train score: %.1f, mean steps: %.1f, test score: %.1f, test extra steps: %.1f, test success ratio: %.2f, elapsed time: %.1f'
                % (episode + 1, self.ds, np.mean(sum_rewards[episode-(n_ckpt-1):episode+1]), np.mean(len_episode[episode-(n_ckpt-1):episode+1]), test_score, np.mean(test_extra_steps), np.sum(test_extra_steps==0) / 100, time.time() - start_time))

        n_test_runs = 100
        test_score, test_extra_steps = self.run_tests(env, n_test_runs, max_steps)
        for k in range(n_test_runs):
            print(test_extra_steps[k])     # prints out extra # of steps taken & the shortest path length for each episode
        print('Final test score: %.1f' % test_score)
        print('Final test success ratio: %.2f' % (np.sum(test_extra_steps==0) / n_test_runs))

    def hard_update(self):
        self.targetQNet.load_state_dict(self.QNet.state_dict())

    def soft_update(self, tau):
        for target_param, local_param in zip(self.targetQNet.parameters(), self.QNet.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def updateQ(self, state, action, reward, next_state, terminal):
        """À COMPLÉTER!
        Cette méthode utilise une transition pour mettre à jour la fonction de valeur Q de l'agent. 
        Une transition est définie comme un tuple (état, action récompense, état suivant).

        :param state: L'état origine
        :param action: L'action
        :param reward: La récompense perçue
        :param next_state: L'état suivant
        """
        # 
        # Store in memory buffer
        self.Ds[self.d], self.Dr[self.d], self.Dsn[self.d], self.Dt[self.d] = state, reward, next_state, terminal # append to replay memory
        self.Da[self.d] = 0; self.Da[self.d, action] = 1     # since Da[d,:] is a one-hot vector
        self.d = (self.d + 1) % self.replay_memory_size      # since D is a circular buffer
        self.ds = self.ds + 1
        
        if self.ds >= self.replay_memory_size:    # starts training once D is full
            
            self.optimizer.zero_grad()
            
            c = np.random.choice(self.replay_memory_size, self.minibatch_size)
            
            x_batch, a_batch, r_batch, y_batch, t_batch = torch.as_tensor(self.Ds[c]), torch.as_tensor(self.Da[c]), torch.as_tensor(self.Dr[c]),  torch.as_tensor(self.Dsn[c]), torch.as_tensor(self.Dt[c])
            current_value = self.QNet(x_batch).gather(1, a_batch.max(1).indices.unsqueeze(1)).squeeze(1)
            next_value = self.targetQNet(y_batch)
            target_value = (next_value.max(1).values * self.gamma * (1. - t_batch) + r_batch)

            loss = self.criterion(current_value, target_value.detach())
            loss.backward()
            self.optimizer.step()

    def select_action(self, state):
        """À COMPLÉTER!
        Cette méthode retourne une action échantilloner selon le processus d'exploration (ici epsilon-greedy).

        :param state: L'état courant
        :return: L'action 
        """
        if np.random.rand() < self.epsilon:
            a = np.random.randint(self.na)      # random action
        else:
            a = self.select_greedy_action(state)
        return a

    def select_greedy_action(self, state):
        """À COMPLÉTER!
        Cette méthode retourne l'action gourmande.

        :param state: L'état courant
        :return: L'action gourmande
        """
        return self.QNet(torch.FloatTensor(state).unsqueeze(0)).argmax()