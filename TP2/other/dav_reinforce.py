from multiprocessing.connection import wait
import gym
import numpy as np
import time

from world.maze import Maze

SIZE_MAZE = 8


# test once by taking greedy actions based on Q values
def Q_test_maze(env, Q, max_steps, speed = 0., display = False):
    n_steps = max_steps
    sum_rewards = 0.
    s = env.reset_using_existing_maze()
    if display:
        env.render()

    for j in range(max_steps):
        mx = np.max(Q[s])
        a = np.random.choice(np.where(Q[s]==mx)[0])     # greedy action with random tie break
        sn, r, terminal = env.step(a)
        if display:
            time.sleep(speed)
            env.render()

        sum_rewards += r
        if terminal:
            n_steps = j+1  # number of steps taken
            break
        s = sn
    return n_steps, sum_rewards

class epsilon_profile:
    def __init__(self, initial = 1., final = 0., dec_episode = 1., dec_step = 0.):
        self.initial = initial          # initial epsilon in epsilon-greedy
        self.final = final              # final epsilon in epsilon-greedy
        self.dec_episode = dec_episode  # amount of decrement of epsilon in each episode is dec_episode / (number of episodes - 1)
        self.dec_step = dec_step        # amount of decrement of epsilon in each step
        
def epsilon_greedy_action(env, Q, state, current_epsilon):
    if np.random.rand() < current_epsilon:
        a = np.random.randint(env.na)      # random action
    else:
        mx = np.max(Q[state])
        a = np.random.choice(np.where(Q[state]==mx)[0])     # greedy action with random tie break
    return a


def q_learning_target(env, Q, alpha, gamma, state, action, reward, next_state):
    return (1.-alpha) * Q[state[0],state[1],action] + alpha * (reward + gamma * np.max(Q[next_state]))

def sarsa_target(env, Q, alpha, gamma, state, action, reward, next_state, epsilon):
    return (1.-alpha) * Q[state[0],state[1],action] + alpha * (reward + gamma * epsilon_greedy_action(env, Q, next_state, epsilon))

def main():

    # env = gym.make('Copy-v0')
    # env = gym.make('CartPole-v0')
    env = Maze(9, 9, 14)
    # env.render('maze11x11.png')     # use this to save the maze figure as a file

    print(env.maze)
    print('length of shortest path:', env.shortest_length)
    print('starting point:', env.init_state)
    print('goal:', env.terminal_state)

    n_episodes = 300
    max_steps = 500
    alpha = 0.2
    gamma = 1.0

    Q = np.zeros([env.ny, env.nx, env.na])

    n_steps = np.zeros(n_episodes) + max_steps
    sum_rewards = np.zeros(n_episodes)  # total reward for each episode
    
    eps_profile = epsilon_profile(1., 1., 0., 0.)

    test_n_steps = np.zeros(n_episodes) + max_steps
    test_sum_rewards = np.zeros(n_episodes)  # total reward for each episode
    
    epsilon = eps_profile.initial

    for episode in range(n_episodes):
        state = env.reset_using_existing_maze()
        for step in range(max_steps):
            action = epsilon_greedy_action(env, Q, state, epsilon)
            next_state, reward, terminal = env.step(action)
            sum_rewards[episode] += reward

            Pi[state[0], state[1], action] = reinforce_target(env, Q, alpha, gamma, state, action, reward, next_state)
            
            if terminal:
                n_steps[episode] = step+1  # number of steps taken
                break

            state = next_state
            epsilon = max(epsilon - eps_profile.dec_step, eps_profile.final)
        if n_episodes > 0:
            epsilon = max(epsilon - eps_profile.dec_episode / (n_episodes - 1.), eps_profile.final)

        test_n_steps[episode], test_sum_rewards[episode] = Q_test_maze(env, Q, max_steps)
    
    print(Q)
    Q_test_maze(env, Q, max_steps, speed=0.1, display=True)

if __name__ == '__main__':
    main()
