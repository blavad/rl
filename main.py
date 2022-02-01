import gym
import numpy as np
import time

from agent import AgentInterface
from agent.qagent import QAgent
from agent.viagent import VIAgent
from epsilon_profile import EpsilonProfile
from world.maze import Maze
from world.deterministic_maze import DeterministicMazeModel

SIZE_MAZE = 8

# test once by taking greedy actions based on Q values


def test_maze(env: Maze, agent: QAgent, max_steps: int, speed: float = 0., display: bool = False):
    n_steps = max_steps
    sum_rewards = 0.
    state = env.reset_using_existing_maze()
    if display:
        env.render()

    for step in range(max_steps):
        action = agent.select_greedy_action(state)
        next_state, reward, terminal = env.step(action)

        if display:
            time.sleep(speed)
            env.render()

        sum_rewards += reward
        if terminal:
            n_steps = step+1  # number of steps taken
            break
        state = next_state
    return n_steps, sum_rewards



# def sarsa_target(env, Q, alpha, gamma, state, action, reward, next_state, epsilon):
#     return (1.-alpha) * Q[state[0], state[1], action] + alpha * (reward + gamma * epsilon_greedy_action(env, Q, next_state, epsilon))


def main():

    env = Maze(15, 15, 30)

    print(env.maze)
    print('num_actions:', env.action_space.n)
    print('length of shortest path:', env.shortest_length)
    print('starting point:', env.init_state)
    print('goal:', env.terminal_state)

    n_episodes = 1000
    max_steps = 2000
    alpha = 0.2
    gamma = 1.0
    eps_profile = EpsilonProfile(1., 1., 0., 0.)

    # agent = RandomAgent(env.action_space.n)
    agent = QAgent(env, eps_profile, gamma, alpha)
    agent.learn(env, n_episodes, max_steps)

    test_maze(env, agent, max_steps, speed=0.1, display=True)


    # n_episodes = 30
    # max_steps = 500
    # gamma = 1.0

    # env = DeterministicMazeModel(15, 15, 30)

    # print(env.maze)
    # print('num_actions:', env.action_space.n)
    # print('length of shortest path:', env.shortest_length)
    # print('starting point:', env.init_state)
    # print('goal:', env.terminal_state)

    # viagent = VIAgent(env, gamma)
    # viagent.solve(0.01)
    # print(viagent.V)
    # test_maze(env, viagent, max_steps, speed=0.1, display=True)


    # randagent = RandomAgent(env.action_space.n)
    # test_maze(env, randagent, max_steps, speed=0.1, display=True)


    # env = DeterministicMazeModel()


if __name__ == '__main__':
    main()
