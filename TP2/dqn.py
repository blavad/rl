import sys
import time
import gym
import numpy as np

from agent.dqn_agent import DQNAgent
from nn.mlp import MLP
from nn.cnn import CNN
from world.maze import Maze
from TP1.epsilon_profile import EpsilonProfile

# test once by taking greedy actions based on Q values
def test_maze(env: Maze, agent: DQNAgent, max_steps: int, speed: float = 0., display: bool = False):
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


def main(nn, opt):

    """ INSTANCIATE MAZE AND LEARNING PARAMETERS """

    env = Maze(5, 5, 5, mode='nn')
    # env = Maze(15, 15, 40)
    # env = DeterministicMazeModel(15, 15, 30)
        
    n_episodes = 100
    max_steps = 100
    alpha = 0.1
    gamma = 1.0
    eps_profile = EpsilonProfile(0.1, 0.1, 1., 0.)

    print('--- maze ---')
    print(env.maze)
    print('num_actions:', env.action_space.n)
    print('length of shortest path:', env.shortest_length)
    print('starting point:', env.init_state)
    print('goal:', env.terminal_state)

    """ INSTANCIATE NEURAL NETWORK """

    if (nn == "mlp"):
        nn = MLP(env.ny, env.nx, env.nf, env.na)
    elif (nn == "cnn"):
        nn = CNN(env.ny, env.nx, env.nf, env.na)
    else:
        print("Error : Unknown Q network name (" + nn + ").")

    print('--- neural network ---')
    num_params = sum(param.numel() for param in nn.parameters() if param.requires_grad)
    print('number of parameters:', num_params)
    print(nn)

    agent = DQNAgent(nn, eps_profile, gamma, alpha)
    agent.learn(env, n_episodes, max_steps)

    # test_maze(env, agent, max_steps, speed=0.1, display=True)


if __name__ == '__main__':
    if (len(sys.argv) > 2):
        main(sys.argv[1], sys.argv[2:])
    if (len(sys.argv) > 1):
        main(sys.argv[1], [])
    else:
        main("mlp", [])
