import sys
import time
import gym
import numpy as np

from agent import AgentInterface
from agent.qagent import QAgent
from agent.viagent import VIAgent
from agent.random_agent import RandomAgent
from epsilon_profile import EpsilonProfile
from world.maze import Maze
from world.deterministic_maze import DeterministicMazeModel

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


def main(agent, opt):

    env = Maze(7, 7, 14)
    # env = DeterministicMazeModel(15, 15, 30)
        
    n_episodes = 30
    max_steps = 500
    alpha = 0.2
    gamma = 1.0
    eps_profile = EpsilonProfile(1., 1., 0., 0.)

    print(env.maze)
    print('num_actions:', env.action_space.n)
    print('length of shortest path:', env.shortest_length)
    print('starting point:', env.init_state)
    print('goal:', env.terminal_state)

    if (agent == "random"):
        agent = RandomAgent(env.action_space.n)
    elif (agent == "vi"):
        agent = VIAgent(env, gamma)
        agent.solve(0.01)
    elif (agent == "qlearning"):
        agent = QAgent(env, eps_profile, gamma, alpha)
        agent.learn(env, n_episodes, max_steps)
    else:
        print("Error : Unknown agent name (" + agent + ").")
    
    test_maze(env, agent, max_steps, speed=0.1, display=True)


if __name__ == '__main__':
    if (len(sys.argv) > 2):
        main(sys.argv[1], sys.argv[2:])
    if (len(sys.argv) > 1):
        main(sys.argv[1], [])
    else:
        main("random", [])
