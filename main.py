import imp
import sys
import time
import argparse
import torch

from agent.qagent import QAgent
from agent.dqn_agent import DQNAgent
from agent.viagent import VIAgent
from agent.random_agent import RandomAgent
from epsilon_profile import EpsilonProfile
from networks import MLP, CNN
from logAnalysis import *
from world.maze import Maze
from world.deterministic_maze import DeterministicMazeModel
from logAnalysis import logAnalysis

# parser = argparse.ArgumentParser(description='Maze parameters')
# parser.add_argument('--algo', type=str, default="random", metavar='a', help='algorithm to use (default: 7)')
# parser.add_argument('--width', type=int, default=7, metavar='w', help='width of the maze (default: 7)')
# parser.add_argument('--height', type=int, default=7, metavar='h', help='height of the maze (default: 7)')
# parser.add_argument('--shortest_path', type=int, default=14, metavar='p', help='shortest distance between starting point and goal point (default: 14)')
# args = parser.parse_args()

# test once by taking greedy actions based on Q values
def test_maze(env: Maze, agent: DQNAgent, max_steps: int, nepisodes : int = 1, speed: float = 0., same = True, display: bool = False):
    n_steps = max_steps
    sum_rewards = 0.
    for _ in range(nepisodes):
        state = env.reset_using_existing_maze() if (same) else env.reset()
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
# 
    # env = Maze(5, 5, min_shortest_length=0) 
    # env = Maze(7, 7, min_shortest_length=15) 
    # env = Maze(9, 9, min_shortest_length=20) # Create a 9x9 maze
    # env = Maze(14, 14, min_shortest_length=40) # Create a 15x15 maze
    env = DeterministicMazeModel.from_file("tests/maze_ex2.txt") # Create a deterministic maze model
    # env = DeterministicMazeModel(15, 15, min_shortest_length=30) # Create a deterministic maze model
        
    # WARNING : Pour Aurélien et Jilles : ces paramètres sont pour DQN (à changer pour VI et Q-learning tabulaire)  
    n_episodes = 200
    max_steps = 50
    gamma = 1.
    alpha = 0.2
    eps_profile = EpsilonProfile(1.0, 0.1)

    # Hyperparamètres de DQN
    final_exploration_episode = 500
    batch_size = 32
    replay_memory_size = 1000
    target_update_frequency = 100
    tau = 1.0

    if (agent == "random"):
        agent = RandomAgent(env.action_space.n)
        test_maze(env, agent, max_steps, speed=0.1, display=True)
    elif (agent == "vi"):
        agent = VIAgent(env, gamma)
        agent.solve(0.01)
        test_maze(env, agent, max_steps, speed=0.1, display=True)
    elif (agent == "qlearning"):
        agent = QAgent(env, eps_profile, gamma, alpha)
        agent.learn(env, n_episodes, max_steps)
        test_maze(env, agent, max_steps, speed=0.1, display=True)
    elif (agent == "dqn"):
        env.mode = "nn" # active le mode DeepRL (l'observation est la grille directement)
        # A COMPLETER 
        # nn = MLP(env.ny, env.nx, env.nf, env.na) 
        nn = CNN(env.ny, env.nx, env.nf, env.na) 
        agent = DQNAgent(nn, eps_profile, gamma, alpha, replay_memory_size, batch_size, target_update_frequency, tau, final_exploration_episode)
        agent.learn(env, n_episodes, max_steps)
        test_maze(env, agent, max_steps, 10, speed=0.1, display=True, same=False)
    elif (agent=="logAnalysis"):
        agent = logAnalysis(opt)
        agent.printCurves()
        return
    else:
        print("Error : Unknown agent name (" + agent + ").")

if __name__ == '__main__':
    """ Usage : python main.py [ARGS]
    - First argument (str) : the name of the agent (i.e. 'random', 'vi', 'qlearning', 'dqn')
    - Second argument (int) : the maze hight
    - Third argument (int) : the maze width
    """
    if (len(sys.argv) > 2):
        main(sys.argv[1], sys.argv[2:])
    if (len(sys.argv) > 1):
        main(sys.argv[1], [])
    else:
        main("random", [])
