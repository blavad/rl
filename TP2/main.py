import sys
import time

from agent.dqn_agent import DQNAgent
from epsilon_profile import EpsilonProfile
from world.maze import Maze

from networks import MLP, CNN

# test once by taking greedy actions based on Q values
def test_maze(env: Maze, agent: DQNAgent, max_steps: int, nepisodes : int = 1, speed: float = 0., same = True, display: bool = False):
    n_steps = max_steps
    sum_rewards = 0.
    for _ in range(nepisodes):
        state = env.reset() if (same) else env.reset()
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
 
    """ INSTANCIE LE LABYRINTHE """ 
    env = Maze(5, 5, min_shortest_length=0) 
    env.mode = "nn" 
    nn = None

    """ INITIALISE LES PARAMETRES D'APPRENTISSAGE """
    # Hyperparamètres basiques
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

    """ INSTANCIE LE RESEAU DE NEURONES """
    if (nn == "mlp"):
        nn = MLP(env.ny, env.nx, env.nf, env.na)
    elif (nn == "cnn"):
        nn = CNN(env.ny, env.nx, env.nf, env.na)
    else:
        print("Error : Unknown neural network (" + nn + ").")
    

    print('--- neural network ---')
    num_params = sum(param.numel() for param in nn.parameters() if param.requires_grad)
    print('number of parameters:', num_params)
    print(nn)

    """  LEARNING PARAMETERS"""
    agent = DQNAgent(nn, eps_profile, gamma, alpha, replay_memory_size, batch_size, target_update_frequency, tau, final_exploration_episode)
    agent.learn(env, n_episodes, max_steps)
    test_maze(env, agent, max_steps, speed=0.1, display=False)

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
