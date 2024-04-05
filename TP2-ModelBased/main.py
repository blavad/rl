import sys
import time
from agent import AgentInterface

from agent.viagent import VIAgent
from agent.random_agent import RandomAgent

from environment.maze import Maze


# test once by taking greedy actions based on Q values
def test_maze(
    env: Maze,
    agent: AgentInterface,
    max_steps: int,
    nepisodes: int = 1,
    speed: float = 0.0,
    same=True,
    display: bool = False,
):
    n_steps = max_steps
    sum_rewards = 0.0
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
                n_steps = step + 1  # number of steps taken
                break
            state = next_state
    return n_steps, sum_rewards


def main(agent, opt):
    # env = Maze(5, 5, min_shortest_length=0)
    env = Maze(7, 7, min_shortest_length=15)
    # env = Maze(20, 8, min_shortest_length=10)
    # env = Maze(9, 9, min_shortest_length=20)  # Create a 9x9 maze
    env = Maze.from_file("data/maze_ex2.txt")  # Create a maze from a file

    max_steps = 50
    gamma = 1.0

    if agent == "random":
        agent = RandomAgent(env.action_space)
        test_maze(env, agent, max_steps, speed=0.1, display=True)
    elif agent == "vi":
        agent = VIAgent(env, gamma)
        agent.solve(0.01)
        test_maze(env, agent, max_steps, speed=0.1, display=True)
    else:
        print("Error : Unknown agent name (" + agent + ").")


if __name__ == "__main__":
    """Usage : python main.py [ARGS]
    - First argument (str) : the name of the agent (i.e. 'random', 'vi', 'qlearning', 'dqn')
    - Second argument (int) : the maze hight
    - Third argument (int) : the maze width
    """
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2:])
    if len(sys.argv) > 1:
        main(sys.argv[1], [])
    else:
        main("random", [])
