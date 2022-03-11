# Good parameters by exercice

## TP 1

### Q-Learning
1. Maze 7x7
- n_episodes = 200
- max_steps = 50
- gamma = 1.
- alpha = 0.2

2. Maze 14x14
- n_episodes = 1000
- max_steps = 200
- gamma = 1.
- alpha = 0.2


## TP 2

### Partie 1 : MLP

**DQN Maze 5x5:** 
*Params:*
- n_episodes = 2000
- max_steps = 50
- gamma = 1.
- alpha = 0.001

*Archi:*

```python
self.flatten = nn.Flatten()
self.layers = nn.Sequential(
    nn.Linear(ny*nx*nf, 64),
    nn.ReLU(),
    nn.Linear(64, na),
)
```


### Partie 2 : CNN

**DQN Maze 5x5:**

*Params:*
- n_episodes = 2000
- max_steps = 50
- gamma = 1.
- alpha = 0.001
- eps_profile = EpsilonProfile(1.0, 0.1, 1., 0.)
- final_exploration_episode = 1000
- batch_size = 32
- replay_memory_size = 1000
- target_update_frequency = 100
- tau = 1.0

*Archi:*

```python
self.layers = nn.Sequential(
    nn.Conv2d(nf, 32, 3, stride=1, padding="same", bias=True),
    nn.ReLU(),
    nn.Conv2d(32, 32, 3, stride=1, padding="same", bias=True),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(ny * nx * 32, 120),
    nn.Linear(120, na)
)
```


**DQN Maze 7x7: (pas top et assez long)**

*Params:*
- n_episodes = 15000
- max_steps = 80
- gamma = 1.
- alpha = 0.00025
- eps_profile = EpsilonProfile(1.0, 0.1, 1., 0.)
- final_exploration_episode = 10000
- batch_size = 64
- replay_memory_size = 10000
- target_update_frequency = 1000
- tau = 1.0

*Archi:*

```python
self.layers = nn.Sequential(
        nn.Conv2d(nf, 32, 4, stride=1, padding="same", bias=True),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding="same", bias=True),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(ny * nx * 64, 200),
        nn.Linear(200, na)
)
```