class EpsilonProfile:
    def __init__(self, initial=1.0, final=0.0, dec_episode=1.0, dec_step=0.0):
        self.initial = initial  # initial epsilon in epsilon-greedy
        self.final = final  # final epsilon in epsilon-greedy
        self.dec_episode = (
            dec_episode  # amount of decrement of epsilon in each episode is dec_episode / (number of episodes - 1)
        )
        self.dec_step = dec_step  # amount of decrement of epsilon in each step
