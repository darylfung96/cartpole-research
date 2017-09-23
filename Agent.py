class Agent:
    def __init__(self, number_actions, env):
        self.number_actions = number_actions
        self.env = env

    def step(self, state):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
