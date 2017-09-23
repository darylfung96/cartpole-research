from Agent import Agent
import numpy as np
import random
import math

explore_rate = 0.5
DECAY_RATE = math.log(3) * 0.01

class QtableAgent(Agent):
    def __init__(self, number_actions, env, bin_size, explore_rate):
        super().__init__(number_actions, env)
        self.bin_size = bin_size
        self.explore_rate = explore_rate
        # get the number of states
        number_state = env.observation_space.shape[0]
        # get the min and max value for each states
        low_high_state = list(zip(env.observation_space.low, env.observation_space.high))
        self.state_bins = [self._make_state_bins(v[0], v[1]) for v in low_high_state]

        # convert to q table
        self.q_table = np.zeros((self.bin_size**number_state, env.action_space.n))

    def step(self, state):
        rand = random.random()

        if rand < self.explore_rate:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])

        observation, reward, done, _ = self.env.step(action)

        self.explore_rate -= DECAY_RATE

        return observation, reward, done

    def train(self):
        pass

    # the state_index is the index for the state,
    # cartPole has 4 states, state[4], state_index refers to the cartPole state index
    # current_index is the current_index in the bin of the cart pole is at
    def binary_search(self, state_index, value):
        current_bin = self.state_bins[state_index]
        bin_length = len(self.state_bins[state_index])

        start = 0
        mid = bin_length/2
        end = bin_length

        while start < end:
            if value < current_bin[mid]:
                end = mid
                mid = (start+end)/2
            else:
                start = mid
                mid = (start+end)/2

        return current_bin[mid]




    def _state_to_bin(self, state):

        # for each value in state, put them to the required bin
        for index, value in enumerate(state):
            break
        pass


    def _state_to_table(self, state):

        pass

    def _make_state_bins(self, low, high):
        return np.linspace(low, high, self.bin_size)