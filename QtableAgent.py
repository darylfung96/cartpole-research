from Agent import Agent
import numpy as np
import random
import math

explore_rate = 0.8
discount_rate = 0.9
learning_rate = 0.1
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
        self.length_state_bins = self.bin_size-2
        # convert to q table
        self.q_table = np.zeros(( self.length_state_bins**number_state, env.action_space.n ))

    def step(self, state):
        rand = random.random()

        self.state = self._state_to_table(state)

        if rand < self.explore_rate:
            self.action = self.env.action_space.sample()
            self.explore_rate -= DECAY_RATE
        else:
            self.action = np.argmax(self.q_table[self.state])

        next_state, self.reward, done, _ = self.env.step(self.action)

        if done:
            self.reward = -1.0

        self.table_next_state = self._state_to_table(next_state)

        self.train()


        return next_state, self.reward, done

    def train(self):
        self.q_table[self.state][self.action] = learning_rate * (self.reward + discount_rate * np.amax(self.q_table[self.table_next_state]) - self.state)


# state managements #
##############################################################################################
    # the state_index is the index for the state,
    # cartPole has 4 states, state[4], state_index refers to the cartPole state index
    # current_index is the current_index in the bin of the cart pole is at
    def _put_to_bin(self, state_index, value):
        current_bin = self.state_bins[state_index]
        bin_length = len(self.state_bins[state_index])

        min_index = 0
        min_value = abs(value-current_bin[0])

        # find the closest number to value
        for index in range(1, bin_length):
            current_value = abs(current_bin[index] - value)
            if  current_value < min_value:
                min_value = current_value
                min_index = index

        return min_index



    def _state_to_bin(self, state):
        state_indexes = []
        state=state.reshape(-1)
        # for each value in state, put them to the required bin
        for index, value in enumerate(state):
            state_indexes.append(self._put_to_bin(index, value))
        return state_indexes


    def _state_to_table(self, state):
        state_indexes = self._state_to_bin(state)
        state = 0
        for index, state_bin in enumerate(state_indexes):
            state += self.length_state_bins**index * state_bin

        return state

    ##############################################################################################


    def _make_state_bins(self, low, high):
        return np.linspace(low, high, self.bin_size)[1:-1]