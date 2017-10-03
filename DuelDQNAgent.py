import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import optimizer, Adam
import numpy as np
from torch.autograd import Variable

from Agent import Agent
from ReplayMemory import ReplayMemory

class DuelDQNAgent(Agent):
    def __init__(self, number_actions, env, batch_size):
        super(DuelDQNAgent, self).__init__(number_actions, env)
        self.batch_size = batch_size
        self.local_network = DuelDQN(number_actions)
        self.memory = ReplayMemory(1024)

        self.optimizer = Adam(self.local_network.parameters())
        self.loss_function = nn.MSELoss()

    def preprocess_state(self, state):
        state = np.array(state, dtype=np.float32)
        return Variable(torch.from_numpy(state))

    def step(self, state):
        state = self.preprocess_state(state)
        action = self.local_network(state).max(1)[1].data[0][0] #[0] is value, [1] is the index for the first []
        next_state, reward, done, _ = self.env.step(action)
        next_state = next_state.reshape(1, -1)
        variable_next_state = self.preprocess_state(next_state)

        self.memory.add_memory(state, reward, action, variable_next_state, done)

        return next_state, reward, done

    def train(self):

        memories = self.memory.get_batches(self.batch_size)

        for memory_state, memory_reward, memory_action, memory_next_state, memory_done in memories:
            memory_action = Variable(torch.LongTensor([[memory_action]]))
            q_value = self.local_network(memory_state).gather(1, memory_action)

            target = self.local_network(memory_next_state).max(1)[0] * 0.90 + memory_reward

            self.optimizer.zero_grad()
            loss = self.loss_function(q_value, target)
            loss.backward()
            self.optimizer.step()






class DuelDQN(nn.Module):

    def __init__(self, number_actions):
        super(DuelDQN, self).__init__()
        self._build_model(number_actions)


    def _build_model(self, number_actions):
        self.fc1 = nn.Linear(4, 20)
        self.fc2 = nn.Linear(20, 20)


        self.value = nn.Linear(20, 1)
        self.advantage = nn.Linear(20, number_actions)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))

        value = self.value(x)
        advantage = self.advantage(x)

        qValue = value.expand_as(advantage) + advantage - advantage.mean(1).expand_as(advantage)

        return qValue