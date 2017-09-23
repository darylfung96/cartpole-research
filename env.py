import gym
import random
import numpy as np

from QtableAgent import QtableAgent
from DQNAgent import DQNAgent

BIN_SIZE = 8


env = gym.make('CartPole-v0')
state = env.reset()

#agent = QtableAgent(env.action_space.n, env, BIN_SIZE, explore_rate=0.5)
agent = DQNAgent(env.action_space.n, env)

total_reward = 0
max_reward = 0

while True:
    env.render()
    state = state.reshape(1, -1)
    state, reward, done = agent.step(state)
    total_reward += reward
    if total_reward > max_reward:
        max_reward = total_reward
        print(max_reward)
    if done:
        state = env.reset()
        total_reward = 0
