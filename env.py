import gym
import random
import numpy as np

from QtableAgent import QtableAgent
from DQNAgent import DQNAgent
from DDQN import DDQNAgent

BIN_SIZE = 8

MAX_EPISODE = 1000
MAX_STEP = 200
BATCH_SIZE = 64

env = gym.make('CartPole-v0')
state = env.reset()

#agent = QtableAgent(env.action_space.n, env, BIN_SIZE, explore_rate=0.5)
agent = DDQNAgent(env.action_space.n, env, BATCH_SIZE)

total_reward = 0
max_reward = 0
#agent.load_model()
for current_episode in range(1, MAX_EPISODE):
    for step in range(MAX_STEP):
        env.render()
        state = state.reshape(1, -1)
        state, reward, done = agent.step(state)
        total_reward += reward

        if total_reward > max_reward:
            max_reward = total_reward

        if total_reward >= 199:
            agent.save_model()

        if done:
            state = env.reset()
            print('episode: ', current_episode)
            print('rewards: ', total_reward)
            print('max_reward: ', max_reward)
            print()
            total_reward = 0
            break


