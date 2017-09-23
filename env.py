import gym
import random
import numpy as np

from QtableAgent import QtableAgent

BIN_SIZE = 7


env = gym.make('CartPole-v0')
env.reset()

agent = QtableAgent(env.action_space.n, env, BIN_SIZE, explore_rate=0.5)



while True:
    number_action = env.action_space.n
    action = random.randint(0, number_action-1)
    env.render()
    state, reward, done, _ = env.step(action)

    if done:
        state = env.reset()
