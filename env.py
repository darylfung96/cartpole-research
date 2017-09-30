import gym
import time
import argparse

from QtableAgent import QtableAgent
from DQNAgent import DQNAgent
from DDQN import DDQNAgent

BIN_SIZE = 8

MAX_EPISODE = 1000
MAX_STEP = 1000
BATCH_SIZE = 32

"""
Parse arguments
"""
argparser = argparse.ArgumentParser('Parse argument')
argparser.add_argument('--env')
argparser.add_argument('--agent')
argparsed = argparser.parse_args()





env = gym.make('CartPole-v1')
state = env.reset()

#agent = QtableAgent(env.action_space.n, env, BIN_SIZE, explore_rate=0.5)
agent = DDQNAgent(env.action_space.n, env, BATCH_SIZE)

total_reward = 0
max_reward = 0
agent.load_model()


def evaluate(state):
    for current_episode in range(1, MAX_EPISODE):
        for step in range(MAX_STEP):
            env.render()
            state, reward, done = agent.step(state)
            time.sleep(0.016)

            if done:
                state = env.reset()
                pass
                #state = env.reset()


def train(state, total_reward, max_reward):
    for current_episode in range(1, MAX_EPISODE):
        for step in range(MAX_STEP):
            #env.render()
            state = state.reshape(1, -1)
            state, reward, done = agent.step(state)
            total_reward += reward

            if total_reward > max_reward:
                max_reward = total_reward

            if total_reward >= 449:
                agent.save_model()

            if done:
                state = env.reset()
                print('episode: ', current_episode)
                print('rewards: ', total_reward)
                print('max_reward: ', max_reward)
                print()
                total_reward = 0
                agent.update_target_weights()
                break

        agent.train()


evaluate(state)
# train(state, total_reward, max_reward)
