import gym
import time
import argparse

from ArgParse import parseArgument

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
argparser.add_argument('--type')
argparsed = argparser.parse_args()


env, agent = parseArgument(argparsed.env, argparsed.agent, BATCH_SIZE)
state = env.reset()


total_reward = 0
max_reward = 0
#agent.load_model()


def evaluate(state):
    for current_episode in range(1, MAX_EPISODE):
        for step in range(MAX_STEP):
            env.render()
            state = state.reshape(1, -1)
            state, reward, done = agent.step(state)
            time.sleep(0.016)

            if done:
                state = env.reset()
                pass


def train(state, total_reward, max_reward):
    for current_episode in range(1, MAX_EPISODE):
        for step in range(MAX_STEP):
            #env.render()
            state = state.reshape(1, -1)
            state, reward, done = agent.step(state)
            total_reward += reward

            if total_reward > max_reward:
                max_reward = total_reward

            if total_reward >= 499:
                agent.save_model()

            if done:
                state = env.reset()
                print('episode: ', current_episode)
                print('rewards: ', total_reward)
                print('max_reward: ', max_reward)
                print()
                total_reward = 0
#                agent.update_target_weights()
                break

        agent.train()


if argparsed.type == 'train':
    train(state, total_reward, max_reward)
elif argparsed.type == 'evaluate':
    evaluate(state)

