import gym
from DDQN import DDQNAgent

def parseArgument(env, agent):
    env = gym.make(env)
    agentReturn = None

    if agent == 'DDQN':
        agentReturn = DDQNAgent
