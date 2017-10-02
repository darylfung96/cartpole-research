import gym
from DDQN import DDQNAgent
from DQNAgent import DQNAgent
from DuelDQNAgent import DuelDQNAgent

def parseArgument(env, agent, batch_size):
    env = gym.make(env)
    agentReturn = None

    if agent == 'ddqn':
        agentReturn = DDQNAgent(env.action_space.n, env, batch_size)
    elif agent == 'dqn':
        agentReturn = DQNAgent(env.action_space.n, env, batch_size)
    elif agent == 'dueldqn':
        agentReturn = DuelDQNAgent(env.action_space.n, env, batch_size)


    return env, agentReturn
