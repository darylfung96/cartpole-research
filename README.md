# cartpole-research

## Deep Reinforcement Learning
</br>

## Current available agent
- DQN
    - Used greedy algorithm to compute the best action based on the Q value of the next state
    - Disadvantages are overestimation of the Q value
    - [DQN Research Paper](https://arxiv.org/pdf/1312.5602.pdf)
- DDQN
    - Just like DQN, but the difference is that it uses a target network for training
    - Reduces overestimation of the Q value and improve stability of training
    - target network is updated after n episodes from the local network
    - [DDQN Research Paper](https://arxiv.org/pdf/1509.06461.pdf)
- DuelDQN
    - Separate action and state 
    - Some states are not important so when we separate them, the agent can determine which states are consider to be important when choosing action and which are not
    - [DuelDQN Research Paper](https://arxiv.org/pdf/1511.06581.pdf)


Dependencies:
- [TensorFlow](https://www.tensorflow.org/)
- Numpy

To run:
    
    Python env.py --env CartPole-v1 --agent ddqn --type train
    
    
Arguments:
- agent
  - dqn
  - ddqn
  - dueldqn

- type
  - train
  - evaluate


To log graph created by TensorBoard (run this in command line inside the parent folder of log_graph):

    tensorboard --logdir './log_graph'
