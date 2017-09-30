# cartpole-research

## Deep Reinforcement Learning
</br>

## Current available agent
- DQN
- DDQN



Dependencies:
- [TensorFlow](https://www.tensorflow.org/)
- Numpy

To run:
    
    Python env.py --env CartPole-v1 --agent ddqn --type train
    
    
Arguments:
- agent
  - dqn
  - ddqn

- type
  - train
  - evaluate


To log graph created by TensorBoard (run this in command line inside the parent folder of log_graph):

    tensorboard --logdir './log_graph'
