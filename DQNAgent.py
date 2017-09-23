import tensorflow as tf
import numpy as np
from Agent import Agent

#TODO: we need to implement experience_replay because agent overestimate q-value
###
# experience replay will get a random sample instead of the next state

class DQNAgent(Agent):
    def __init__(self, number_actions, env,):
        super(DQNAgent, self).__init__(number_actions, env)

        self.input = tf.placeholder("float", [None, 4])
        self.target = tf.placeholder("float", [None, 2])

        self.w_fc1, self.b1 = self._declare_variable([4, 10])
        self.layer1 = tf.nn.relu(tf.matmul(self.input, self.w_fc1) + self.b1)

        self.w_fc2, self.b2 = self._declare_variable([10, 10])
        self.layer2 = tf.nn.relu(tf.matmul(self.layer1, self.w_fc2) + self.b2)

        self.w_fc3, self.b3 = self._declare_variable([10, number_actions])
        self.output = tf.nn.relu(tf.matmul(self.layer2, self.w_fc3) + self.b3)

        self.train_step = tf.train.AdamOptimizer().minimize(tf.reduce_mean(tf.square(self.target - self.output)))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.memory = []


    """
    argument: state
    ------------------
    state is the current state it is in.
    we will train the state by getting the next state using env.step
    
    We get the target value with the formula:
    current State q value = reward + discount_rate * next best q value
    
    self.train_step will train the optimizer
    """
    def step(self, state):
        action = self.sess.run(self.output, feed_dict={self.input: state})
        target = np.zeros_like(action)
        action = np.argmax(action)

        observation, reward, done, _ = self.env.step(action)


        observation = observation.reshape(1, -1)

        target = self.sess.run(self.output, feed_dict={self.input: observation})
        print(target, '1')
        target[0][action] = reward + 0.9 * np.max(target[0])
        print(target, '2')
        self.sess.run(self.train_step, feed_dict={self.target: target, self.input: state})

        return observation, reward, done



    """
    Declare the variable of weights and biases
    """
    def _declare_variable(self, shape):

        fan_in = shape[0]
        fan_out = shape[1]

        value_bound = np.sqrt(6./(fan_in+fan_out))
        bias_shape = [shape[1]]

        weights = tf.Variable(tf.random_uniform(shape, minval=-value_bound, maxval=value_bound))
        biases = tf.Variable(tf.zeros(bias_shape))

        return weights, biases