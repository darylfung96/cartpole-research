import tensorflow as tf
import numpy as np
from Agent import Agent

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

    def step(self, state):
        action = self.sess.run(self.output, feed_dict={self.input: state})
        action = np.argmax(action)

        observation, reward, done = self.env(action)

        self.target =
        self.sess.run(self.train_step)






    def _declare_variable(self, shape):

        fan_in = shape[0]
        fan_out = shape[1]

        value_bound = np.sqrt(6./(fan_in+fan_out))
        bias_shape = [shape[1]]

        weights = tf.Variable(tf.random_uniform(shape, minval=-value_bound, maxval=value_bound))
        biases = tf.Variable(tf.zeros(bias_shape))

        return weights, biases