import tensorflow as tf
import numpy as np
from Agent import Agent
from ReplayMemory import ReplayMemory
import random

###
# experience replay will get a random sample instead of the next state


class DQNAgent(Agent):
    def __init__(self, number_actions, env, batch_size):
        super(DQNAgent, self).__init__(number_actions, env)

        self.batch_size = batch_size
        self.explore_rate = 0.8

        self.input = tf.placeholder("float", [None, 4])
        self.target = tf.placeholder("float", [None, 2])

        self.w_fc1, self.b1 = self._declare_variable([4, 20])
        self.layer1 = tf.nn.relu(tf.matmul(self.input, self.w_fc1) + self.b1)

        self.w_fc2, self.b2 = self._declare_variable([20, 20])
        self.layer2 = tf.nn.relu(tf.matmul(self.layer1, self.w_fc2) + self.b2)

        self.w_fc3, self.b3 = self._declare_variable([20, number_actions])
        self.output = tf.matmul(self.layer2, self.w_fc3) + self.b3

        self.train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(tf.reduce_mean(tf.square(self.target - self.output)))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.memory = ReplayMemory(10000) # max memory is 1000

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

        self.env.render()
        if random.random() <self.explore_rate:
            best_action = self.env.action_space.sample()
            self.explore_rate -= 0.01
        else:
            action = self.sess.run(self.output, feed_dict={self.input: state})
            best_action = np.argmax(action)

        next_state, reward, done, _ = self.env.step(best_action)
        next_state = next_state.reshape(1, -1)

        if done:
            reward = -150



        if self.batch_size > self.memory.get_length():
            self.memory.add_memory(state, reward, best_action, next_state, done)
        else:
            # training step
            memories = self.memory.get_batches(self.batch_size)
            memory_inputs = []
            memory_targets = []

            for memory_state, memory_reward, memory_action, memory_next_state, memory_done in memories:
                target = self.sess.run(self.output, feed_dict={self.input: memory_state})
                best_next_action = np.max(self.sess.run(self.output, feed_dict={self.input: memory_next_state})[0])
                target[0][memory_action] = memory_reward if memory_done else memory_reward + 0.90 * best_next_action
                # store values to learn
                memory_inputs.append(memory_state[0])
                memory_targets.append(target[0])

            self.sess.run(self.train_step, feed_dict={self.target: memory_targets, self.input: memory_inputs})

        return next_state, reward, done

    def save_model(self):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, './model/model.ckpt')
        print('Model saved in %s' % save_path)

    def load_model(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './model/model.ckpt')
        print('Model restored')



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