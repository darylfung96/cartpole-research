from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout, Input
from tensorflow.contrib.keras.python.keras.optimizers import Adam
from tensorflow.contrib.keras.python.keras import backend as K
from Agent import Agent
from ReplayMemory import ReplayMemory
import random
import numpy as np
import tensorflow as tf

UPDATE_FREQUENCY = 8

class DDQNAgent(Agent):
    def __init__(self, number_actions, env, batch_size):
        super(DDQNAgent, self).__init__(number_actions, env)
        input_shape = env.observation_space.shape
        self.explore_rate = 0.8
        self.batch_size = batch_size
        self.episode_gone = 0

        self.local_network = self._build_model(input_shape, number_actions)
        self.local_network_weights = self.local_network.trainable_weights
        self.target_network = self._build_model(input_shape, number_actions)
        self.update_target_weights()

        self.total_reward = 0
        self.memory = ReplayMemory(1024)

        # log summary
        self.score = tf.placeholder(tf.int32)
        tf.summary.scalar('reward', self.score)
        self.summary_op = tf.summary.merge_all()
        self.reward_writer = tf.summary.FileWriter('./log_graph/DDQN', graph=K.get_session().graph)


    """
    Huber loss function
    sqrt( 1 + error ^2 ) - 1
    """
    def _loss_function(self, true_value, prediction):
        error = true_value - prediction
        return K.mean(K.sqrt(1 + K.square(error)) - 1, axis=-1)

    def update_target_weights(self):
        if self.episode_gone % 10 == 0:
            self.target_network.set_weights(self.local_network.get_weights())
            self.current_total_step = 0
            print('updating target weights...')


    def _build_model(self, input_shape, output_shape):
        state_input = Input(shape=[4])
        layer1 = Dense(20, activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform')(state_input)
        layer2 = Dense(20, activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform')(layer1)
        outputs = Dense(output_shape, activation='linear', kernel_initializer='random_uniform', bias_initializer='random_uniform')(layer2)

        model = Model(inputs=state_input, outputs=outputs)
        adam = Adam()
        model.compile(optimizer=adam, loss=self._loss_function)

        return model

    def update_graph(self):
        summary = K.get_session().run(self.summary_op, feed_dict={ self.score: self.total_reward})
        self.reward_writer.add_summary(summary, self.episode_gone)

    def step(self, state):
        state = np.reshape(state, [1, 4])
        if random.random() < self.explore_rate:
            best_action = self.env.action_space.sample()
            self.explore_rate -= 0.01
        else:
            action = self.local_network.predict(state)[0]
            best_action = np.argmax(action)

        next_state, reward, done, _ = self.env.step(best_action)
        next_state = next_state.reshape(1, 4)
        self.total_reward += 1

        if done:
            if self.total_reward < 449:
                reward = -100
            # add summary
            self.update_graph()

            self.total_reward = 0
            self.episode_gone += 1
            self.update_target_weights()


        self.memory.add_memory(state, reward, best_action, next_state, done)

        return next_state, reward, done

    def save_model(self):
        self.local_network.save_weights('./ddqn_model/ddqn.h5')
        print('saving model...')

    def load_model(self):
        self.local_network.load_weights('./ddqn_model/ddqn.h5')
        self.update_target_weights()
        print('loading model...')

    def train(self):
        memories = self.memory.get_batches(self.batch_size)

        for memory_state, memory_reward, memory_action, memory_next_state, memory_done in memories:
            target = self.local_network.predict(memory_state)

            a = self.local_network.predict(memory_next_state)[0]
            target_a = self.target_network.predict(memory_next_state)

            target[0][memory_action] = memory_reward if memory_done \
            else memory_reward + 0.90 * target_a[0][np.argmax(a)]
            self.local_network.fit(memory_state, target, verbose=0)

