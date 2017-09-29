from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout, Input
from tensorflow.contrib.keras.python.keras.optimizers import Adam
from tensorflow.contrib.keras.python.keras import backend as K
from Agent import Agent
from ReplayMemory import ReplayMemory
import random
import numpy as np
import tensorflow as tf

class DDQNAgent(Agent):
    def __init__(self, number_actions, env, batch_size):
        super(DDQNAgent, self).__init__(number_actions, env)
        input_shape = env.observation_space.shape
        self.explore_rate = 0.8
        self.batch_size = batch_size

        self.local_network = self._build_model(input_shape, number_actions)
        self.local_network_weights = self.local_network.trainable_weights
        self.target_network = self._build_model(input_shape, number_actions)
        self.update_target_weights()

        self.total_reward = 0
        self.memory = ReplayMemory(1024)

    def loss_function(self, true_value, prediction):
        error = true_value - prediction
        return K.mean(K.sqrt(1 + K.square(error)) - 1, axis=-1)

    def update_target_weights(self):
        self.target_network.set_weights(self.local_network.get_weights())


    def _build_model(self, input_shape, output_shape):
        state_input = Input(shape=[4])
        layer1 = Dense(20, activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform')(state_input)
        layer2 = Dense(20, activation='relu', kernel_initializer='random_uniform', bias_initializer='random_uniform')(layer1)
        outputs = Dense(output_shape, activation='linear', kernel_initializer='random_uniform', bias_initializer='random_uniform')(layer2)

        model = Model(inputs=state_input, outputs=outputs)
        adam = Adam()
        model.compile(optimizer=adam, loss=self.loss_function)#self.loss_function)

        return model

    def step(self, state):
        self.env.render()
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
            if self.total_reward < 199:
                reward = -150
            self.total_reward = 0
            self.update_target_weights()

        self.memory.add_memory(state, reward, best_action, next_state, done)
        # training step
        memories = self.memory.get_batches(self.batch_size)
        memory_inputs = []
        memory_targets = []

        for memory_state, memory_reward, memory_action, memory_next_state, memory_done in memories:
            # target = self.sess.run(self.output, feed_dict={self.input: memory_state})
            # self.target_network.predict(memory_state)
            # best_next_action = np.max(self.sess.run(self.output, feed_dict={self.input: memory_next_state})[0])
            # target[0][memory_action] = memory_reward if memory_done else memory_reward + 0.90 * best_next_action
            # # store values to learn
            # memory_inputs.append(memory_state[0])
            # memory_targets.append(target[0])

        #     DDQN learning
        #     target q value = reward + gamma * target(max local q)
            target = self.local_network.predict(memory_state)

            a = self.local_network.predict(memory_next_state)[0]
            t_a = self.target_network.predict(memory_next_state)

            target[0][memory_action] = memory_reward if memory_done \
                else memory_reward + 0.90 * t_a[0][np.argmax(a)]
            #memory_inputs.append(memory_state[0])
            #memory_targets.append(target[0])
            self.local_network.fit(memory_state, target, verbose=0)

        #self.local_network.fit(np.array(memory_inputs), np.array(memory_targets), batch_size=len(memory_inputs), verbose=0)
        #self.update_target_weights()
        #self.sess.run(self.train_step, feed_dict={self.target: memory_targets, self.input: memory_inputs})

        return next_state, reward, done
