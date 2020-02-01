#参考→https://qiita.com/sugulu/items/bc7c70e6658f204f85f9

import warnings
warnings.simplefilter('ignore')

import gym
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
from keras.layers.recurrent import LSTM
from gym import wrappers
from keras import backend as K
import tensorflow as tf

import csv

def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)
    return K.mean(loss)


class Network:
    def __init__(self, learning_rate=0.01, state_size=4,action_size=2, hidden_size=10):
        with tf.name_scope('Model') as scope:
            self.model = Sequential()
#            self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
            self.model.add(Reshape((1,state_size)))
            self.model.add(
                LSTM(hidden_size,input_shape = (None, state_size))
            )
            self.model.add(Dense(hidden_size, activation='relu'))
            self.model.add(Dense(action_size, activation='linear'))
            self.optimizer = Adam(lr=learning_rate)

        self.model.compile(loss=huberloss, optimizer=self.optimizer)

    def replay(self, memory, batch_size, gamma, targetQN ,state_size):
        inputs = np.zeros((batch_size, state_size))
        targets = np.zeros((batch_size, 2))
        mini_batch = memory.sample(batch_size)

        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            target = reward_b

            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                retmainQs = self.model.predict(next_state_b)[0]
                next_action = np.argmax(retmainQs)
                target = reward_b + gamma * targetQN.model.predict(next_state_b)[0][next_action]

            targets[i] = self.model.predict(state_b)
            targets[i][action_b] = target

        self.model.fit(inputs, targets, epochs=1, verbose=0)


class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        return len(self.buffer)

class Actor:
    def get_action(self, epsilon, state, episode, mainQN):
        
        retTargetQs = mainQN.model.predict(state)[0]
        if epsilon <= np.random.uniform(0,1):
            action = np.argmax(retTargetQs)

        else:
            action = np.random.choice([0,1])

        return action

def Experiment(fileNumber):
#----------Hyper Parameters-----------#

    gamma = 0.99
    epsilon = 0.25
    hidden_size = 8
    learning_rate = 0.001
    memory_size = 10000
    batch_size = 32

#-------------Env Settings----------------#
    num_episodes = 1000
    display_per_episodes = 10
    max_number_of_steps = 200
    env = gym.make('CartPole-v0')
    islearned = 0
    total_score = 0
    state_size = 12

    mainQN = Network(state_size = state_size, hidden_size=hidden_size, learning_rate=learning_rate)
    targetQN = Network(state_size = state_size, hidden_size=hidden_size, learning_rate=learning_rate)

    memory = Memory(max_size=memory_size)
    actor = Actor()

    f = open('results/LSTM/result' + fileNumber + '.csv', 'w', newline="")
    writer = csv.writer(f)
    writer.writerow(['episode','total steps','time'])

    success_num = 0
    total_steps = 0

    for episode in range(num_episodes):
        env.reset()
        state, reward, done, _ = env.step(env.action_space.sample())
        state1 = np.reshape(state, [1,4])
        state2 = state1
        state3 = state2

        state = np.concatenate([state1,state2,state3], axis = 1)

        targetQN.model.set_weights(mainQN.model.get_weights())

        for t in range(2, max_number_of_steps + 1):

            env.render()

            action = actor.get_action(epsilon, state ,episode, mainQN)
            state3 = state2
            state2 = state1
            state1, reward, done, info = env.step(action)
            #convert List to Array
            state1 = np.reshape(state1, [1,4])

            next_state = np.concatenate([state1,state2,state3], axis=1)

            total_steps += 1
            total_score += 1

            reward = 0.005

            #giving rewards
            if done:
                next_state = np.zeros(state.shape)
                #環境が200stepまでに終了する場合がある(バグ？)
                if t < 195:
                    reward = -1
                    success_num = 0
                else:
                    success_num += 1
            
            memory.add((state, action, reward, next_state))
            state = next_state

            if (memory.len() > batch_size) and not islearned:
                mainQN.replay(memory, batch_size, gamma, targetQN, state_size)

            targetQN.model.set_weights(mainQN.model.get_weights())

            if done:

                writer.writerow([episode, total_steps, t])
                break

        if success_num >= 10:
            print('learning successed')
            break

        if episode % display_per_episodes == 0 and episode != 0:
            print("episode " + str(episode) + " done. " + "average time = " + str(total_score/display_per_episodes) + " total_steps = " + str(total_steps))
            total_score = 0

    print('finish.')
    env.close()

def main():
    expMax = 10
    i = 0

    for i in range(expMax):
        print('Experiment ' + str(i) + ' start')
        Experiment(str(i))

    return

main()