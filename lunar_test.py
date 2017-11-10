import random
import gym
import numpy as np
from collections import deque

gym.envs.registration.register(
    id='LunarLander-v3',
    entry_point='lunar_lander:LunarLander',
    max_episode_steps=1000,
    reward_threshold=200,
)
env = gym.make('LunarLander-v3')

from lunar_lander import LunarLander

"""
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
"""
EPISODES = 1


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))

        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        t = []
        s = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            t.append(target_f)
            s.append(state)
        #self.model.fit(state, target_f, epochs=1, verbose=0)
        self.model.fit(np.concatenate(s), np.concatenate(t), epochs=1, verbose=0)


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.epsilon = 0.3
        #self.model = load_model(name)
        self.model.summary()

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    #    env = gym.make('LunarLander-v3')
    env = LunarLander()

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    #agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 1
    if 0:
        agent.load("model.dat")
    for e in range(EPISODES):
        #agent.load("../../Downloads/model_900.h5")
        #agent.epsilon = 0.0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        tot_rew = 0
        for time in range(300):


            #action = agent.act(state)
            action = 0
            next_state, reward, done, _ = env.step(action)
            tot_rew += reward
            if time < 100:
                save_state = next_state
            #reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            #agent.remember(state, action, reward, next_state, done)
            state = next_state
            if e % 100 == 0:
                env.render()
                print(reward)
            if done:
                env.set_state(save_state)
                #print("episode: {}/{}, score: {}, time {}, e: {:.2}"
                #      .format(e, EPISODES, tot_rew/time, time, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            print("heh")
            #agent.replay(batch_size)
            #agent.save("model.dat")
