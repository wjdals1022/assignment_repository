import numpy as np
import os
import pandas as pd
import random
from collections import defaultdict
import gym
import gym_minigrid
import matplotlib.pyplot as plt

class SARSA:
    def __init__(self, actions, agent_indicator=10, epsilon=0.99, epsilon_min=0.1, epsilon_decay=0.995):
        self.actions = actions
        self.agent_indicator = agent_indicator
        self.alpha = 0.001
        self.gamma = 0.9
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_values = defaultdict(lambda: [0.0] * actions)
        
    def _convert_state(self, s):
        return np.where(s == self.agent_indicator)[0][0]
        
    def update(self, state, action, reward, next_state, next_action):
        state = self._convert_state(state)
        next_state = self._convert_state(next_state)
        
        q_value = self.q_values[state][action]
        
        next_q_value = self.q_values[next_state][next_action]
        
        td_error = reward + self.gamma * next_q_value - q_value
        self.q_values[state][action] += self.alpha * td_error
    
    def act(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state = self._convert_state(state)
            q_values = self.q_values[state]
            action = np.argmax(q_values)
        return action
    
    def decay_epsilon(self, reward):
        if reward > 0:  # 보상을 받을 때만 epsilon을 감소
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    

from utils import gen_wrapped_env, show_video

env = gen_wrapped_env('MiniGrid-SimpleCrossingS9N1-v0')
obs = env.reset()

agent_position = obs[0]

agent = SARSA(3, agent_position)

rewards = []
for ep in range(10000):
    done = False
    obs = env.reset()
    action = agent.act(obs)
    
    ep_rewards = 0
    while not done:
        next_obs, reward, done, info = env.step(action)

        next_action = agent.act(next_obs)

        agent.update(obs, action, reward, next_obs, next_action)
        
        ep_rewards += reward
        
        # 보상을 받을 때만 epsilon을 줄입니다.
        agent.decay_epsilon(reward)
        
        obs = next_obs
        action = next_action
    
    rewards.append(ep_rewards)
    if (ep+1) % 20 == 0:
        print("episode: {}, rewards: {}".format(ep+1, ep_rewards))

env.close()

{s: np.round(q, 5).tolist() for s, q in agent.q_values.items()}
show_video()

os.makedirs('./logs', exist_ok=True)
pd.Series(rewards).to_csv('./logs/rewards_sarsa.csv')
