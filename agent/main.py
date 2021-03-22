from dqn import Agent
from env import Env
import numpy as np
import tensorflow as tf
import os

data_path = '/Users/ScottJeen/OneDrive - University of Cambridge/research/modelling/emerson_data/env'

if __name__ == '__main__':
    env = Env(data_path)
    n_episodes = 1000
    agent = Agent( alpha=0.0005, gamma=0.99, epsilon=1.0,
                    n_actions=env.n_actions, input_dims=env.input_dims,
                    batch_size=env.episode_length, epsilon_dec=0.999,
                    epsilon_end=0.01, mem_size=1000000, fname='dqn_model.h5')

    scores = []
    epsilon = []

    for i in range(n_episodes):
        done = False
        score = 0
        observation, index = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, index_, reward, done = env.step(observation, action, index)
            agent.remember(observation, action, reward, observation_, done)
            observation =  observation_
            index = index
            agent.learn()
            score += reward

        scores.append(score)
        epsilon.append(agent.epsilon)

        avg_score = np.mean(scores[max(0, i-100):i+1])
        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)

        if i % 10 == 0 and i > 0:
            agent.save_model()
