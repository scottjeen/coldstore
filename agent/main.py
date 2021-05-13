from dqn import Agent
from env import Env
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

data_path = os.path.join(os.getcwd(), 'env')

if __name__ == '__main__':
    env = Env(data_path, new_model=False)
    n_episodes = 350
    agent = Agent(alpha=0.0005, gamma=0.99, discrete_actions=env.discrete_actions,\
                    action_space=env.action_space, n_actions=env.n_actions,\
                    action_items=env.action_items, epsilon=1.0, batch_size=64,\
                    input_dims=env.input_dims,epsilon_dec=0.999,
                    epsilon_end=0.01, mem_size=1000000, fname='dqn_model.h5')

    scores = []
    avg_scores = []
    epsilon = []

    for i in tqdm(range(n_episodes)):
        done = False
        score = 0
        observation, index = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, index_, reward, done = env.step(observation, action, index)
            agent.remember(observation, action, reward, observation_, done)
            observation =  observation_
            index = index_
            agent.learn()
            score += reward

        scores.append(score)
        epsilon.append(agent.epsilon)

        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        avg_scores.append(avg_score)
        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)

        if i % 10 == 0 and i > 0:
            agent.save_model()

    x = [i+1 for i in range(n_episodes)]
    plt.figure(figsize=(10,5))
    plt.plot(x, scores)
    plt.savefig('scores.png', dpi=300)

    plt.figure(figsize=(10,5))
    plt.plot(x, avg_scores)
    plt.savefig('avg_scores.png', dpi=300)

    plt.figure(figsize=(10,5))
    plt.plot(x, epsilon)
    plt.savefig('epsilon.png', dpi=300)
