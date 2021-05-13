import numpy as np
from ddpg import Agent
import os
from env import Env
from utils import evaluate_agent
import pickle
from tqdm import tqdm

data_path = os.path.join(os.getcwd(), 'data/env')

if __name__ == '__main__':
    env = Env(data_path, new_model=False)
    agent = Agent(input_dims=env.input_dims, env=env, n_actions=env.action_space)

    # n_episodes = 200
    #
    # scores = []
    # avg_ep_scores = []
    # load_checkpoint = False
    #
    # for i in tqdm(range(n_episodes)):
    #     observation, index = env.reset()
    #     done = False
    #     score = 0
    #     temps = []
    #     while not done:
    #         action = agent.choose_action(observation)
    #         observation_, index_, reward, done = env.step(observation, action, index)
    #         agent.remember(observation, action, reward, observation_, done)
    #         observation = observation_
    #         index = index_
    #         agent.learn()
    #         score += reward
    #         temps.append(env.mean_temp)
    #
    #     scores.append(score)
    #     avg_ep_score = np.mean(scores[-env.episode_length:])
    #     avg_ep_scores.append(avg_ep_score)
    #     print('episode:', i, 'score: %.2f' % score, 'avg score: %.2f' % avg_ep_score, 'avg_temp: %.2f degrees' % np.mean(temps))
    #
    # agent.save_model()
    #
    # with open('results/scores_2.pickle', 'wb') as f:
    #     pickle.dump(scores, f)
    #
    # with open('results/avg_ep_scores_2.pickle', 'wb') as f:
    #     pickle.dump(avg_ep_scores, f)

    rb_control, agent_control = evaluate_agent(env, agent)

    with open('results/evaluation/rb_control.pickle', 'wb') as f:
        pickle.dump(rb_control, f)

    with open('results/evaluation/agent_control_2.pickle', 'wb') as f:
        pickle.dump(agent_control, f)
