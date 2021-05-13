    # x = [i+1 for i in range(n_episodes)]
    # plt.figure(figsize=(10,5))
    # plt.plot(x, scores)
    # plt.savefig('ddpg_scores.png', dpi=300)
    #
    # plt.figure(figsize=(10,5))
    # plt.plot(x, avg_scores)
    # plt.savefig('ddpg_avg_scores.png', dpi=300)

from env import Env
from ddpg import Agent
from utils import evaluate_agent
import os
import pickle
import pandas as pd

data_path = os.path.join(os.getcwd(), 'data/env')
env = Env(data_path, new_model=False)
agent = Agent(env.input_shape, env=env, n_actions=env.action_space)

rb_control, agent_control = evaluate_agent(env, agent)

with open('results/evaluation/rb_control.pickle', 'wb') as f:
    pickle.dump(rb_control, f)

with open('results/evaluation/agent_control.pickle', 'wb') as f:
    pickle.dump(agent_control, f)
