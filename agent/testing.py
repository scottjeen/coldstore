from dqn import Agent
from env import Env
import numpy as np
import tensorflow as tf

data_path = '/Users/ScottJeen/OneDrive - University of Cambridge/research/modelling/emerson_data/env'

env = Env(data_path, new_model=False)

agent = Agent(alpha=0.0005, gamma=0.99, discrete_actions=env.discrete_actions,\
                action_space=env.action_space, n_actions=env.n_actions,\
                action_items=env.action_items, epsilon=1.0, batch_size=env.episode_length,\
                input_dims=env.input_dims,epsilon_dec=0.999,
                epsilon_end=0.01, mem_size=1000000, fname='dqn_model.h5')

agent.load_model()

agent_actions = []
true_actions = []
temps = []
agent_powers = []
true_powers = []

observation, index = env.reset(start=True)
while index < env.data_norm.shape[0]:
    action = agent.choose_action(observation, exploit=True)
    observation_, index_, reward, done = env.step(observation, action, index)
    observation =  observation_
    index = index_
    true_action = (env.a_trans.inverse_transform(observation[env.action_idx].reshape(1,-1)))

    agent_actions.append(env.a_trans.inverse_transform(action))
    true_actions.append(true_action)
    temps.append(env.t_trans.inverse_transform(observation_[env.target_idx].reshape(1,-1)))
    agent_powers.append(np.sum(env.action_kw))
    true_powers.append(np.sum(true_action))

print('total true power consumption: {:.2f} kW'.format(np.sum(true_powers)))
print('total agent power consumption: {:.2f} kW'.format(np.sum(agent_powers)))


x = [np.arange(env.data_norm.shape[0])]

plt.figure(figsize=(10,5))
plt.title('temps and humidities for first 1000 timesteps')
plt.plot(x, temps[0:1000])
plt.xlabel('time')
plt.ylabel('temp')
plt.legend()
plt.savefig('/evaluation/temps.png', dpi=300)

plt.figure(figsize=(10,5))
plt.title('power consumption for first 1000 timesteps')
plt.plot(x, agent_powers[0:1000], color='red')
plt.plot(x, true_powers[0:1000], color='red')
plt.xlabel('time')
plt.ylabel('power (kw)')
plt.legend()
plt.savefig('/evaluation/power.png', dpi=300)
