import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, action_space, action_items, discrete_actions, discrete=True):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.uint8 if self.discrete else np.float32
        self.action_memory = np.zeros(self.mem_size, dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.action_space = action_space
        self.action_items = action_items
        self.discrete_actions = discrete_actions

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)

        if self.discrete:
            a_idx = get_a_idx(action[0], N=[self.discrete_actions]*self.action_items) # find index of each action in action space of length 10e6
            self.action_memory[index] = a_idx
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = Sequential([
                Dense(fc1_dims, input_shape=(input_dims,)),
                Activation('relu'),
                Dense(fc2_dims),
                Activation('relu'),
                Dense(n_actions)
    ])

    model.compile(optimizer=Adam(lr=lr), loss='mse')

    return model

# calculates action index rather than naive search
def get_a_idx(p, N):
    index = 0
    skip = 1
    for dimension in reversed(range(len(N))):
        index += skip * p[dimension]
        skip *= N[dimension]
    return index

class Agent(object):
    def __init__(self, alpha, gamma, discrete_actions, action_space, n_actions, action_items, epsilon, batch_size,
                input_dims, epsilon_dec=0.999, epsilon_end=0.01,
                mem_size=1000000, fname='dqn_model.h5'):
        self.action_space = action_space
        self.n_actions = n_actions
        self.discrete_actions = discrete_actions
        self.action_items = action_items
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname

        self.memory = ReplayBuffer(mem_size, input_dims,\
                                    n_actions, action_space,\
                                    action_items, discrete_actions, discrete=True)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 128, 128)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state, exploit=False):
        state = state[np.newaxis, :]
        rand = np.random.random()

        if exploit:
            action_vals = self.q_eval.predict(state)
            choice = np.argmax(action_vals)
            action = self.action_space[choice]
            return action.reshape(1,-1)

        else:
            if rand < self.epsilon:
                choice = np.random.choice(self.n_actions)
                action = self.action_space[choice]
            else:
                action_vals = self.q_eval.predict(state)
                choice = np.argmax(action_vals)
                action = self.action_space[choice]
            return action.reshape(1,-1)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)

        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(new_states)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + self.gamma*np.max(q_next, axis=1)*dones

        _ = self.q_eval.fit(states, q_target, verbose=0)

        self.epsilon = self.epsilon*self.epsilon_dec \
                        if self.epsilon > self.epsilon_min else self.epsilon_min

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
