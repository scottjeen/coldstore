import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import datetime
import itertools

class Env(object):
    def __init__(self, data_path, new_model, train_length=20, target_length=1,\
                num_units=100, dropout=0.1, lr=0.005, temp_target=4.0,\
                episode_length=64, discrete_actions=11, action_items=6):

        pd_data = pd.read_pickle(data_path)
        pd_data = pd_data.drop(columns=['COOLER HUMIDITY 3'])
        self.train_length = train_length
        self.target_length = target_length
        self.num_units = num_units
        self.dropout = dropout
        self.lr = lr
        self.temp_target = temp_target
        self.episode_length = episode_length
        self.step_cntr = 0
        self.discrete_actions = discrete_actions
        self.action_items = action_items
        self.action_space = np.asarray(list(itertools.product(range(self.discrete_actions),\
                                                            range(self.discrete_actions),\
                                                            range(self.discrete_actions),\
                                                            range(self.discrete_actions),\
                                                            range(self.discrete_actions),\
                                                            range(self.discrete_actions)))) # all combinations of discrete actions

        # get cooler dataac
        cooler_cols = pd_data.columns.str.contains('COOLER|DEW|OUTSIDE TEMP|OUTSIDE HUMIDITY|WIND|PRESSURE \(kPa\)', regex=True)
        action_cols = pd_data.columns.str.contains('COOLER COMP POWER')
        self.action_idx = np.where(action_cols==True)
        self.target_idx = [6,7,8,13,14]
        self.data = pd_data.iloc[:,cooler_cols]
        target = self.data.iloc[:,self.target_idx]

        self.n_actions = self.discrete_actions**self.action_items
        self.input_dims = len(self.data.columns)


        def normalise(df):
            vals = df.values
            transformer = MinMaxScaler(feature_range=(-1,1))
            transformer.fit(vals)
            norm = transformer.transform(vals)
            return norm, transformer

        self.data_norm, self.f_trans = normalise(self.data)
        self.t_norm, self.t_trans = normalise(target)
        _, self.a_trans = normalise(self.data.iloc[:,self.action_idx[0]]) # get transformer only


        # calculate normalised target value
        dummy = np.asarray([0.6,0.6,0.6,self.temp_target,self.temp_target]).reshape(1,-1)
        self.temp_target1_norm = self.t_trans.transform(dummy)[0][-2]
        self.temp_target2_norm = self.t_trans.transform(dummy)[0][-1]

        # sequence splitting function for LSTM
        def split_sequence(feature_sequence, target_sequence, train_length, target_length):
            X, y = [], []
            for i in range(len(feature_sequence)):
              # find the end of this pattern
              end_train = i + train_length
              end_target = end_train + target_length
              # check if we are beyond the sequence
              if end_train > len(feature_sequence)-target_length:
                break
              # gather input and output parts of the pattern
              seq_x, seq_y = feature_sequence[i:end_train], target_sequence[end_train:end_target]
              X.append(seq_x)
              y.append(seq_y)
            return np.asarray(X), np.asarray(y)

        X, y = split_sequence(self.data_norm, self.t_norm, self.train_length, self.target_length)
        X = X.reshape((X.shape[0], X.shape[1], X[0].shape[1]))
        y = y.reshape((y.shape[0], y.shape[1] * y[0].shape[1]))

        # split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # create the net
        if new_model:
            self.model = Sequential([
                            LSTM(self.num_units, activation='relu', return_sequences=True, kernel_initializer='he_normal', input_shape=(train_length, X_train.shape[2])),
                            LSTM(self.num_units, activation='relu', return_sequences=True, kernel_initializer='he_normal'),
                            Dropout(self.dropout),
                            LSTM(self.num_units, activation='relu', kernel_initializer='he_normal'),
                            Dense(self.num_units, activation='relu', kernel_initializer='he_normal'),
                            Dense(self.num_units, activation='relu', kernel_initializer='he_normal'),
                            Dropout(self.dropout),
                            Dense(self.num_units, activation='relu', kernel_initializer='he_normal'),
                            Dense(self.num_units, activation='relu', kernel_initializer='he_normal'),
                            Dense(y_train.shape[1])
                            ])


            optimizer = tf.optimizers.Adam(learning_rate=lr)
            self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            es = EarlyStopping(monitor='val_loss', patience=2)
            self.model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test), callbacks=[es])
            self.model.save('env_model')
        else:
            self.model = load_model('env_model')

    def step(self, state, action, step_idx):

        # convert one-hot action encoding to lstm format i.e. kW
        action_kw = np.asarray([i*0.6 for i in action]).reshape(1,-1) # multiply one-hot encoding by 0.6kW
        self.action_kw = self.a_trans.transform(action_kw) # normalise
        state[self.action_idx[0]] = self.action_kw # update state with new actions
        self.data_norm[step_idx] = state # update environment with new state-action pair

        # create history for LSTM and predict
        history = self.data_norm[step_idx-self.train_length:step_idx]
        history[-1] = state # update last row in history with new actions
        history = history.reshape((1, self.train_length, self.input_dims)) # correct input shape for lstm
        self.target_ = self.model.predict(history)

        # append predictions to real values e.g. weather
        step_idx_ = step_idx+1
        true_observation_ = self.data_norm[step_idx_]
        true_observation_[self.target_idx] = self.target_ # add predictions to real data
        self.observation_ = true_observation_

        # calculate reward
        energy = np.sum(self.observation_[self.action_idx[0]])
        temp_diff = max(0, ((self.observation_[-2] + self.observation_[-1])\
                                - (self.temp_target1_norm + self.temp_target2_norm))) # difference between sum of two cooler temps and temp threshold
        self.reward = -energy-10000000*temp_diff

        self.step_cntr += 1
        if self.step_cntr % self.episode_length == 0:
            done = True
        else:
            done = False

        return self.observation_, step_idx_, self.reward, done

    def reset(self, start=False):
        if start:
            self.step_idx = self.train_length
            return self.data_norm[self.step_idx], self.step_idx
        else:
            self.step_idx = np.random.randint(low=self.train_length, high=self.data_norm.shape[0], size=1)[0]
            return self.data_norm[self.step_idx], self.step_idx
