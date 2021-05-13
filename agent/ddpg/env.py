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
    def __init__(self, data_path, new_model, train_length=5, target_length=1,\
                num_units=100, dropout=0.1, lr=0.005, temp_target=2.0,\
                episode_length=3360, action_space=6):

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
        self.action_space = action_space

        # get cooler data
        self.target_cols = ['COOLER TEMP 1', 'COOLER TEMP 2']
        cooler_cols = pd_data.columns.str.contains('COOLER|DEW|OUTSIDE TEMP|OUTSIDE HUMIDITY|WIND|PRESSURE \(kPa\)', regex=True)
        action_cols = pd_data.columns.str.contains('COOLER COMP POWER')
        self.action_idx = np.where(action_cols==True)
        self.target_idx = [pd_data.columns.get_loc(c) for c in self.target_cols]
        self.data = pd_data.iloc[:,cooler_cols]

        # self.state_space = self.data.drop(self.data.columns[self.action_idx[0]], axis=1)
        target = self.data[self.target_cols]

        self.input_dims = (len(self.data.columns),)
        # self.input_shape = (len(self.state_space.columns),)


        def normalise(df):
            vals = df.values
            transformer = MinMaxScaler(feature_range=(-1,1))
            transformer.fit(vals)
            norm = transformer.transform(vals)
            return norm, transformer

        # normalise
        self.data_norm, self.f_trans = normalise(self.data)
        # self.state_space_norm, self.s_trans = normalise(self.state_space)
        self.t_norm, self.t_trans = normalise(target)
        _, self.a_trans = normalise(self.data.iloc[:,self.action_idx[0]]) # get transformer only


        # calculate normalised target value
        dummy = np.asarray([self.temp_target,self.temp_target]).reshape(1,-1)
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
            self.model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test), callbacks=[es])
            self.model.save('env_model')
        else:
            self.model = load_model('env_model')

    def step(self, state, action, step_idx, evaluate=False):
        # create library of past state-actions for lstm
        if evaluate and self.step_cntr == 0:
            self.past_sa = np.array(self.data_norm[max(0, step_idx-self.train_length):step_idx+1])
        elif not evaluate and self.step_cntr % self.episode_length == 0: # then we are at start of episode
            self.past_sa = np.array(self.data_norm[max(0, step_idx-self.train_length):step_idx]) # instantiate new past state-actions from truth

        self.past_sa[-1][self.action_idx[0]] = action
        history = self.past_sa[-self.train_length:]
        history = history.reshape((1, self.train_length, self.input_dims[0])) # correct input shape for lstm
        self.target_ = self.model.predict(history)
        self.temps = self.t_trans.inverse_transform(self.target_)

        # append predictions to real values e.g. weather
        step_idx_ = step_idx+1
        next_sa = np.array(self.data_norm[step_idx_])
        # true_observation_ = self.state_space_norm[step_idx_]

        next_sa[self.target_idx] = self.target_ # add predictions to real data
        # true_observation_[self.target_idx] = self.target_

        self.past_sa = np.vstack([self.past_sa, next_sa])
        self.observation_ = next_sa

        # calculate reward
        self.action_kw = self.a_trans.inverse_transform(action.numpy().reshape(1,-1))[0] # get kw values by converting tensor to numpy
        power_reward = -np.sum(self.action_kw)
        temp_diff1 = max(0, ((self.temps[0][-2] - self.temp_target)))
        temp_diff2 = max(0, ((self.temps[0][-1] - self.temp_target)))
        temp_reward = (temp_diff1+temp_diff2)*(-1000)
        self.mean_temp = np.mean(self.temps[-2:])
        self.reward = power_reward + temp_reward

        self.step_cntr += 1

        # run episode continuously for evaluation
        if evaluate:
            done = self.step_cntr == self.data.shape[0]-self.train_length
        else:
            if self.step_cntr % self.episode_length == 0:
                done = True
            else:
                done = False

        return self.observation_, step_idx_, self.reward, done

    def reset(self, start=False):
        if start:
            self.step_idx = self.train_length - 1
            return self.data_norm[self.step_idx], self.step_idx
        else:
            self.step_idx = np.random.randint(low=self.train_length, high=self.data_norm.shape[0]-self.episode_length, size=1)[0]
            return self.data_norm[self.step_idx], self.step_idx
