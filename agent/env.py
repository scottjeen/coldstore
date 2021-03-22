import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import datetime

class Env(object):
    def __init__(self, data_path, train_length=20, target_length=1, num_units=100, dropout=0.1, lr=0.005, temp_target=4.0, episode_length=64):
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

        # get cooler data
        cooler_cols = pd_data.columns.str.contains('COOLER|DEW|OUTSIDE TEMP|OUTSIDE HUMIDITY|WIND|PRESSURE \(kPa\)', regex=True)
        action_cols = pd_data.columns.str.contains('COOLER COMP POWER')
        action_idx = np.where(action_cols==True)
        target_idx = [6,7,8,13,14]
        self.data = pd_data.iloc[:,cooler_cols]
        target = self.data.iloc[:,target_idx]

        self.n_actions = len(action_cols[action_cols==True])
        self.input_dims = len(self.data.columns)

        def normalise(df):
            vals = df.values
            transformer = MinMaxScaler(feature_range=(-1,1))
            transformer.fit(vals)
            norm = transformer.transform(vals)
            return norm, transformer

        self.f_norm, self.f_trans = normalise(self.data)
        self.t_norm, self.t_trans = normalise(target)

        # calculate normalised target value
        dummy = [[0.6,0.6,0.6,0.6,self.temp_target,self.temp_target]]
        self.temp_target1 = self.t_trans.transform(dummy)[-2]
        self.temp_target2 = self.t_trans.transform(dummy)[-1]

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

        X, y = split_sequence(self.f_norm, self.t_norm, self.train_length, self.target_length)
        X = X.reshape((X.shape[0], X.shape[1], X[0].shape[1]))
        y = y.reshape((y.shape[0], y.shape[1] * y[0].shape[1]))

        # split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # create the net
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


    def step(state, action, step_idx):

        self.state[action_idx] = action # update state with new actions
        self.data[self.step_idx] = state # update environment with new state-action pair

        # create history for LSTM and predict
        history = self.data[step_idx-self.train_length:step_idx]
        history[-1] = self.state # update last row in history with new actions
        history = history.reshape((1, self.train_length, self.input_dims)) # correct input shape for lstm
        self.target_ = self.model.predict(history)

        # append predictions to real values e.g. weather
        step_idx_ = step_idx+1
        true_observation_ = self.data[step_idx_]
        true_observation_[action_idx] = self.target_ # add predictions to real data
        self.observation_ = true_observation_

        # calculate reward
        energy = np.sum(self.observation_[action_idx], axis=1)
        temp_diff = np.max(0, ((self.observation_[-2] + self.observation_[-1]) - (self.temp_target1 + self.temp_target2))) # difference between sum of two cooler temps and temp threshold
        self.reward = -energy-10000000*temp_diff

        done == True if self.step_cntr % self.episode_length == 0 else False

        self.step_cntr += 1

        return self.observation_, self.step_idx_, self.reward, done

    def reset():
        self.step_idx = np.random.randint(self.data.shape[0], size=1)
        return self.data[idx,:], self.step_idx
