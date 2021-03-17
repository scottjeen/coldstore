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
    def __init__(self, path, train_length=20, target_length=1, num_units=100, dropout=0.1, lr=0.005):
        self.data = pd.read_pickle(path)
        self.train_length = train_length
        self.target_length = target_length
        self.num_units = num_units
        self.dropout = dropout
        self.lr = lr

        # get cooler data
        cooler_cols = self.data.columns.str.contains('COOLER|DEW|OUTSIDE TEMP|OUTSIDE HUMIDITY|WIND|PRESSURE \(kPa\)', regex=True)
        action_cols = self.data.columns.str.contains('COOLER COMP POWER')
        cooler = self.data.iloc[:,cooler_cols]
        target = cooler.iloc[:,[14]]
        features = cooler

        self.n_actions = len(action_cols[action_cols==True])
        self.input_dims = len(self.data.columns)

        def normalise(df):
            vals = df.values
            transformer = MinMaxScaler(feature_range=(-1,1))
            transformer.fit(vals)
            norm = transformer.transform(vals)
            return norm, transformer

        self.f_norm, self.f_trans = normalise(features)
        self.t_norm, self.t_trans = normalise(target)

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
        es = EarlyStopping(monitor='val_loss', patience=5)
        self.model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test), callbacks=[es])


    def step(self.state, self.action):

        reward = function of temperature diff etc
        return self.state_, self.reward, done 

    def reset():
        return self.random.state
