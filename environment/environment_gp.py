import pandas as pd
import numpy as np
import tensorflow as tf
import gpflow
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data_path = '/Users/ScottJeen/OneDrive - University of Cambridge/research/Modelling/emerson_data/env'

# read in the pickled data
data = pd.read_pickle(data_path)

# cleaning function
def cleaning(data):
   # get cooler data
  cooler_cols = data.columns.str.contains('COOLER|DEW|OUTSIDE TEMP|OUTSIDE HUMIDITY|WIND|PRESSURE \(kPa\)', regex=True)
  cooler = data.iloc[:,cooler_cols]

  # get target matrix of temperatures
  target = cooler.iloc[:,[14, 15]]
  features = cooler

  def normalise(df):
    from sklearn.preprocessing import MinMaxScaler
    # retrieve values
    vals = df.values

    # instantiate transformer
    transformer = MinMaxScaler(feature_range=(-1,1))

    # fit transformer
    transformer.fit(vals)

    # normalise
    norm = transformer.transform(vals)

    return norm, transformer

  X, X_trans = normalise(features)
  y, y_trans = normalise(target)

  # convert to np arrays
  X, y = np.asarray(X), np.asarray(y)

  return X, y, X_trans, y_trans

# clean the data
X, y, X_trans, y_trans = cleaning(data)

def create_model(X, y):

    # build kernel
    k = gpflow.kernels.Matern52()

    # build model
    model = gpflow.models.GPR(data=(X, Y), kernal=k, mean_function=None) # mean = 0
