# data cleaning function
def cleaning(data, train_length, target_length, LSTM=True, off=False, on=False):

  # get cooler data
  cooler_cols = data.columns.str.contains('COOLER|DEW|OUTSIDE TEMP|OUTSIDE HUMIDITY|WIND|PRESSURE \(kPa\)', regex=True)
  cooler = data.iloc[:,cooler_cols]
  target_cols = ['COOLER TEMP 1', 'COOLER TEMP 2']
  target = cooler[target_cols]
  bad_cols = ['COOLER SLAB TEMP OUTSIDE', 'COOLER SLAB TEMP INSIDE 1', 'COOLER SLAB TEMP INSIDE 2', 'COOLER SOIL TEMP INSIDE 1', 'OUTSIDE TEMP (oC)', 'OUTSIDE HUMIDITY (%)', 'WIND DIR (DEGREES)', 'WIND (km/h)']
  cooler = cooler.drop(cooler[bad_cols], axis=1)

  # get target matrix of temperatures
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

  f_norm, f_trans = normalise(features)
  t_norm, t_trans = normalise(target)

  # define sequence splitting function for LSTM
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

  if LSTM:
    X, y = split_sequence(f_norm, t_norm, train_length, target_length)

    # reshape into [samples, steps, features]
    X = X.reshape((X.shape[0], X.shape[1], X[0].shape[1]))
    y = y.reshape((y.shape[0], y.shape[1] * y[0].shape[1]))

  from sklearn.model_selection import train_test_split
  # split into train/test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

  if on:
    X[:,:,[0,1,2,3,4,5]] = 1
    return X
  elif off:
    X[:,:,[0,1,2,3,4,5]] = -1
    return X

  return X_train, X_test, y_train, y_test, t_trans, target, X
