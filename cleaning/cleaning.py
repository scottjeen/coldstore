import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
#% matplotlib inline
from codecarbon import track_emissions
e_path = "/Users/ScottJeen/OneDrive - University of Cambridge/Admin/phd_emissions"
pd.set_option('display.max_columns', None)

import glob
import re
from datetime import datetime

path_jonluca = "/Users/ScottJeen/OneDrive - University of Cambridge/Research/Modelling/Emerson/jonluca_data/*.csv"
dfs = []

for fname in glob.glob(path_jonluca):
    df = pd.read_csv(fname, header=1)
    df = df.drop('Status',axis=1)

    # get feature name from path
    column_name = re.findall("[A-Z][A-Z].*\.",fname)

    # rename features
    df = df.rename(columns={df.columns[0]: 'Datetime', df.columns[1]: column_name[0]})

    # format datatime
    datetime_format = '%b %d, %Y %H:%M:%S %p'
    df['Datetime'] = pd.to_datetime(df['Datetime'], format=datetime_format)
    df = df.set_index('Datetime')

    # remove multiple entries at each timestep
    df = df[~df.index.duplicated(keep='first')]

    dfs.append(df)

# join columns on datatime and sort alphabetically
data = dfs[0].join(dfs[1:], how='inner')
data = data.sort_index(axis=1)

# drop faulty freezer temperature sensor feature
data = data.drop(['FREEZER SLAB TEMP DOOR INSIDE.'], axis=1)

# normalize humidity features
hum = data.columns.str.contains('HUMIDITY')
data.loc[:,hum] = data.loc[:,hum] / 100

# get power data from amps (power (kW) = amps * 600V / 1000)
amp = data.columns.str.contains('COMP AMP')
data.loc[:,amp] = data.loc[:,amp] * 600 / 1000

# rename columns
new_cols = pd.Series(data.columns).str.replace('AMP', 'POWER (kW)').str.replace('.','')
new_cols = list(new_cols)
data.columns = new_cols

# add total power feature
power_features = data.columns.str.contains('POWER')
data['TOTAL POWER (kW)'] = data.loc[:,power_features].sum(axis=1)

# add energy feature (assume power is constant for 3 minute period between datapoints)
data['TOTAL ENERGY (kWh)'] = data['TOTAL POWER (kW)'] * (60/3)

# create datetime convertor
def datetime_conv(df, hour_format='%H', date_format='%Y-%m-%d', hour='Hour', date='Date'):

    # format hour feature to padded 24h
    df[hour] = df[hour] - 1
    df[hour] = df[hour].astype(str)
    df[hour] = df[hour].str.pad(width=2, side='left', fillchar='0')

    # convert to datetime
    df[date] = pd.to_datetime(df[date], format=date_format)
    df[hour] = pd.to_datetime(df[hour], format=hour_format)

    x = []

    for index, row in df.iterrows():
        d = row.loc[date].date()
        t = row.loc[hour].time()
        x.append(dt.datetime.combine(d, t))

    df['Datetime'] = pd.Series(x)

    # drop old date and time cols
    df = df.drop([hour, date], axis=1)

    # set index to datetime
    df = df.set_index('Datetime')

    return df

import datetime as dt

# import elec data
path_elec = '/Users/ScottJeen/OneDrive - University of Cambridge/Research/Modelling/Emerson/elec_data/*.csv'

dfs_elec = []
files = [fname for fname in glob.glob(path_elec)]

# read hourly price data
hourly_price = pd.read_csv(files[0], header=3)
hourly_price = hourly_price.drop(hourly_price.columns[6:], axis=1)

# run datetime convertor
hourly_price = datetime_conv(hourly_price)

# rename columns
cols = hourly_price.columns
new_cols = {cols[0]: 'PRICE ($/MWH)',\
            cols[1]: '1 HOUR PRICE PREDICT',\
            cols[2]: '2 HOUR PRICE PREDICT',\
            cols[3]: '3 HOUR PRICE PREDICT'
           }

hourly_price = hourly_price.rename(new_cols, axis=1)

# read elec supply data
hourly_supply = pd.read_csv(files[1])

# run datetime convertor
hourly_supply = datetime_conv(hourly_supply, date_format='%d/%m/%Y')

# rename columns
hourly_supply = hourly_supply.rename({'Total Output': "TOTAL SUPPLY_MW",\
                                     'NUCLEAR': 'NUCLEAR_MW',\
                                      'GAS': 'GAS_MW',\
                                      'HYDRO': 'HYDRO_MW',\
                                      'WIND': 'WIND_MW',\
                                      'SOLAR': 'SOLAR_MW',\
                                      'BIOFUEL': 'BIOFUEL_MW'
                                     },\
                                     axis=1)

# cache timeseries index
index = data.index

# merge jonluca and prices
data = data.merge(hourly_price,\
                  left_on=[data.index.date, data.index.hour],\
                  right_on=[hourly_price.index.date, hourly_price.index.hour],\
                  how='left'
                  ).set_index(index) # keep 3 minute datetime index

data = data.drop(['key_0', 'key_1'], axis=1)

# merge jonluca/prices and supply
data = data.merge(hourly_supply,\
                  left_on=[data.index.date, data.index.hour],\
                  right_on=[hourly_supply.index.date, hourly_supply.index.hour],\
                  how='left'
                  ).set_index(index) # keep 3 minute datetime index

data = data.drop(['key_0', 'key_1'], axis=1)

# create grid emission features
gas_intensity = 400 # kg/MWh

data['GRID EMISSION INTENSITY_kg/MWh'] = (data['GAS_MW'] / data['TOTAL SUPPLY_MW']) * gas_intensity
data['GRID EMISSIONS_kgs'] = data['GRID EMISSION INTENSITY_kg/MWh'] * (3/60) # 3 minute intervals

print(data.head())
