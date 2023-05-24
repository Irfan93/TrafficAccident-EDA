# -*- coding: utf-8 -*-
"""
Created on Tue May 19 13:27:22 2020

@author: irfan
"""

#Libraries Required
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from matplotlib import pyplot
from math import sqrt
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, Model
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Dropout, Dense, LSTM, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)
import time
from datetime import datetime

# Change Directory
os.chdir(r"C:\Users\irfan\Desktop\SPEED")
#Load Temperature Data Train
Speeddf = read_csv('Speeddf.csv')
Speeddf.Date = pd.Series.to_string(Speeddf.Date)

Speeddf.Date = '-'.join([Speeddf.Date[:0], Speeddf.Date[1:2], Speeddf.Date[3:]])



Speeddf.Date = pd.Series(Speeddf.Date, dtype="str")
Speeddf.Date = Speeddf.Date.str.zfill(8)
Speeddf.Date = time.strptime(Speeddf.Date, "%m%d%Y")

Speeddf['Date'] = datetime(year=int(Speeddf.Date[3:6]), month=int(Speeddf.Date[1:2]), day=int(Speeddf.Date[0:0]))
Speeddf.Date =  pd.to_datetime(Speeddf['Date'], format='%m%d%Y)



dfs = dict(tuple(Speeddf.groupby('Segment')))

Incidentdf = read_csv('Primary_incidents.csv')

dfs1 = dict(tuple(Incidentdf.groupby('tmcN')))

dfs2 = dict(tuple(Incidentdf.groupby('tmcP')))


