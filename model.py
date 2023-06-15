import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from pickle import dump, load

# Read Data
dataFrame = pd.read_csv('seattle-weather.csv')
print(dataFrame.head())  # see the first 5 rows
print(dataFrame.isnull().sum())  # will count number of rows
dataFrame.dropna(inplace=True)  # drop any NaN just in case
dataFrame["date"] = pd.to_datetime(dataFrame["date"])  # clean date to datetime format
dataFrame = dataFrame.set_index(['date'])
print(dataFrame)

# Assign X and y Dataframe
pd.set_option('display.max_columns', None)  # print all columns
X = dataFrame.iloc[:, 0:5]  # all rows and cols 0-4
y = dataFrame.iloc[:, -1]  # all rows and last col
print(X.head())
print(y.head())

#  Convert text to categorical data and to numpy array
X = pd.get_dummies(X)
y = pd.get_dummies(y)  # one hot encoding categorical values
X = X.values  # convert from pandas to numpy array
y = y.values
print("Converted to Categorical X\n", X)
print("Converted to Categorical y\n", y)

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

#  Scaler
dump(scaler, open('scaler.pkl', 'wb'))  # save the scaler
print("Scaled X\n", X)
print("Scaled y\n", y)