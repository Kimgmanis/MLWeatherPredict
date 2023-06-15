from model import model, y_train, X_train
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam
from pickle import dump, load

# Train
model.fit(X_train, y_train, epochs=100, verbose=1)

# Loss vs epochs
loss = model.history.history['loss']
sns.lineplot(x=range(len(loss)), y=loss)
