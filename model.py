import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam
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

print("X Shape: ", X.shape)
print("y Shape: ", y.shape)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model ini

model = Sequential()
model.add(Dense(9,activation='relu'))
model.add(Dense(45,activation='relu')) # 9 x 5
model.add(Dense(45,activation='relu'))
model.add(Dense(5,activation='softmax'))

adam = Adam(learning_rate=0.001) # you may have to change learning_rate, if the model does not learn.
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# use loss = 'categorical_crossentropy' for multi-class classification.
# For classification only: use metrics = ['accuracy']. It shows successful predictions / total predictions
