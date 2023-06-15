from model import model, y_train, X_train, X_test, y_test
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
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

# Model accuracy
model.evaluate(X_test, y_test, verbose=1)

# Confusion Matrix
y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)
y_test = y_test.argmax(axis=1)

cm = confusion_matrix(y_test, y_pred)
print(cm)
# Visualize confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

model.save('my_model.h5')