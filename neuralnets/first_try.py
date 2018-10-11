import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pprint as pp
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation

def load_in(path):
    return pickle.load(open(path,'rb'))

lebron = load_in('../data/player_data/lebron_2018.pkl')

X = lebron['X'][:-1]
y = lebron['y'][:-1]

scaler = StandardScaler()

X = scaler.fit_transform(X)
train_ratio = 2/3

train_n = int(len(y)*train_ratio)

# Define a Sequential Model
model = Sequential()


# Add layers (perhaps add this to a loop)
model.add(Dense(32, activation='relu', input_dim=95))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))

# Choose the loss and optimisation scheme.
model.compile(loss='mean_squared_error',
              optimizer='adagrad',
              metrics=['mae','mse'])

X_train = X[:train_n]
y_train = y[:train_n]
X_test = X[train_n:]
y_test = y[train_n:]

print(np.shape(X))
print(np.shape(X_train))
print(np.shape(X_test))
model.fit(X_train, y_train, epochs=1, batch_size=1)
score = model.evaluate(X_test, y_test, batch_size=1)

print(model.metrics_names)
print(score)




