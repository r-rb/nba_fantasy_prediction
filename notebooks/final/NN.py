import pickle
import tensorflow as tf
import numpy as np
import pprint as pp
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LeakyReLU
from keras.callbacks import TensorBoard, EarlyStopping
from keras import backend as K
from time import time 
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import TensorBoard, EarlyStopping
from time import time 

def create_model(h1_neuron=1, h2_neuron = 1,optimizer='adam'):
    model = Sequential()
    # Add layers (perhaps add this to a loop)
    model.add(Dense(h1_neuron, input_shape=(720,), activation='relu'))
    model.add(Dense(h2_neuron, activation='relu'))
    model.add(Dense(1, activation='relu'))
    # Choose the loss and optimisation scheme.
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mae','mse'])
    # fix random seed for reproducibility
    return model
seed = 0
np.random.seed(seed)
train_ratio = 1/2

train_n = int(len(y)*train_ratio)

# load dataset
# split into input (X) and output (Y) variables
X_train = X[:train_n]
y_train = y[:train_n]
X_test = X[train_n:]
y_test = y[train_n:]

# create model

modelA = KerasRegressor(build_fn=create_model ,verbose=0)

batch_size = [1024]
epochs = [50]
neurons = [10, 15, 20, 25, 30, 35, 40, 45, 50]

param_grid={"batch_size": batch_size, "epochs": epochs, "neurons": neurons, }

# define the grid search parameters

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
grid = GridSearchCV(estimator=modelA, param_grid=param_grid,scoring='r2', n_jobs=1)
grid_result = grid.fit(X[:train_n], y[:train_n], validation_data= (X_test, y_test),callbacks = [tensorboard, earlystopping])

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))