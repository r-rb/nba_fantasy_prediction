import pickle
import tensorflow as tf
import numpy as np
import pprint as pp
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, TimeDistributed, GRU
from keras.callbacks import TensorBoard
from keras import backend as K
from time import time 

def load_in(path):
    return pickle.load(open(path,'rb'))

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot ) )


def get_model(model):
    model.save('../my_model.h5')


def shape_data( filename):
    player = load_in(filename)
    X = player['X'][:-1]
    y = player['y'][:-1]

    scaler = StandardScaler()

    X = scaler.fit_transform(X)
    train_ratio = 2/4

    train_n = int(len(y)*train_ratio)

    X_train = X[:train_n]
    y_train = y[:train_n]
    X_test = X[train_n:]
    y_test = y[train_n:]

    print(np.shape(X))
    print(np.shape(X_train))
    print(np.shape(X_test))
    print(np.shape(y_train))

    X_train = np.reshape(X_train, (X_train.shape[0],1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1,X_test.shape[1]))

    return (X_train, X_test, y_train, y_test, train_n,y)




def run_model(X_train, X_test, y_train, y_test, input_output_neurons, hidden_neurons, train_n,y):
        
    # Add layers (perhaps add this to a loop)
    model = Sequential()
    model.add(GRU(input_output_neurons, unroll=False, return_sequences=True))
    model.add(GRU(hidden_neurons, input_shape=(hidden_neurons,), return_sequences=False))
    model.add((Dense(1)))
    model.add(Activation("linear"))

    # Choose the loss and optimisation scheme.
    model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['mae','mse','accuracy',coeff_determination])

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    x = model.fit(X_train, y_train, epochs=3000, batch_size=20, validation_data= (X_test, y_test),shuffle = False,callbacks = [tensorboard])
    score = model.evaluate(X_test, y_test, batch_size=1)

    print(model.metrics_names)  
    print(score)    

    y_pred = model.predict(X_test)
    plt.plot(list(range(train_n,len(y))),y_pred,'r')
    plt.axhline(y = np.average(y))
    plt.plot(list(range(train_n,len(y))),y[train_n:],'b')
    plt.show()

input_output_neurons = 1
hidden_neurons = 50
filename = '../data/player_data/lebron_2018.pkl'

model = shape_data(filename)

run_model(model[0],model[1],model[2],model[3],input_output_neurons,hidden_neurons,model[4],model[5])
