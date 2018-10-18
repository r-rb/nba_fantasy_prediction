import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
import datetime as dt
import pprint as pp
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from simulated_annealing.optimize import SimulatedAnneal
from sklearn.metrics import r2_score

def load_in(path):
    return pickle.load(open(path,'rb'))
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot ) )
def save_model(model):
    model.save('../my_model.h5')
#
def my_r2(estimator,X_not,y_not,X=X_valid,y=y_valid):
    return r2_score(y_valid,estimator.predict(X))


filename = './train_data_FINAL.pkl'
train_data = load_in(filename)
feature_matrix = train_data['input']
output = train_data['output']

training_ratio = 0.8
n = int(len(output) * training_ratio)
X_train,X_valid,y_train,y_valid = feature_matrix[:n],feature_matrix[n:],output[:n],output[n:]

print('Training on ', len(X_train), ' samples and validating on ', len(X_valid), ' samples.' )
rfr = RandomForestRegressor()

param_grid = {"max_features": np.arange(20,720,50),
                "min_samples_split": np.arange(500,10000,500),
                "n_estimators": [100, 200, 300],
                "max_depth": np.arange(50,1050,100),
                "random_state": [0]}
combs =1 
for k,v in param_grid.items():
    combs *= len(v)
print(combs)

sa = SimulatedAnneal(rfr, param_grid, T=1000.0, T_min=0.00001, alpha=0.9,
                         verbose=True,scoring= my_r2,n_jobs = 3,max_runtime =10800)
sa.fit(X_train, y_train)

print('\n Best Regressor:')
print(sa.best_score_, sa.best_params_)
optimized_rfr = sa.best_estimator_

pickle.dump(optimized_rfr, open("./best/rfr_model.pkl", 'wb'))
pickle.dump(sa.best_params_, open("./best/rfr_params.pkl", 'wb'))