import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
import datetime as dt
import pprint as pp
import pandas as pd
import seaborn as sns
from scipy.stats.stats import pearsonr
sns.set()

X = np.load("./notebooks/X.npy")
y = np.load("./notebooks/y.npy")
y = y[:-1]
X = X[:-1]

past_n = 5
features = ['ast','blk','dreb','fg3_pct','fg3a','fg3m','fg_pct','fga','fgm','ft_pct',
            'fta','ftm','min','oreb','pf','plus_minus','pts','reb','stl','to','days_since_injury','fantasy_points']

for g in range(len(features)):
    plt.figure(g)
    subplot_num = 231
    pearsons = []
    for i in range(past_n):
        last_i = X[:,i::past_n]
        plt.subplot(subplot_num)
        plt.title("{} from the {}-th game".format(features[g],i))
        plt.plot(last_i[:,g],y,'bo')
        pearsons.append(pearsonr(last_i[:,g],y)[0])
        subplot_num+=1
    plt.subplot(subplot_num)
    plt.bar(list(range(past_n)),pearsons)
plt.show()