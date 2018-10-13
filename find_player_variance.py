import pickle
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import pprint as pp
import matplotlib.pyplot as plt 

def load_in(path):
    return pickle.load(open(path,'rb'))


def order_by_variance():
    player_data = load_in('data/player_game_table.pkl')
    player_ids = load_in('data/player_vs_table.pkl')

    d = {'player_id': [], 'var': []}
 

    k = []
    pp.pprint(player_ids.columns.values)
    for index, row in player_ids.iterrows():
        x = player_data[player_data['player_id'] == row['player_vs_id']]    
        z = x['fantasy_points']
        l = z.var()    
        d['player_id'].append(row['player_vs_id'])
        d['var'].append(l)

    df = pd.DataFrame(data = d)  
    df.to_pickle('data/player_variance')      


