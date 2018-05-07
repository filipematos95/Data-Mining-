#########################################
#                                       #
#               Group 24                #
#               (2018)                  #
#           Vrije Universiteit          #
#               evaluation              #
#                                       #
#########################################

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy as copy


"""
predictions evaluated by Discounted cumulative gain measure (uses functions from sources indicated)
"""


###################################### does an evaluation of predictions ########################################


# should be 4 columns searchid, booked, clicked, pred  
def score(ex):
    ex['booked'] = 5*ex['booked']
    ex['points'] = ex[['booked', 'clicked']].apply(max, axis = 1) 
    ex[['booked', 'clicked']].apply(np.max, axis = 1)
    ex['ndcg'] = ex.sort_values(['pred'],ascending=True).groupby('Search_id')['points'].apply(ndcg_at_k)
    return ex[['Search_id', 'ndcg']].dropna()



# The below functions were taken from -> credits: https://gist.github.com/bwhite/3726239 (fixed version a bit)


# Returns Discounted cumulative gain
def dcg_at_k(r, k):
    
    r = np.asfarray(r)[:k]
    if r.size:

        if r.size > 1:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        else:
            return np.sum(r / np.log2(np.arange(2, r.size + 1)))
    return 0.


# returns Score is normalized discounted cumulative gain (ndcg)
def ndcg_at_k(r):
    k = len(r)
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max



###################################### read filename and output dataframe ########################################



if len(sys.argv) > 1: 
    filename = sys.argv[1]
else:
    print('plz specify filename, example is loaded instead')
    filename = 'ex.csv'
print('file will be written to score.csv')
    
pred = pd.read_csv(filename)

temp = score(pred)
print('The obtained average normalized discounted cumulative gain: ', np.mean(temp['ndcg']), ' (std = ', temp['ndcg'].std(), ').')
temp.to_csv('score.csv', index =False)

