#########################################
#                                       #
#               Group 24                #
#               (2018)                  #
#           Vrije Universiteit          #
#            evaluate widget            #
#                                       #
#########################################

import numpy as np
import itertools
import pandas as pd
import numpy as np
import sys
import copy as copy
import Orange 


"""
widget that evaluates ndcg score of a bunch of searchids
"""


# should be 5 columns searchid, clicked, booked, lr, prop_id  
def score(ex):
    book = 5*ex['booked']
    click = ex['clicked']
    ex['points'] = ex[['booked', 'clicked']].apply(max, axis = 1) 
    ex[['booked', 'clicked']].apply(np.max, axis = 1)
    ex = ex.sort_values(['srch_id', 'pred'],ascending=[True, False]) 
    ex['score'] = ex.groupby('srch_id').apply(lambda x: ndcg_at_k(x.points.values))
    return ex


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



test = in_data.copy()
a = pd.DataFrame(np.array(test), columns = ['booked', 'clicked', 'pred','srch_id', 'prop_id'])
out_data = Orange.data.Table(score(a).dropna()[['srch_id', 'score']])


