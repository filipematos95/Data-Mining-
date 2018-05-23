#########################################
#                                       #
#               Group 24                #
#               (2018)                  #
#           Vrije Universiteit          #
#          LambdaMART score             #
#                                       #
#########################################

import pyltr
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math
import copy
import re
import numpy as np

"""
scoring of a LabmdaMart model
"""



################################### scoring functions #########################################


#### version 1
def compute(r):
    #k = min(len(r),5)
    k = len(r)
    return ndcg_at_k(r,k)

def dcg_at_k(r,k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg


#### version 2
def get_max_ndcg(k, *ins):
    '''This is a function to get maxium value of DCG@k. That is the DCG@k of sorted ground truth list. '''
    #print ins
    l = [i for i in ins]
    l = copy.copy(l[0])
    l.sort(None,None,True)
    #print l
    max = 0.0
    for i in range(k):
        #print l[i]/math.log(i+2,2)
        max += (math.pow(2, l[i])-1)/math.log(i+2,2)
        #max += l[i]/math.log(i+2,2)
    return max


def get_ndcg(s, k):
    '''This is a function to get ndcg '''
    z = get_max_ndcg(k, s)
    dcg = 0.0
    for i in range(k):
        #print s[i]/math.log(i+2,2)
        dcg += (math.pow(2, s[i])-1)/math.log(i+2,2)
        #dcg += s[i]/math.log(i+2,2)
    if z ==0:
        z = 1;
    ndcg = dcg/z
    #print "Line:%s, NDCG@%d is %f with DCG = %f, z = %f"%(s, k, ndcg,dcg, z)
    return ndcg


# function that returns in two ways the computed ndcg score
def scores(Epred, Eqids, Ey):
    
    X = pd.DataFrame([Epred, Eqids])
    X = X.T
    X. columns = ['prob', 'srch_id']
    X['points'] = Ey

    X_sort = X.sort_values(['srch_id', 'prob'],ascending=[True, False]) 
    
    # score 1
    X_sort['score'] = X_sort.groupby('srch_id').apply(lambda x: compute(x.points.values))
    print('score 1 = ', X_sort[['score']].dropna().mean())
    
    # score2
    X_sort['score2'] = X_sort.groupby('srch_id').apply(lambda x: get_ndcg(list(x.points.values),len(x)))
    print('score 2 = ', X_sort[['score2']].dropna().mean())
    
    return pd.DataFrame([X_sort['score'].dropna(),X_sort['score2'].dropna()]).T


