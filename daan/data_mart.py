#########################################
#                                       #
#               Group 24                #
#               (2018)                  #
#           Vrije Universiteit          #
#          LambdaMART data              #
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
process dta for training a LabmdaMart model
"""




######################################### dataset function ###################################



# devide data in three sets
def data_sets(filename, col, nrows, k=10):
    
    data = pd.read_csv(filename, skiprows=(1,2), nrows=nrows)
    index = pd.DataFrame(data['srch_id'].unique()).sample(n = len(data['srch_id'].unique()))
    index1 = index[0: int(len(index)*(1/3.0) )]
    index2 = index[ int(len(index)*(1/3.0)) : int(len(index)*(2/3.0)) ]
    index3 = index[int(len(index)*(2/3.0)):]
    data['booking_bool'] = 5*data['booking_bool']
    data['calc'] = data[['click_bool','booking_bool']].apply(np.max, axis=1)
    #data['k'] = data.groupby('srch_id')['srch_id'].transform(lambda x: np.size(x))
    #data = data[data['k']>k].copy()
    
    df1 = data[data['srch_id'].isin( list(index1[0]) )]
    df2 = data[data['srch_id'].isin( list(index2[0]) )]
    df3 = data[data['srch_id'].isin( list(index3[0]) )]

    # make them equal size
    min_size = min(min(len(df1), len(df2)), len(df3))
    df1 = df1[0:min_size]
    df2 = df2[0:min_size]
    df3 = df3[0:min_size]
    
    Ty = df1['calc']
    TX = df1[col] 
    Tqids = df1['srch_id']

    Vy = df2['calc']
    VX = df2[col] 
    Vqids = df2['srch_id']
    
    Ey = df3['calc']
    EX = df3[col] 
    Eqids = df3['srch_id']
    
    return Ty, TX, Tqids, Vy, VX, Vqids, Ey, EX, Eqids


# fill data with means and if not possible with zeros
def fill_data(Ty, TX, Tqids, Vy, VX, Vqids, Ey, EX, Eqids):
    TX.fillna(TX.mean(),inplace = True)
    VX.fillna(VX.mean(),inplace = True)
    EX.fillna(EX.mean(),inplace = True)

    TX.fillna(0,inplace = True)
    VX.fillna(0,inplace = True)
    EX.fillna(0,inplace = True)
    
    return Ty, TX, Tqids, Vy, VX, Vqids, Ey, EX, Eqids


# transform data to numpy array
def to_array(Ty, TX, Tqids, Vy, VX, Vqids, Ey, EX, Eqids):
    Ty = np.array(Ty)
    Vy = np.array(Vy)
    Ey = np.array(Ey)
    TX = np.array(TX)
    VX = np.array(VX)
    EX = np.array(EX)
    Tqids = np.array(Tqids)
    Vqids = np.array(Vqids)
    Eqids = np.array(Eqids)
    return Ty, TX, Tqids, Vy, VX, Vqids, Ey, EX, Eqids


# test data
def test_data(filename, col):
    
    data = pd.read_csv(filename)
    TX = data[col] 
    Tqids = data['srch_id']
    Tprop = data['prop_id']
    return TX, Tqids, Tprop


# fill test data
def fill_data_test(TX):
    TX.fillna(TX.mean(),inplace = True)
    TX.fillna(0,inplace = True)
    return TX
    

# transform test data to numpy array
def to_array_test(TX, Tqids):
    TX = np.array(TX)
    Tqids = np.array(Tqids)
    return TX, Tqids
