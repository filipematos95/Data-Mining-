#########################################
#                                       #
#               Group 24                #
#               (2018)                  #
#           Vrije Universiteit          #
#       Prepare's test/train set        #
#                                       #
#########################################

import pandas as pd
import random 
import numpy as np
import sys
import copy as copy


"""
File split a train and test set by search id partition
"""


def split(file_train_in, file_test_in, fraction):
    
    # read in the data
    train = pd.read_csv(file_train_in, low_memory=False)
    test = pd.read_csv(file_test_in, low_memory=False)

    # train set
    grouped = train.iloc[2:,:].groupby('booked')
    min_group = round(min(len(grouped.get_group('1.0')),len(grouped.get_group('0.0')) ) * fraction)
    booked = grouped.get_group('0.0').sample(n = min_group)
    nonbooked = grouped.get_group('1.0').sample(n = min_group)
    train_set = pd.concat([booked, nonbooked], axis = 0)
    train_set = train_set.sample(n = len(train_set))
    index = train.iloc[0:2,:].copy()
    
    # test set to do validations on
    temp = test.iloc[2:,:].copy()
    test_set = temp.loc[~ temp['srch_id'].isin(train_set['srch_id'].unique())].copy()
    
    test_set = pd.concat([index, test_set], axis = 0)
    train_set = pd.concat([index, train_set], axis = 0)

    return train_set, test_set

fraction = 0.9

if len(sys.argv) > 2: 
    file_in_train = sys.argv[1]
    file_in_test = sys.argv[2]
    if len(sys.argv) > 3:
        fraction =  sys.argv[3]
    print('sample data will be written to /sample')
else:
    print('Please specify filename in train and filename in test')
    quit()

train, test = split(file_train_in, file_test_in, fraction)
train.to_csv('sample/train_sample.csv', index = False)
test.to_csv('sample/test_sample.csv', index= False)



