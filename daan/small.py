#########################################
#                                       #
#               Group 24                #
#               (2018)                  #
#           Vrije Universiteit          #
#             asmall sample             #
#                                       #
#########################################

import pandas as pd
import random 
import numpy as np
import sys
import copy as copy


"""
take little set of: split a train and test set by search id partition
"""

nrows = 10000
skiprows = 0

if len(sys.argv) > 2: 
    file_in_train = sys.argv[1]
    file_in_test = sys.argv[2]
    if len(sys.argv) > 3:
        nrows =  sys.argv[3]
    if len(sys.argv) > 4:
        skiprows =  sys.argv[4]
else:
    print('Please specify filename in train and filename in test')
    quit()

    
train = pd.read_csv(file_in_train, nrows= 10000, skiprows = range(2,skiprows), low_memory=False)
test = pd.read_csv(file_in_test, nrows= 10000, skiprows = range(2,skiprows), low_memory=False)

train.to_csv('train_sample_small.csv', index = False)
test.to_csv('test_sample_small.csv', index= False)

