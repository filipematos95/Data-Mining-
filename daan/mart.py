#########################################
#                                       #
#               Group 24                #
#               (2018)                  #
#           Vrije Universiteit          #
#                LambdaMART             #
#                                       #
#########################################

#                   Credits pyltr
#               
#           source: https://github.com/jma127/pyltr
#
#           Copyright (c) 2015, Jerry Ma
#               All rights reserved.

import pyltr
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math
import copy
import re
import pickle
import numpy as np

"""
This file performs a LabmdaMart training of a model by the use of the package
pyltr shown above.
"""

from score import *
from data_mart import *
from features import *

###################################### loading and saving trained model #######################



# Save model in new folder
def save(model, foldername = "\sample"):
    try:
        pathfile = os.path.join(os.getcwd(), foldername)
        os.makedirs(pathfile)
        
    except:
        print("folder already exists")
    with open(os.path.join(pathfile, 'model'), 'wb') as output:  # Overwrites any existing file.
            pickle.dump(model, output)


# Load model from folder
def load(foldername):
    
    directory = os.path.join(os.getcwd(), foldername)

    try:
        with open(os.path.join(directory, 'model'), 'rb') as input:
            model = pickle.load(input)
    except:
        print("No such folder exists")    
    return model



####################################### clean results ############################

`

# prints stats of the model
def print_stats(model, Epred, Ey, Eqids, sets):
    print('Random ranking:', metric.calc_mean_random(Eqids, Ey))
    print('Our model:', metric.calc_mean(Eqids, Ey, Epred))
    print('number of estimators: ', model.estimators_fitted_)
    #print('feature importance', model.feature_importances_)
    print('train score', model.train_score_)
    print('ordered feature importance',
        pd.DataFrame([model.feature_importances_, sets[7].columns]).T.sort_values(0, ascending = False))



###################################### used models ##############################



### read data
filename = 'data/train.csv'
nrows = 2000000


################################################# log #####################################


# 1. first all features were used to train a model to abtain feature importance

### process data
sets = data_sets(filename, importance, nrows, 10)
sets = fill_data(*sets)
Ty, TX, Tqids, Vy, VX, Vqids, Ey, EX, Eqids = to_array(*sets)

### set up model
metric = pyltr.metrics.NDCG(k=10)

monitor = pyltr.models.monitors.ValidationMonitor(
    VX, Vy, Vqids, metric=metric, stop_after=100)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=1000,
    learning_rate=0.19,
    max_features=0.7,
    query_subsample=1.0,
    max_leaf_nodes=5,
    min_samples_leaf=10,
    verbose=1,
)

model.fit(TX, Ty, Tqids, monitor=monitor)

Epred = model.predict(EX)

### results
print_stats(model, Epred, Ey, Eqids, sets)
s = scores(Epred, Eqids, Ey)

# model 1 to determine importance
save(model, "importance")   
load('importance')

# 2.best 5 were used

### process data
sets = data_sets(filename, best_5, nrows, 10)
sets = fill_data(*sets)
Ty, TX, Tqids, Vy, VX, Vqids, Ey, EX, Eqids = to_array(*sets)


### set up model
metric = pyltr.metrics.NDCG(k=10)

monitor = pyltr.models.monitors.ValidationMonitor(
    VX, Vy, Vqids, metric=metric, stop_after=100)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=1000,
    learning_rate=0.19,
    max_features=0.7,
    query_subsample=1.0,
    max_leaf_nodes=5,
    min_samples_leaf=10,
    verbose=1,
)

model.fit(TX, Ty, Tqids, monitor=monitor)

Epred = model.predict(EX)

### results
print_stats(model, Epred, Ey, Eqids, sets)
s = scores(Epred, Eqids, Ey)

# model 1 to determine importance
save(model, "best_5")   
load('best_5')

# 3. best 10 were used

### process data
sets = data_sets(filename, best_10, nrows, 10)
sets = fill_data(*sets)
Ty, TX, Tqids, Vy, VX, Vqids, Ey, EX, Eqids = to_array(*sets)

### set up model
metric = pyltr.metrics.NDCG(k=10)

monitor = pyltr.models.monitors.ValidationMonitor(
    VX, Vy, Vqids, metric=metric, stop_after=100)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=1000,
    learning_rate=0.19,
    max_features=0.7,
    query_subsample=1.0,
    max_leaf_nodes=5,
    min_samples_leaf=10,
    verbose=1,
)

model.fit(TX, Ty, Tqids, monitor=monitor)

Epred = model.predict(EX)

### results
print_stats(model, Epred, Ey, Eqids, sets)
s = scores(Epred, Eqids, Ey)

# model 1 to determine importance
save(model, "best_10")   
load('best_10')


# 4. best 15 were used

### process data
sets = data_sets(filename, best_15, nrows, 10)
sets = fill_data(*sets)
Ty, TX, Tqids, Vy, VX, Vqids, Ey, EX, Eqids = to_array(*sets)

### set up model
metric = pyltr.metrics.NDCG(k=10)

monitor = pyltr.models.monitors.ValidationMonitor(
    VX, Vy, Vqids, metric=metric, stop_after=100)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=1000,
    learning_rate=0.19,
    max_features=0.7,
    query_subsample=1.0,
    max_leaf_nodes=5,
    min_samples_leaf=10,
    verbose=1,
)

model.fit(TX, Ty, Tqids, monitor=monitor)

Epred = model.predict(EX)

### results
print_stats(model, Epred, Ey, Eqids, sets)
s = scores(Epred, Eqids, Ey)

# model 1 to determine importance
save(model, "best_15")   

# 5. best 20 were used

### process data
sets = data_sets(filename, best_20, nrows, 10)
sets = fill_data(*sets)
Ty, TX, Tqids, Vy, VX, Vqids, Ey, EX, Eqids = to_array(*sets)

### set up model
metric = pyltr.metrics.NDCG(k=10)

monitor = pyltr.models.monitors.ValidationMonitor(
    VX, Vy, Vqids, metric=metric, stop_after=100)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=1000,
    learning_rate=0.19,
    max_features=0.7,
    query_subsample=1.0,
    max_leaf_nodes=5,
    min_samples_leaf=10,
    verbose=1,
)

model.fit(TX, Ty, Tqids, monitor=monitor)

Epred = model.predict(EX)

### results
print_stats(model, Epred, Ey, Eqids, sets)
s = scores(Epred, Eqids, Ey)

# model 1 to determine importance
save(model, "best_20")   



