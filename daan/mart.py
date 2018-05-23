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

### process data
sets = data_sets(filename, col, nrows, 10)
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

#### setup new model with 5best features





################################################# log #####################################


# 1. first all features were used to train a model to abtain feature importance

# 2.best 5 were used

### process data
best_5 = ['hotel_quality_1', 'hotel_quality_2', 'ump', 'price_usd', 'price_usd_med']
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
best_10 = ['hotel_quality_1', 'hotel_quality_2', 'ump', 'price_usd', 'price_usd_med',
    'per_fee', 'score1d2', 'orig_destination_distance', 'prop_location_score2', 'score2ma']
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


################################################# crap #####################################


col = [
 'orig_destination_distance',
 'price_usd',
 'promotion_flag',
 'prop_brand_bool',
 'prop_country_id',
 'prop_id',
 'prop_location_score1',
 'prop_location_score2',
 'prop_log_historical_price',
 'prop_review_score',
 'prop_starrating',
 'random_bool',
 'rate_sum',
 'inv_sum',
 'diff_mean',
 'rate_abs',
 'inv_abs',
 'prop_location_score1_mean',
 'prop_location_score2_mean',
 'prop_log_historical_price_mean',
 'price_usd_mean',
 'promotion_flag_mean',
 'orig_destination_distance_mean',
 'prop_location_score1_std',
 'prop_location_score2_std',
 'prop_log_historical_price_std',
 'price_usd_std',
 'promotion_flag_std',
 'orig_destination_distance_std',
 'prop_location_score1_med',
 'prop_location_score2_med',
 'prop_log_historical_price_med',
 'price_usd_med',
 'promotion_flag_med',
 'orig_destination_distance_med',
 'ump',
 'price_diff',
 'starrating_diff',
 'per_fee',
 'prop_starrating_mean',
 'prop_starrating_std',
 'prop_starrating_med',
 'score2ma',
 'total_fee',
 'score1d2',
 'hotel_quality_1',
 'hotel_quality_2'] 

# 2 


 # all_features = ['srch_id', 'site_id', 'visitor_location_country_id',
#   'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id',
#   'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool',
#   'prop_location_score1', 'prop_location_score2',
#   'prop_log_historical_price', 'position', 'price_usd', 'promotion_flag',
#   'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
#   'srch_adults_count', 'srch_children_count', 'srch_room_count',
#   'srch_saturday_night_bool', 'srch_query_affinity_score',
#   'orig_destination_distance', 'random_bool', 'click_bool',
#   'gross_bookings_usd', 'booking_bool', 'rate_sum', 'inv_sum',
#   'diff_mean', 'rate_abs', 'inv_abs']

# col = ['rate_sum', 'inv_sum','prop_starrating', 'prop_review_score', 'prop_brand_bool',
#     'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id', 
#     'prop_location_score1', 'prop_location_score2', 'prop_log_historical_price',
#     'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
#     'srch_adults_count', 'srch_children_count', 'srch_room_count']


############################################# old stuff #################################


# this model worked very good

# this worked very well score (5.19)
col = ['hotel_quality_1', 'price_usd_med', 'prop_id', 'hotel_quality_2',
    'score2ma', 'score1d2', 'price_usd', 'total_fee', 'ump', 'prop_location_score2',
    'promotion_flag_mean', 'price_usd_mean', 'per_fee', 'prop_log_historical_price',
    'price_diff', 'promotion_flag', 'rate_sum', 'prop_log_historical_price_med',
    'prop_country_id', 'starrating_diff', 'prop_location_score2_mean']
