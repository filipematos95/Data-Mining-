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


"""
features used for training a LabmdaMart model
- all
- importance (around 30)
- best_5
- best_10
- best_15
- best_20
- col old used model
"""

all_features = ['booking_bool',
 'click_bool',
 'gross_bookings_usd',
 'orig_destination_distance',
 'position',
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
 'site_id',
 'srch_adults_count',
 'srch_booking_window',
 'srch_children_count',
 'srch_destination_id',
 'srch_id',
 'srch_length_of_stay',
 'srch_query_affinity_score',
 'srch_room_count',
 'srch_saturday_night_bool',
 'visitor_hist_adr_usd',
 'visitor_hist_starrating',
 'visitor_location_country_id',
 'rate_sum',
 'inv_sum',
 'diff_mean',
 'rate_abs',
 'inv_abs',
 'visitor_hist_starrating_mean',
 'prop_id_mean',
 'visitor_hist_adr_usd_mean',
 'prop_location_score1_mean',
 'prop_location_score2_mean',
 'prop_log_historical_price_mean',
 'position_mean',
 'price_usd_mean',
 'promotion_flag_mean',
 'srch_destination_id_mean',
 'srch_length_of_stay_mean',
 'srch_booking_window_mean',
 'srch_adults_count_mean',
 'srch_children_count_mean',
 'srch_room_count_mean',
 'srch_saturday_night_bool_mean',
 'srch_query_affinity_score_mean',
 'orig_destination_distance_mean',
 'visitor_hist_starrating_std',
 'prop_id_std',
 'visitor_hist_adr_usd_std',
 'prop_location_score1_std',
 'prop_location_score2_std',
 'prop_log_historical_price_std',
 'position_std',
 'price_usd_std',
 'promotion_flag_std',
 'srch_destination_id_std',
 'srch_length_of_stay_std',
 'srch_booking_window_std',
 'srch_adults_count_std',
 'srch_children_count_std',
 'srch_room_count_std',
 'srch_saturday_night_bool_std',
 'srch_query_affinity_score_std',
 'orig_destination_distance_std',
 'visitor_hist_starrating_med',
 'prop_id_med',
 'visitor_hist_adr_usd_med',
 'prop_location_score1_med',
 'prop_location_score2_med',
 'prop_log_historical_price_med',
 'position_med',
 'price_usd_med',
 'promotion_flag_med',
 'srch_destination_id_med',
 'srch_length_of_stay_med',
 'srch_booking_window_med',
 'srch_adults_count_med',
 'srch_children_count_med',
 'srch_room_count_med',
 'srch_saturday_night_bool_med',
 'srch_query_affinity_score_med',
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

importance = [
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
 
best_5 = ['hotel_quality_1', 'hotel_quality_2', 'ump', 'price_usd', 'price_usd_med']
 
best_10 = ['hotel_quality_1', 'hotel_quality_2', 'ump', 'price_usd', 'price_usd_med',
    'per_fee', 'score1d2', 'orig_destination_distance', 'prop_location_score2', 'score2ma']

best_15 = ['hotel_quality_1', 'hotel_quality_2', 'ump', 'price_usd', 'price_usd_med',
    'per_fee', 'score1d2', 'orig_destination_distance', 'prop_location_score2', 'score2ma',
    'random_bool', 'total_fee', 'prop_location_score2_mean', 'prop_id', 'price_usd_mean']

best_20 = ['hotel_quality_1', 'hotel_quality_2', 'ump', 'price_usd', 'price_usd_med',
    'per_fee', 'score1d2', 'orig_destination_distance', 'prop_location_score2', 'score2ma',
    'random_bool', 'total_fee', 'prop_location_score2_mean', 'prop_id', 'price_usd_mean',
    'prop_location_score2_med', 'starrating_diff', 'prop_log_historical_price_mean',
    'promotion_flag_mean', 'prop_location_score2_std']
    
# this worked very well score (5.19)
col = ['hotel_quality_1', 'price_usd_med', 'prop_id', 'hotel_quality_2',
    'score2ma', 'score1d2', 'price_usd', 'total_fee', 'ump', 'prop_location_score2',
    'promotion_flag_mean', 'price_usd_mean', 'per_fee', 'prop_log_historical_price',
    'price_diff', 'promotion_flag', 'rate_sum', 'prop_log_historical_price_med',
    'prop_country_id', 'starrating_diff', 'prop_location_score2_mean']
