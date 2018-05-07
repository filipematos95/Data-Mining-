#########################################
#                                       #
#               Group 24                #
#               (2018)                  #
#           Vrije Universiteit          #
#           test data preprocess        #
#                                       #
#########################################

import pandas as pd
import numpy as np
import copy


"""
File reads in testdata andprocess it to data for specific countries to predict for
"""

test = pd.read_csv('test_set_VU_DM_2014.csv')
country_group = test.groupby('prop_country_id')

for name, group in country_group:

    # normal stuff (we excluded 'site_id' form train set somehow)
    normal = ['srch_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id',
       'prop_id', 'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2',
       'prop_log_historical_price', 'price_usd', 'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
       'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'srch_query_affinity_score',
       'orig_destination_distance', 'random_bool']

    df = group[normal].copy()
            
    rate = ['comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate', 'comp5_rate',
    	 'comp6_rate', 'comp7_rate', 'comp8_rate']
    
    inv = ['comp1_inv', 'comp2_inv', 'comp3_inv', 'comp4_inv', 'comp5_inv', 'comp6_inv', 
            'comp7_inv', 'comp8_inv']
    
    diff = ['comp1_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff',
            'comp4_rate_percent_diff', 'comp5_rate_percent_diff', 'comp6_rate_percent_diff',
            'comp7_rate_percent_diff', 'comp8_rate_percent_diff']
    
    df['comp_rate'] =  group[rate].mean(axis = 0)                # price competition (1=better, 0=none, -1=bad)
    df['comp_inv'] = group[inv].mean(axis = 0)                  # availibility competition (1=better, 0=same)
    df['comp_rate_percent_diff_min'] = group[diff].min(axis = 0) # % differences price competition
    df['comp_rate_percent_diff_max'] = group[diff].max(axis = 0)

    meta1 = ['d', 'd','c', 'c','d']
    meta2 = ['d', 'c', 'c','d', 'c','c']
    meta3 =['c','c', 'd','d', 'c', 'c'] # discrete (d), continuous (c), string (s)
    meta4 = ['c', 'c','c', 'd','c','c', 'd']    
    meta5 = ['c', 'c','c', 'c', 'd', 'd']

    df['booked'] = np.nan
    df['clicked'] = np.nan
    index = pd.DataFrame(meta1 + meta2 + meta3 + meta4 + meta5, index= df.columns).T
    extra = index.copy()
    extra[extra != np.nan] = np.nan
    result = pd.concat([index, extra, df], axis = 0)

    result.to_csv('test_countries/' + str(name) + '.csv', index =False)


