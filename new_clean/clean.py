#########################################
#                                       #
#               Group 24                #
#               (2018)                  #
#           Vrije Universiteit          #
#               cleaning                #
#                                       #
#########################################

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

"""
features added 
"""
if len(sys.argv) > 1:
    filename = sys.argv[1]
else: 
    "Please specify the filename"
    filename = "TEST"

def combine(filename,chunksize = 1000000):
    pool = Pool(4)

    df_split = pd.read_csv(filename,chunksize = chunksize,engine = 'c')
    result = pd.concat(pool.map(clean,df_split))
    #test = pd.read_csv('/Volumes/FILIPE1/Data mIning/Data Mining VU data/test_set_VU_DM_2014.csv')
    #total = pd.concat([train, test], axis = 0)

    pool.close()
    pool.join()

    return result

def clean(total):

    rate = ['comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate', 'comp5_rate', 
            'comp6_rate', 'comp7_rate', 'comp8_rate'] 
            
    inv = ['comp1_inv', 'comp2_inv', 'comp3_inv', 'comp4_inv', 'comp5_inv', 'comp6_inv',  
        'comp7_inv', 'comp8_inv'] 
    
    diff = ['comp1_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff', 
        'comp4_rate_percent_diff', 'comp5_rate_percent_diff', 'comp6_rate_percent_diff', 
        'comp7_rate_percent_diff', 'comp8_rate_percent_diff'] 
    
    total['rate_sum'] = total[rate].sum(axis=1)
    total['inv_sum'] = total[rate].sum(axis=1)
    total['diff_mean'] = total[rate].mean(axis=1)
    total['rate_abs'] = total[rate].min(axis=1)
    total['inv_abs'] = total[inv].min(axis=1)

    total.drop(rate + inv + diff, axis = 1, inplace = True)
    total.drop('date_time', axis = 1, inplace = True)
    
    num = ['visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_location_score1',
     'prop_location_score2', 'prop_log_historical_price', 'price_usd',
     'promotion_flag', 'srch_length_of_stay', 'srch_booking_window',
     'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
     'srch_query_affinity_score', 'orig_destination_distance', 'prop_starrating']
    
    num_mean = [x + "_mean" for x in num]
    total[num_mean] = total.groupby("prop_id")[num].transform('mean')
    
    num_std = [x + "_std" for x in num]
    total[num_std] = total.groupby("prop_id")[num].transform('std')
    
    num_med = [x + "_med" for x in num]
    total[num_med] = total.groupby("prop_id")[num].transform('median')

    total['ump'] = np.exp(total['prop_log_historical_price']) - total['price_usd']
    total['price_diff'] = total['visitor_hist_adr_usd_mean'] - total['price_usd_mean']
    total['starrating_diff'] = total['visitor_hist_starrating_mean'] - total['prop_starrating_mean']
    total['per_fee'] = total['price_usd'] * total['srch_room_count'] / (total['srch_adults_count'] + total['srch_children_count'])
    total['score2ma'] = total['prop_location_score2'] * total['srch_query_affinity_score_mean']
    total['total_fee'] = total['price_usd'] * total['srch_room_count']
    total['score1d2'] = (total['prop_location_score2'] + 0.0001) / (total['prop_location_score1'] + 0.0001)
    total['hotel_quality_1'] = total.groupby('prop_id')['click_bool'].transform('mean')
    total['hotel_quality_2'] = total.groupby('prop_id')['booking_bool'].transform('mean')

    return total

total = combine(filename)
total.to_csv(filename[:-4]+'_clean.csv',index = False)