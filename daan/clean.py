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
import matplotlib.pyplot as plt
import copy as copy

"""
Data is cleaned by imputing missing values and apply transformations on features.
    - obtain statistic from rates, invs and diffs
    - remove gross_bookings_usd, date_time and rates, invs and diffs
    - visitor_hist_starrating that are zero and nan -> mean
    - visitor_hist_adr_usd that are nan and zero -> median -> than take squareroot
    - srch_query_affinity_score that are nan -> mean -> than take exponential
    - orig_destination_distance that are nan -> mean -> than take thrid power squar
    - srch_booking_window -> log
    - srch_length_of_stay -> sqrt
    - price_usd -> sqrt if price_usd > 100 mean((price_usd<1000)) else
    - diff_mean -> log
"""

def impute(df):
    
    df = df.copy()
    df[df['visitor_hist_starrating']==0]['visitor_hist_starrating'] = df['visitor_hist_starrating'].mean()
    df['visitor_hist_starrating'] = df['visitor_hist_starrating'].fillna(df['visitor_hist_starrating'].mean())
    df[df['visitor_hist_adr_usd']==0]['visitor_hist_adr_usd'] = df['visitor_hist_adr_usd'].median()
    df['visitor_hist_adr_usd'] = df['visitor_hist_adr_usd'].fillna(df['visitor_hist_adr_usd'].median())
    df['visitor_hist_adr_usd'] = np.sqrt(df['visitor_hist_adr_usd'])
    df['srch_query_affinity_score'] = df['srch_query_affinity_score'].fillna(df['srch_query_affinity_score'].mean())
    df['srch_query_affinity_score'] = np.exp(df['srch_query_affinity_score'])
    df['orig_destination_distance'] = df['orig_destination_distance'].fillna(df['orig_destination_distance'].mean())
    df['orig_destination_distance'] = np.power(df['orig_destination_distance'], 1.0/3 )
    df['srch_booking_window'] = np.log(df['srch_booking_window']+0.1)
    df['srch_length_of_stay'] = np.sqrt(df['srch_length_of_stay'])
    df.loc[df['price_usd']<1000,'price_usd'] = np.sqrt(df[df['price_usd']<1000]['price_usd'])
    df.loc[df['price_usd']>=1000,'price_usd'] = df[df['price_usd']<1000]['price_usd'].mean()

    rate = ['comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate', 'comp5_rate', 
        'comp6_rate', 'comp7_rate', 'comp8_rate'] 
        
    inv = ['comp1_inv', 'comp2_inv', 'comp3_inv', 'comp4_inv', 'comp5_inv', 'comp6_inv',  
        'comp7_inv', 'comp8_inv'] 
    
    diff = ['comp1_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff', 
        'comp4_rate_percent_diff', 'comp5_rate_percent_diff', 'comp6_rate_percent_diff', 
        'comp7_rate_percent_diff', 'comp8_rate_percent_diff'] 

    df['rate_sum'] = df[rate].sum(axis=1)
    df['inv_sum'] = df[inv].sum(axis=1)
    df['diff_mean'] = np.log(df[diff]).mean(axis=1)
    df['rate_abs'] = df[rate].min(axis=1)
    df['inv_abs'] = df[inv].min(axis=1)

    df.drop(rate + inv + diff, axis = 1, inplace = True)
    df.drop('date_time', axis =1, inplace = True)
    
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype('float32')
        elif df[col].dtype == np.int64:
            df[col] = df[col].astype('int32')
    return df

if len(sys.argv) > 2: 
    file_in = sys.argv[1]
    file_out = sys.argv[2]
else:
    print('Please specify filename in and filename out')
    quit()

to_clean = pd.read_csv(file_in)
clean = impute(to_clean)
clean.to_csv(file_out, index = False)
