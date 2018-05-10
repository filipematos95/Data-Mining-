import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('c:/Users/b_daan/Desktop/dm/data/training_set_VU_DM_2014.csv')


def inpute(df):

    # visitor_hist_starrating # zero -> mean 
    df[df['visitor_hist_starrating']==0]['visitor_hist_starrating'] = df['visitor_hist_starrating'].mean()
    df['visitor_hist_starrating'] = df['visitor_hist_starrating'].fillna(df['visitor_hist_starrating'].mean())
    
    # visitor_hist_adr_usd # nan, zero ->  median
    df[df['visitor_hist_adr_usd']==0]['visitor_hist_adr_usd'] = df['visitor_hist_adr_usd'].median()
    df['visitor_hist_adr_usd'] = df['visitor_hist_adr_usd'].fillna(df['visitor_hist_adr_usd'].median())
    
    # srch_query_affinity_score # nan -> mean
    df['srch_query_affinity_score'] = df['srch_query_affinity_score'].fillna(df['srch_query_affinity_score'].mean())
    
    # orig_destination_distance # zero -> mean
    df['orig_destination_distance'] = df['orig_destination_distance'].fillna(df['orig_destination_distance'].mean())

    rate = ['comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate', 'comp5_rate', 
        'comp6_rate', 'comp7_rate', 'comp8_rate'] 
        
    inv = ['comp1_inv', 'comp2_inv', 'comp3_inv', 'comp4_inv', 'comp5_inv', 'comp6_inv',  
        'comp7_inv', 'comp8_inv'] 
    
    diff = ['comp1_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff', 
        'comp4_rate_percent_diff', 'comp5_rate_percent_diff', 'comp6_rate_percent_diff', 
        'comp7_rate_percent_diff', 'comp8_rate_percent_diff'] 

    df['rate_sum'] = df[rate].sum(axis=1)
    df['inv_sum'] = df[inv].sum(axis=1)
    df['diff_mean'] = df[diff].mean(axis=1)
    df['rate_abs'] = df[rate].min(axis=1)
    df['inv_abs'] = df[inv].min(axis=1)

    df.drop(rate + inv + diff, axis = 1, inplace = True)
    return df
   
 
test =  inpute(train)

#test.isnull().sum()
test.to_csv('c:/Users/b_daan/Desktop/dm/data/clean.csv', index = False)

